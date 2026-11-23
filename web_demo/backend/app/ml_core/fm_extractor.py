"""Trích xuất embedding center-pooling + LLR cho nhiều biến thể trong MỘT forward pass.

Mỗi biến thể đóng góp 3 chuỗi vào batch: [masked_ref, ref, alt]
  * masked_ref -> LLR = log p(alt) - log p(ref) tại token đột biến (Masked LM)
  * ref / alt  -> hidden state lớp cuối tại token đột biến (E_ref / E_alt)

Kết quả tương đương phiên bản gốc (3 forward riêng, batch=1, output_hidden_states=True):
hidden_states[-1] của encoder ESM chính là last_hidden_state (sau emb_layer_norm_after).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .models import BioModelManager

logger = logging.getLogger("app.ml_core.fm_extractor")


@dataclass
class EmbeddingBatch:
    center_ref: np.ndarray  # (N, D) float32 (đã lượng tử hoá fp16 như lúc train)
    center_alt: np.ndarray  # (N, D)
    llr: np.ndarray  # (N,)

    @staticmethod
    def concat(parts: Sequence["EmbeddingBatch"]) -> "EmbeddingBatch":
        return EmbeddingBatch(
            np.concatenate([p.center_ref for p in parts]),
            np.concatenate([p.center_alt for p in parts]),
            np.concatenate([p.llr for p in parts]),
        )


def _pad_to(tensor: torch.Tensor, length: int, value: int) -> torch.Tensor:
    return F.pad(tensor, (0, length - tensor.shape[1]), value=value) if tensor.shape[1] < length else tensor


class FeatureExtractor:
    def __init__(self, manager: BioModelManager):
        self.manager = manager
        self.tokenizer = manager.tokenizer
        self.model = manager.model
        self.device = manager.device
        self._base = self.model.base_model  # EsmForMaskedLM.esm
        self._lm_head = getattr(self.model, "lm_head", None)

    def _tokenize(self, sequences: List[str]):
        kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
        if self.tokenizer.is_fast:
            kwargs["return_offsets_mapping"] = True
        tokenized = self.tokenizer(sequences, **kwargs)
        if "attention_mask" not in tokenized:
            tokenized["attention_mask"] = torch.ones_like(tokenized["input_ids"])
        return tokenized

    @torch.inference_mode()
    def embed(self, ref_seqs: List[str], alt_seqs: List[str], centers: List[int], seq_type: str) -> EmbeddingBatch:
        try:
            return self._embed(ref_seqs, alt_seqs, centers, seq_type)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(ref_seqs) == 1:
                raise
            mid = len(ref_seqs) // 2
            logger.warning("CUDA OOM with %d variants per forward; splitting into %d + %d",
                           len(ref_seqs), mid, len(ref_seqs) - mid)
            return EmbeddingBatch.concat([
                self.embed(ref_seqs[:mid], alt_seqs[:mid], centers[:mid], seq_type),
                self.embed(ref_seqs[mid:], alt_seqs[mid:], centers[mid:], seq_type),
            ])

    def _embed(self, ref_seqs, alt_seqs, centers, seq_type) -> EmbeddingBatch:
        n = len(ref_seqs)
        tok_ref = self._tokenize(list(ref_seqs))
        tok_alt = self._tokenize(list(alt_seqs))
        var_idx = self.manager.get_variant_token_index(tok_ref, seq_type, centers, tokenized_alt=tok_alt)

        rows = torch.arange(n)
        idx = torch.tensor(var_idx)
        ref_ids = tok_ref["input_ids"][rows, idx]
        alt_ids = tok_alt["input_ids"][rows, idx]

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        length = max(tok_ref["input_ids"].shape[1], tok_alt["input_ids"].shape[1])
        ids_ref = _pad_to(tok_ref["input_ids"], length, pad_id)
        ids_alt = _pad_to(tok_alt["input_ids"], length, pad_id)
        att_ref = _pad_to(tok_ref["attention_mask"], length, 0)
        att_alt = _pad_to(tok_alt["attention_mask"], length, 0)
        ids_masked = ids_ref.clone()
        ids_masked[rows, idx] = self.manager.mask_token_id

        input_ids = torch.cat([ids_masked, ids_ref, ids_alt]).to(self.device)
        attention = torch.cat([att_ref, att_ref, att_alt]).to(self.device)
        hidden = self._base(input_ids=input_ids, attention_mask=attention)[0]  # (3N, L, D)

        idx_dev = idx.to(self.device)
        rows_dev = rows.to(self.device)
        h_masked = hidden[rows_dev, idx_dev]
        h_ref = hidden[n + rows_dev, idx_dev]
        h_alt = hidden[2 * n + rows_dev, idx_dev]

        if self._lm_head is not None:
            logits = self._lm_head(h_masked)
        else:  # model không tách lm_head: chạy lại phần masked qua model đầy đủ
            logits = self.model(input_ids=input_ids[:n], attention_mask=attention[:n]).logits[rows_dev, idx_dev]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        llr = log_probs[rows_dev, alt_ids.to(self.device)] - log_probs[rows_dev, ref_ids.to(self.device)]

        # Lượng tử hoá fp16 như pipeline huấn luyện (embedding được lưu dạng float16).
        def _np(t: torch.Tensor) -> np.ndarray:
            return t.float().cpu().numpy().astype(np.float16).astype(np.float32)

        return EmbeddingBatch(_np(h_ref), _np(h_alt), llr.cpu().numpy().astype(np.float32))
