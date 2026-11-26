"""Trích xuất cửa sổ 33 token (±16 quanh biến thể) của lớp ẩn cuối cho chuỗi ref và alt.

Khớp token_embedding_cache.py + window_token_cache.py của repo huấn luyện: hidden state lớp cuối ở fp16, vị trí
đột biến được ánh xạ riêng cho ref và alt, vị trí ngoài chuỗi được đánh dấu False trong mask.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from .models import BioModelManager

logger = logging.getLogger("app.ml_core.fm_extractor")

K_RADIUS = 16
WINDOW_TOKENS = 2 * K_RADIUS + 1
TOKEN_DIM = 1280


@dataclass
class TokenWindow:
    """Cửa sổ token của một biến thể trên một nhánh; trục 0: ref, alt."""
    hidden: np.ndarray  # (2, 33, 1280) float16
    mask: np.ndarray  # (2, 33) bool

    @property
    def center_ref(self) -> np.ndarray:
        return self.hidden[0, K_RADIUS]

    @property
    def center_alt(self) -> np.ndarray:
        return self.hidden[1, K_RADIUS]


class FeatureExtractor:
    def __init__(self, manager: BioModelManager):
        self.manager = manager
        self.tokenizer = manager.tokenizer
        self.device = manager.device
        self._base = manager.model.base_model  # EsmForMaskedLM.esm: output[0] = hidden_states[-1]

    def _tokenize(self, sequences: List[str]):
        kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
        if self.tokenizer.is_fast:
            kwargs["return_offsets_mapping"] = True
        tokenized = self.tokenizer(sequences, **kwargs)
        if "attention_mask" not in tokenized:
            tokenized["attention_mask"] = torch.ones_like(tokenized["input_ids"])
        return tokenized

    @torch.inference_mode()
    def embed(self, ref_seqs: List[str], alt_seqs: List[str], centers: List[int], seq_type: str) -> List[TokenWindow]:
        try:
            return self._embed(ref_seqs, alt_seqs, centers, seq_type)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(ref_seqs) == 1:
                raise
            mid = len(ref_seqs) // 2
            logger.warning("CUDA OOM with %d variants per forward; splitting into %d + %d",
                           len(ref_seqs), mid, len(ref_seqs) - mid)
        return (self.embed(ref_seqs[:mid], alt_seqs[:mid], centers[:mid], seq_type)
                + self.embed(ref_seqs[mid:], alt_seqs[mid:], centers[mid:], seq_type))

    def _windows(self, hidden: torch.Tensor, attention: torch.Tensor, indices: Sequence[int]):
        offsets = torch.arange(-K_RADIUS, K_RADIUS + 1, device=hidden.device)
        positions = torch.tensor(list(indices), device=hidden.device)[:, None] + offsets
        valid = (positions >= 0) & (positions < hidden.shape[1])
        positions = positions.clamp(0, hidden.shape[1] - 1)
        window = hidden.gather(1, positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        mask = attention.bool().gather(1, positions) & valid
        return window.to(torch.float16).cpu().numpy(), mask.cpu().numpy()

    def _embed(self, ref_seqs, alt_seqs, centers, seq_type) -> List[TokenWindow]:
        n = len(ref_seqs)
        tok_ref = self._tokenize(list(ref_seqs))
        tok_alt = self._tokenize(list(alt_seqs))
        # Tokenize có thể dịch/tách token khác nhau giữa hai allele nên mỗi allele có chỉ số riêng.
        idx_ref = self.manager.get_variant_token_index(tok_ref, seq_type, centers, tokenized_alt=tok_alt)
        idx_alt = self.manager.get_variant_token_index(tok_alt, seq_type, centers, tokenized_alt=tok_ref)

        def forward(ids: torch.Tensor, attention: torch.Tensor):
            attention = attention.to(self.device)
            return self._base(input_ids=ids.to(self.device), attention_mask=attention)[0], attention

        if tok_ref["input_ids"].shape == tok_alt["input_ids"].shape:  # gộp ref + alt vào một forward
            hidden, attention = forward(torch.cat([tok_ref["input_ids"], tok_alt["input_ids"]]),
                                        torch.cat([tok_ref["attention_mask"], tok_alt["attention_mask"]]))
            window, mask = self._windows(hidden, attention, list(idx_ref) + list(idx_alt))
            ref, alt, mask_ref, mask_alt = window[:n], window[n:], mask[:n], mask[n:]
        else:
            ref, mask_ref = self._windows(*forward(tok_ref["input_ids"], tok_ref["attention_mask"]), idx_ref)
            alt, mask_alt = self._windows(*forward(tok_alt["input_ids"], tok_alt["attention_mask"]), idx_alt)
        return [TokenWindow(np.stack([ref[i], alt[i]]), np.stack([mask_ref[i], mask_alt[i]])) for i in range(n)]
