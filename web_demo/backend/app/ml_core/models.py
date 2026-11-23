"""Nạp Foundation Model dạng Masked LM (NT-v1 500M, ESM2 650M) và ánh xạ vị trí đột biến -> token.

Giữ nguyên các shim tương thích để remote code (trust_remote_code) đời cũ chạy được trên transformers mới.
"""
from __future__ import annotations

import logging
import types
from typing import List, Sequence

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

logger = logging.getLogger("app.ml_core.models")

_LEGACY_CONFIG_DEFAULTS = {
    "is_decoder": False,
    "add_cross_attention": False,
    "chunk_size_feed_forward": 0,
    "position_embedding_type": "absolute",
    "use_cache": False,
}


# --------------------------------------------------------------------------------------------
# Shim tương thích transformers
# --------------------------------------------------------------------------------------------
def _compat_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    if head_mask is None:
        return [None] * num_hidden_layers
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError("head_mask should be 1D or 2D")
    head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
    if is_attention_chunked:
        head_mask = head_mask.unsqueeze(-1)
    return head_mask


def _compat_get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
    if dtype is None:
        dtype = next(self.parameters()).dtype
    if device is None:
        device = attention_mask.device
    if attention_mask.dim() == 3:
        extended = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        if getattr(self.config, "is_decoder", False):
            from transformers.modeling_utils import ModuleUtilsMixin

            extended = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(input_shape, attention_mask, device)
        else:
            extended = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"Wrong attention mask shape: input_shape={input_shape}, attention_mask={attention_mask.shape}")
    extended = extended.to(dtype=dtype)
    return (1.0 - extended) * torch.finfo(dtype).min


def _shim_pruneable_heads() -> bool:
    import transformers.pytorch_utils as pu

    if hasattr(pu, "find_pruneable_heads_and_indices"):
        return False
    try:
        from transformers.modeling_utils import find_pruneable_heads_and_indices

        pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        return True
    except ImportError:
        pass

    def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        heads = set(heads) - already_pruned_heads
        mask = torch.ones(n_heads, head_size)
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        return heads, torch.arange(len(mask))[mask].long()

    pu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices
    return True


def _shim_legacy_config() -> bool:
    installed = False
    targets = []
    try:
        from transformers.models.esm.configuration_esm import EsmConfig

        targets.append(EsmConfig)
    except ImportError:
        pass
    from transformers.configuration_utils import PretrainedConfig

    targets.append(PretrainedConfig)
    for cls in targets:
        for attr, value in _LEGACY_CONFIG_DEFAULTS.items():
            if not hasattr(cls, attr):
                setattr(cls, attr, value)
                installed = True
    return installed


def _shim_pretrained_model() -> bool:
    from transformers.modeling_utils import PreTrainedModel

    installed = False
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
        installed = True
    if not hasattr(PreTrainedModel, "get_head_mask"):
        PreTrainedModel.get_head_mask = _compat_get_head_mask
        installed = True
    return installed


# (chuỗi nhận diện lỗi, shim tương ứng)
_SHIMS = (
    ("find_pruneable_heads_and_indices", _shim_pruneable_heads),
    ("'EsmConfig' object has no attribute", _shim_legacy_config),
    ("all_tied_weights_keys", _shim_pretrained_model),
)


def _patch_remote_esm_instance(model) -> None:
    """Remote EsmModel đời cũ không kế thừa đủ mixin của transformers mới."""
    esm_core = getattr(model, "esm", None)
    if esm_core is None:
        return
    cls = esm_core.__class__
    for name, fn in (("get_head_mask", _compat_get_head_mask),
                     ("get_extended_attention_mask", _compat_get_extended_attention_mask)):
        if not hasattr(cls, name):
            setattr(cls, name, fn)
        if not hasattr(esm_core, name):
            setattr(esm_core, name, types.MethodType(fn, esm_core))


# --------------------------------------------------------------------------------------------
# Model manager
# --------------------------------------------------------------------------------------------
class BioModelManager:
    """Tokenizer + Masked LM trên thiết bị đích (fp16 trên CUDA, fp32 trên CPU)."""

    def __init__(self, model_id: str, device: str = "cuda", cuda_dtype: str = "fp16"):
        self.model_id = model_id
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable; loading %s on CPU", model_id)
            device = "cpu"
        self.device = str(device)
        # half precision không ổn định trên CPU -> CPU luôn fp32
        use_half = self.device.startswith("cuda") and cuda_dtype == "fp16"
        self.dtype = torch.float16 if use_half else torch.float32
        self._announced: set = set()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=True)
        except Exception as exc:  # noqa: BLE001 - một số remote tokenizer không có bản fast
            logger.warning("Fast tokenizer unavailable for %s (%s); using slow tokenizer", self.model_id, exc)
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "<pad>"
        return tokenizer

    def _load_model(self):
        logger.info("Loading %s (%s) on %s", self.model_id, str(self.dtype).replace("torch.", ""), self.device)
        applied = set()
        while True:
            try:
                model = AutoModelForMaskedLM.from_pretrained(self.model_id, dtype=torch.float16, trust_remote_code=True)
                break
            except Exception as exc:
                message = str(exc)
                shim = next((fn for marker, fn in _SHIMS if marker in message and fn not in applied), None)
                if shim is None:
                    raise
                logger.info("Applying transformers compatibility shim %s", shim.__name__)
                shim()
                applied.add(shim)
        _patch_remote_esm_instance(model)
        return model.to(device=self.device, dtype=self.dtype).eval()

    # ----------------------------------------------------------------------------------------
    def _infer_variant_indices_from_token_diffs(self, input_ids_ref, input_ids_alt, attention_mask=None) -> List[int]:
        """DNA k-mer không có offset_mapping: vị trí đột biến = giữa cụm token khác nhau giữa ref/alt."""
        special_ids = set(self.tokenizer.all_special_ids or [])
        indices = []
        for i in range(input_ids_ref.shape[0]):
            valid_len = int(attention_mask[i].sum().item()) if attention_mask is not None else int(input_ids_ref.shape[1])
            ref_tokens = input_ids_ref[i, :valid_len].tolist()
            alt_tokens = input_ids_alt[i, :valid_len].tolist()
            diff = [j for j, (r, a) in enumerate(zip(ref_tokens, alt_tokens))
                    if r != a and r not in special_ids and a not in special_ids]
            if not diff:
                raise ValueError("No differing token between ref/alt DNA sequences (tokenizer has no offset_mapping).")
            indices.append(diff[len(diff) // 2])
        return indices

    def _announce_strategy_once(self, seq_type: str, strategy: str) -> None:
        if (seq_type, strategy) not in self._announced:
            self._announced.add((seq_type, strategy))
            logger.info("Token mapping model=%s seq_type=%s strategy=%s", self.model_id, seq_type, strategy)

    def get_variant_token_index(
        self,
        tokenized_ref,
        seq_type: str,
        target_char_positions: Sequence[int],
        tokenized_alt=None,
    ) -> List[int]:
        """Chỉ số token chứa vị trí đột biến cho từng mẫu trong batch."""
        batch_size = tokenized_ref["input_ids"].shape[0]
        positions = [int(p) for p in target_char_positions]
        if len(positions) != batch_size:
            raise ValueError(f"{len(positions)} variant positions for a batch of {batch_size}.")

        if "offset_mapping" in tokenized_ref:
            strategy = "offset_mapping"
            indices = []
            for i, pos in enumerate(positions):
                offsets = tokenized_ref["offset_mapping"][i].tolist()
                idx = next((t for t, (start, end) in enumerate(offsets) if start <= pos < end), -1)
                if idx < 0:
                    raise ValueError(f"Cannot map char_pos={pos} to a token index (sample {i}).")
                indices.append(idx)
        elif seq_type == "dna":
            if tokenized_alt is None:
                raise ValueError(f"{self.model_id} has no offset_mapping; tokenized_alt is required for DNA.")
            strategy = "dna_token_diff_fallback"
            indices = self._infer_variant_indices_from_token_diffs(
                tokenized_ref["input_ids"], tokenized_alt["input_ids"], tokenized_ref.get("attention_mask")
            )
        else:
            # Protein char-level (ESM): token = ký tự + số special token ở đầu chuỗi.
            strategy = "protein_char_fallback"
            special = set(self.tokenizer.all_special_ids)
            indices = []
            for i, pos in enumerate(positions):
                ids = tokenized_ref["input_ids"][i].tolist()
                lead = 0
                while lead < len(ids) and ids[lead] in special:
                    lead += 1
                indices.append(pos + lead)

        attention_mask = tokenized_ref.get("attention_mask")
        for i, idx in enumerate(indices):
            valid_len = int(attention_mask[i].sum().item()) if attention_mask is not None else int(tokenized_ref["input_ids"].shape[1])
            if not 0 <= idx < valid_len:
                raise ValueError(f"Variant token index {idx} outside valid token length {valid_len} (sample {i}).")
        self._announce_strategy_once(seq_type, strategy)
        return indices

    @property
    def mask_token_id(self) -> int:
        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            raise ValueError(f"Tokenizer of {self.model_id} has no mask_token_id.")
        return int(mask_id)
