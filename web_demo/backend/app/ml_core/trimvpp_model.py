"""TriMVPP (Tri-Modal Variant Pathogenicity Predictor): cross-attention trên cửa sổ 33 token ref/alt.

Phép tính giữ nguyên từ repo huấn luyện (module06_architecture_experiments):
  * `_VariantCenteredWindow`          - token_topk_experiment.py
  * `_ScalarFeatureTokenizer`         - transformer_3modal_experiment.py
  * `_CrossAttentionBlock`            - cross_attention_3modal_experiment.py
  * `TriMVPP` = `CrossAttentionThreeModalities` (đủ 3 nhánh) và `CrossAttentionSubsetModel` (ablation):
    khi bật đủ 3 nhánh hai lớp này tính giống nhau nên gộp lại; tên tham số giữ nguyên để nạp strict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn

BRANCH_ORDER = ("dna", "protein", "bio")


@dataclass(frozen=True)
class TriMVPPConfig:
    """Siêu tham số chung của cả 7 model (_resume_manifest.json); dropout không ảnh hưởng suy luận."""
    k_radius: int = 16
    model_dim: int = 512
    num_heads: int = 8
    num_layers: int = 3
    ffn_multiplier: int = 4
    token_dim: int = 1280
    bio_dim: int = 20
    geom_dim: int = 8
    dropout: float = 0.1955626033738947
    attention_dropout: float = 0.18776012701077605
    feature_dropout: float = 0.2507543060224074


class _VariantCenteredWindow(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, dropout: float):
        super().__init__()
        self.projection = nn.Linear(input_dim * 2, model_dim)
        self.relative_position_embedding = nn.Embedding(129, model_dim)
        self.variant_embedding = nn.Parameter(torch.zeros(model_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, mask, variant_indices, k_radius, modality_embedding):
        _, alleles, length, hidden = tokens.shape
        if alleles != 2:
            raise ValueError(f"Expected ref/alt allele axis of length 2, got {alleles}")
        offsets = torch.arange(-int(k_radius), int(k_radius) + 1, device=tokens.device)
        positions = variant_indices.unsqueeze(-1) + offsets.view(1, 1, -1)
        in_bounds = (positions >= 0) & (positions < length)
        safe_positions = positions.clamp(0, max(length - 1, 0))
        local_tokens = tokens.gather(2, safe_positions.unsqueeze(-1).expand(-1, -1, -1, hidden))
        local_masks = mask.gather(2, safe_positions) & in_bounds

        ref_tokens, alt_tokens = local_tokens[:, 0], local_tokens[:, 1]
        ref_mask, alt_mask = local_masks[:, 0], local_masks[:, 1]
        ref_tokens = ref_tokens * ref_mask.unsqueeze(-1).to(ref_tokens.dtype)
        alt_tokens = alt_tokens * alt_mask.unsqueeze(-1).to(alt_tokens.dtype)
        paired = torch.cat((ref_tokens, alt_tokens), dim=-1).to(self.projection.weight.dtype)
        window_mask = ref_mask | alt_mask

        projected = self.projection(paired)
        relative_ids = (offsets + 64).clamp(min=0, max=self.relative_position_embedding.num_embeddings - 1)
        projected = projected + self.relative_position_embedding(relative_ids).unsqueeze(0)
        center_mask = offsets.eq(0).view(1, -1, 1)
        projected = projected + center_mask.to(projected.dtype) * self.variant_embedding
        projected = projected + modality_embedding.view(1, 1, -1)
        return self.dropout(projected), window_mask


class _ScalarFeatureTokenizer(nn.Module):
    """Mỗi đặc trưng vô hướng đã chuẩn hoá thành một token: x_j * W_j + e_j."""

    def __init__(self, n_features: int, model_dim: int, dropout: float):
        super().__init__()
        self.value_vectors = nn.Parameter(torch.empty(n_features, model_dim))
        self.feature_vectors = nn.Parameter(torch.empty(n_features, model_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        tokens = values.unsqueeze(-1) * self.value_vectors.unsqueeze(0)
        return self.dropout(tokens + self.feature_vectors.unsqueeze(0))


class _CrossAttentionBlock(nn.Module):
    """Một nhánh truy vấn chuỗi token ghép của các nhánh còn lại."""

    def __init__(self, cfg: TriMVPPConfig):
        super().__init__()
        dim = cfg.model_dim
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=cfg.num_heads, dropout=cfg.attention_dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(cfg.dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * cfg.ffn_multiplier), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(dim * cfg.ffn_multiplier, dim), nn.Dropout(cfg.dropout),
        )

    def forward(self, query_tokens, context_tokens, context_mask, return_attention=False):
        context = self.context_norm(context_tokens)
        update, weights = self.cross_attention(
            self.query_norm(query_tokens), context, context,
            key_padding_mask=~context_mask, need_weights=bool(return_attention),
            average_attn_weights=False,
        )
        query_tokens = query_tokens + self.attention_dropout(update)
        query_tokens = query_tokens + self.ffn(self.ffn_norm(query_tokens))
        return (query_tokens, weights) if return_attention else query_tokens


class TriMVPP(nn.Module):
    """Các nhánh cross-attention song song; nhánh tắt thì không tồn tại trong mạng. Một nhánh: không có attention."""

    def __init__(self, active_branches: Sequence[str], cfg: TriMVPPConfig = TriMVPPConfig()):
        super().__init__()
        self.active_branches = tuple(b for b in BRANCH_ORDER if b in active_branches)
        if not self.active_branches:
            raise ValueError("At least one branch must be active.")
        dim = cfg.model_dim
        self.k_radius = cfg.k_radius
        if "dna" in self.active_branches:
            self.dna_modality = nn.Parameter(torch.zeros(dim))
            self.dna_window = _VariantCenteredWindow(cfg.token_dim, dim, cfg.dropout)
        if "protein" in self.active_branches:
            self.protein_modality = nn.Parameter(torch.zeros(dim))
            self.protein_window = _VariantCenteredWindow(cfg.token_dim, dim, cfg.dropout)
        if "bio" in self.active_branches:
            self.bio_modality = nn.Parameter(torch.zeros(dim))
            self.geometry_modality = nn.Parameter(torch.zeros(dim))
            self.bio_tokenizer = _ScalarFeatureTokenizer(cfg.bio_dim, dim, cfg.feature_dropout)
            self.geom_tokenizer = _ScalarFeatureTokenizer(cfg.geom_dim, dim, cfg.feature_dropout)

        n_layers = cfg.num_layers if len(self.active_branches) > 1 else 0
        self.cross_layers = nn.ModuleList(
            nn.ModuleDict({branch: _CrossAttentionBlock(cfg) for branch in self.active_branches})
            for _ in range(n_layers)
        )
        # Pooling = mean nên pool_scores không tham gia tính toán; giữ lại để nạp checkpoint strict.
        self.pool_scores = nn.ModuleDict({
            branch: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1)) for branch in self.active_branches
        })
        width = dim * len(self.active_branches)
        self.output_norm = nn.LayerNorm(width)
        self.classifier = nn.Sequential(
            nn.Linear(width, dim * 2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        token_dna: Optional[torch.Tensor] = None,  # (B, 2, 33, 1280), trục 1: ref, alt
        mask_dna: Optional[torch.Tensor] = None,  # (B, 2, 33) bool
        token_protein: Optional[torch.Tensor] = None,
        mask_protein: Optional[torch.Tensor] = None,
        bio: Optional[torch.Tensor] = None,  # (B, 20)
        geom: Optional[torch.Tensor] = None,  # (B, 8)
        return_attention: bool = False,
    ):
        """Trả logit (B,). Với return_attention=True trả thêm {"layer_{l}_{query}": (B, heads, Lq, Lk)};
        trục key là các nhánh còn lại ghép theo BRANCH_ORDER (giống notebook 19 của nhóm)."""
        states, masks = {}, {}
        for branch, tokens, mask in (("dna", token_dna, mask_dna), ("protein", token_protein, mask_protein)):
            if branch not in self.active_branches:
                continue
            # Cửa sổ đã cắt sẵn quanh biến thể nên token đột biến luôn ở chỉ số k_radius.
            index = torch.full((tokens.shape[0], 2), self.k_radius, dtype=torch.long, device=tokens.device)
            states[branch], masks[branch] = getattr(self, f"{branch}_window")(
                tokens, mask, index, self.k_radius, getattr(self, f"{branch}_modality"))
        if "bio" in self.active_branches:
            states["bio"] = torch.cat((
                self.bio_tokenizer(bio) + self.bio_modality.view(1, 1, -1),
                self.geom_tokenizer(geom) + self.geometry_modality.view(1, 1, -1),
            ), dim=1)
            masks["bio"] = torch.ones(states["bio"].shape[:2], dtype=torch.bool, device=bio.device)

        attention = {}
        for layer_index, layer in enumerate(self.cross_layers, start=1):
            # Mọi nhánh đọc cùng trạng thái đầu vào của lớp; không nhánh nào được cập nhật trước.
            updated = {}
            for branch in self.active_branches:
                result = layer[branch](
                    states[branch],
                    torch.cat([states[other] for other in self.active_branches if other != branch], dim=1),
                    torch.cat([masks[other] for other in self.active_branches if other != branch], dim=1),
                    return_attention=return_attention,
                )
                if return_attention:
                    updated[branch], attention[f"layer_{layer_index}_{branch}"] = result
                else:
                    updated[branch] = result
            states = updated

        pooled = []
        for branch in self.active_branches:
            weight = masks[branch].unsqueeze(-1).to(states[branch].dtype)
            pooled.append((states[branch] * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0))
        logits = self.classifier(self.output_norm(torch.cat(pooled, dim=1))).squeeze(-1)
        return (logits, attention) if return_attention else logits
