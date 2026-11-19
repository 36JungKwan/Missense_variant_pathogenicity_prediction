"""Tri-modal cross-attention using tokenwise alternate-minus-reference inputs."""
from __future__ import annotations

import torch
from torch import nn

from core.module06_architecture_experiments.architecture_experiment_runner import (
    run_architecture_optuna,
)
from core.module06_architecture_experiments.cross_attention_3modal_experiment import (
    _CrossAttentionBlock,
)
from core.module06_architecture_experiments.transformer_3modal_experiment import (
    _ScalarFeatureTokenizer,
)


DEFAULT_CROSS_ATTENTION_3MODAL_TOKEN_DIFF_CONFIG = {
    "k_radius": 8,
    "model_dim": 256,
    "num_heads": 8,
    "num_layers": 2,
    "ffn_multiplier": 4,
    "pooling": "attention",
    "dropout": 0.2,
    "attention_dropout": 0.1,
    "feature_dropout": 0.05,
    "max_epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "early_stop_patience": 6,
    "lr_factor": 0.5,
    "lr_patience": 3,
    "seed": 42,
}

# Keep the optimization range aligned with the existing variant-window study.
DEFAULT_CROSS_ATTENTION_3MODAL_TOKEN_DIFF_SEARCH_SPACE = {
    "k_radius": {"type": "categorical", "choices": [16]},
    "model_dim": {"type": "categorical", "choices": [512]},
    "num_heads": {"type": "categorical", "choices": [4]},
    "num_layers": {"type": "int", "low": 3, "high": 3},
    "ffn_multiplier": {"type": "categorical", "choices": [4]},
    "pooling": {"type": "categorical", "choices": ["mean"]},
    "dropout": {"type": "float", "low": 0.27, "high": 0.27},
    "attention_dropout": {"type": "float", "low": 0.0924, "high": 0.0924},
    "feature_dropout": {"type": "float", "low": 0.215, "high": 0.215},
    "learning_rate": {
        "type": "float", "low": 0.000011, "high": 0.000011, "log": True,
    },
    "weight_decay": {
        "type": "float", "low": 0.000398, "high": 0.000398, "log": True,
    },
}
DEFAULT_CROSS_ATTENTION_3MODAL_TOKEN_DIFF_N_TRIALS = 10


class _VariantCenteredTokenDifference(nn.Module):
    """Align ref/alt token windows and project (alt - ref) at each offset."""

    def __init__(self, input_dim, model_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(input_dim, model_dim)
        self.relative_position_embedding = nn.Embedding(129, model_dim)
        self.variant_embedding = nn.Parameter(torch.zeros(model_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.variant_embedding, std=0.02)

    def forward(self, tokens, mask, variant_indices, k_radius, modality_embedding):
        batch, alleles, length, hidden = tokens.shape
        if alleles != 2:
            raise ValueError(f"Expected ref/alt allele axis of length 2, got {alleles}")

        offsets = torch.arange(-int(k_radius), int(k_radius) + 1, device=tokens.device)
        positions = variant_indices.unsqueeze(-1) + offsets.view(1, 1, -1)
        in_bounds = (positions >= 0) & (positions < length)
        safe_positions = positions.clamp(0, max(length - 1, 0))
        gather_positions = safe_positions.unsqueeze(-1).expand(-1, -1, -1, hidden)
        local_tokens = tokens.gather(2, gather_positions)
        local_masks = mask.gather(2, safe_positions) & in_bounds

        ref_tokens, alt_tokens = local_tokens[:, 0], local_tokens[:, 1]
        ref_mask, alt_mask = local_masks[:, 0], local_masks[:, 1]
        ref_tokens = ref_tokens * ref_mask.unsqueeze(-1).to(ref_tokens.dtype)
        alt_tokens = alt_tokens * alt_mask.unsqueeze(-1).to(alt_tokens.dtype)
        token_delta = (alt_tokens - ref_tokens).to(self.projection.weight.dtype)
        # A tokenwise difference is defined only where both allele windows have
        # a valid token at the same variant-relative offset.
        window_mask = ref_mask & alt_mask
        token_delta = token_delta * window_mask.unsqueeze(-1).to(token_delta.dtype)

        projected = self.projection(token_delta)
        relative_ids = (offsets + 64).clamp(
            min=0, max=self.relative_position_embedding.num_embeddings - 1
        )
        projected = projected + self.relative_position_embedding(relative_ids).unsqueeze(0)
        center_mask = offsets.eq(0).view(1, -1, 1)
        projected = projected + center_mask.to(projected.dtype) * self.variant_embedding
        projected = projected + modality_embedding.view(1, 1, -1)
        return self.dropout(projected), window_mask


class CrossAttentionThreeModalitiesTokenDiff(nn.Module):
    """DNA/protein token deltas plus the unchanged tri-modal attention path."""

    def __init__(self, dna_dim, protein_dim, bio_dim, geom_dim, config):
        super().__init__()
        model_dim = int(config["model_dim"])
        self.k_radius = int(config["k_radius"])
        dropout = float(config["dropout"])
        feature_dropout = float(config["feature_dropout"])

        self.dna_modality = nn.Parameter(torch.zeros(model_dim))
        self.protein_modality = nn.Parameter(torch.zeros(model_dim))
        self.bio_modality = nn.Parameter(torch.zeros(model_dim))
        self.geometry_modality = nn.Parameter(torch.zeros(model_dim))
        self.dna_window = _VariantCenteredTokenDifference(dna_dim, model_dim, dropout)
        self.protein_window = _VariantCenteredTokenDifference(
            protein_dim, model_dim, dropout
        )
        self.bio_tokenizer = _ScalarFeatureTokenizer(bio_dim, model_dim, feature_dropout)
        self.geom_tokenizer = _ScalarFeatureTokenizer(geom_dim, model_dim, feature_dropout)

        self.cross_layers = nn.ModuleList()
        for _ in range(int(config["num_layers"])):
            self.cross_layers.append(nn.ModuleDict({
                modality: _CrossAttentionBlock(
                    model_dim=model_dim,
                    num_heads=int(config["num_heads"]),
                    ffn_multiplier=int(config["ffn_multiplier"]),
                    dropout=dropout,
                    attention_dropout=float(config["attention_dropout"]),
                )
                for modality in ("dna", "protein", "bio")
            }))

        self.pooling = str(config["pooling"])
        self.pool_scores = nn.ModuleDict({
            modality: nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
            for modality in ("dna", "protein", "bio")
        })
        self.output_norm = nn.LayerNorm(model_dim * 3)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    @staticmethod
    def _pool(tokens, mask, mode, scorer):
        if mode == "attention":
            scores = scorer(tokens).squeeze(-1).masked_fill(~mask, -torch.inf)
            weights = torch.softmax(scores, dim=1)
            return torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        if mode == "mean":
            valid = mask.unsqueeze(-1).to(tokens.dtype)
            return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        raise ValueError(f"Unsupported Cross-Attention pooling mode: {mode}")

    def forward(self, v_dna, v_prot, bio_features, geom_features=None,
                token_dna=None, token_dna_mask=None, token_dna_variant_index=None,
                token_protein=None, token_protein_mask=None,
                token_protein_variant_index=None):
        if token_dna is None or token_protein is None:
            raise ValueError(
                "CrossAttentionThreeModalitiesTokenDiff requires both FM token caches."
            )
        dna, dna_mask = self.dna_window(
            token_dna, token_dna_mask, token_dna_variant_index,
            self.k_radius, self.dna_modality,
        )
        protein, protein_mask = self.protein_window(
            token_protein, token_protein_mask, token_protein_variant_index,
            self.k_radius, self.protein_modality,
        )
        bio = self.bio_tokenizer(bio_features) + self.bio_modality.view(1, 1, -1)
        bio_mask = torch.ones(bio.shape[:2], dtype=torch.bool, device=bio.device)
        if geom_features is not None:
            geometry = self.geom_tokenizer(geom_features) + self.geometry_modality.view(1, 1, -1)
            bio = torch.cat((bio, geometry), dim=1)
            bio_mask = torch.cat((
                bio_mask,
                torch.ones(geometry.shape[:2], dtype=torch.bool, device=geometry.device),
            ), dim=1)

        for layer in self.cross_layers:
            old_dna, old_protein, old_bio = dna, protein, bio
            dna = layer["dna"](
                old_dna,
                torch.cat((old_protein, old_bio), dim=1),
                torch.cat((protein_mask, bio_mask), dim=1),
            )
            protein = layer["protein"](
                old_protein,
                torch.cat((old_dna, old_bio), dim=1),
                torch.cat((dna_mask, bio_mask), dim=1),
            )
            bio = layer["bio"](
                old_bio,
                torch.cat((old_dna, old_protein), dim=1),
                torch.cat((dna_mask, protein_mask), dim=1),
            )

        vectors = [
            self._pool(dna, dna_mask, self.pooling, self.pool_scores["dna"]),
            self._pool(protein, protein_mask, self.pooling, self.pool_scores["protein"]),
            self._pool(bio, bio_mask, self.pooling, self.pool_scores["bio"]),
        ]
        fused = self.output_norm(torch.cat(vectors, dim=1))
        return self.classifier(fused).squeeze(-1)


def _builder(dimensions, config):
    return CrossAttentionThreeModalitiesTokenDiff(
        dna_dim=dimensions["token_dna"],
        protein_dim=dimensions["token_protein"],
        bio_dim=dimensions["bio"],
        geom_dim=dimensions["geometry"],
        config=config,
    )


def run_cross_attention_3modal_token_diff_optuna(
    context, base_config=None, search_space=None, n_trials=None,
    display_fn=None, experiment_key="cross_attention_3modal_token_diff_seed_42",
):
    """Tune tri-modal cross-attention with tokenwise alt-minus-ref inputs."""
    if context.get("token_embedding_cache_index") is None:
        raise KeyError(
            "token_embedding_cache_index is required. Run the shared token extraction cell first."
        )
    base = dict(
        DEFAULT_CROSS_ATTENTION_3MODAL_TOKEN_DIFF_CONFIG
        if base_config is None else base_config
    )
    space = dict(
        DEFAULT_CROSS_ATTENTION_3MODAL_TOKEN_DIFF_SEARCH_SPACE
        if search_space is None else search_space
    )
    cache_index = context["token_embedding_cache_index"]
    run_context = dict(context)
    run_context["input_data_signature"] = {
        "token_cache_index_path": cache_index.get("index_path"),
        "token_cache_root": cache_index.get("cache_root"),
        "token_models": {
            "dna": "nt_v1_500m_dna_1001",
            "protein": "esm2_650m_prot_501",
        },
        "token_batch_size": context.get("token_batch_size"),
        "architecture": "parallel_tri_modal_cross_attention",
        "ref_alt_representation": "tokenwise_alt_minus_ref_aligned_by_variant_offset",
        "k_semantics": "k_tokens_per_side; window_length=2*k+1",
    }
    return run_architecture_optuna(
        context=run_context,
        base_config=base,
        search_space=space,
        n_trials=(
            DEFAULT_CROSS_ATTENTION_3MODAL_TOKEN_DIFF_N_TRIALS
            if n_trials is None else int(n_trials)
        ),
        model_builder=_builder,
        architecture="CrossAttention_ThreeModalities_TokenDiff",
        variant_builder=lambda cfg: (
            f"token_diff_radius_{cfg['k_radius']}_window_"
            f"{2 * int(cfg['k_radius']) + 1}_layers_{cfg['num_layers']}_"
            f"dim_{cfg['model_dim']}_pool_{cfg['pooling']}"
        ),
        experiment_key=experiment_key,
        source="module06_cross_attention_three_modalities_token_diff",
        display_fn=display_fn,
        manifest_architecture="parallel_tri_modal_cross_attention_tokenwise_alt_minus_ref",
    )
