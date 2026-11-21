"""Tri-modal cross-attention over variant-centered FM windows and feature tokens."""
from __future__ import annotations

import torch
from torch import nn

from core.module06_architecture_experiments.architecture_experiment_runner import (
    run_architecture_optuna,
)
from core.module06_architecture_experiments.token_topk_experiment import (
    _VariantCenteredWindow,
)
from core.module06_architecture_experiments.transformer_3modal_experiment import (
    _ScalarFeatureTokenizer,
)


DEFAULT_CROSS_ATTENTION_3MODAL_CONFIG = {
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
DEFAULT_CROSS_ATTENTION_3MODAL_SEARCH_SPACE = {
    "k_radius": {"type": "categorical", "choices": [16]},
    "model_dim": {"type": "categorical", "choices": [512]},
    "num_heads": {"type": "categorical", "choices": [4, 8]},
    "num_layers": {"type": "int", "low": 3, "high": 3},
    "ffn_multiplier": {"type": "categorical", "choices": [4, 8]},
    "pooling": {"type": "categorical", "choices": ["mean"]},
    "dropout": {"type": "float", "low": 0.05, "high": 0.3},
    "attention_dropout": {"type": "float", "low": 0.02, "high": 0.3},
    "feature_dropout": {"type": "float", "low": 0.05, "high": 0.3},
    "learning_rate": {"type": "float", "low": 1e-6, "high": 2e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 0.1, "log": True},
}
DEFAULT_CROSS_ATTENTION_3MODAL_N_TRIALS = 40


class _CrossAttentionBlock(nn.Module):
    """One modality queries the concatenated tokens of the other two modalities."""

    def __init__(self, model_dim, num_heads, ffn_multiplier, dropout, attention_dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(model_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * ffn_multiplier, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query_tokens, context_tokens, context_mask):
        update, _ = self.cross_attention(
            self.query_norm(query_tokens),
            self.context_norm(context_tokens),
            self.context_norm(context_tokens),
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        query_tokens = query_tokens + self.attention_dropout(update)
        query_tokens = query_tokens + self.ffn(self.ffn_norm(query_tokens))
        return query_tokens


class CrossAttentionThreeModalities(nn.Module):
    """Parallel DNA, protein, and bio/geometry cross-attention branches."""

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
        self.dna_window = _VariantCenteredWindow(dna_dim, model_dim, dropout)
        self.protein_window = _VariantCenteredWindow(protein_dim, model_dim, dropout)
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

    def encode_features(self, v_dna, v_prot, bio_features, geom_features=None,
                        token_dna=None, token_dna_mask=None, token_dna_variant_index=None,
                        token_protein=None, token_protein_mask=None,
                        token_protein_variant_index=None):
        if token_dna is None or token_protein is None:
            raise ValueError("CrossAttentionThreeModalities requires both FM token caches.")
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
            # Compute each update from the same layer input; no modality is updated first.
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

        dna_vector = self._pool(dna, dna_mask, self.pooling, self.pool_scores["dna"])
        protein_vector = self._pool(
            protein, protein_mask, self.pooling, self.pool_scores["protein"]
        )
        bio_vector = self._pool(bio, bio_mask, self.pooling, self.pool_scores["bio"])
        fused = self.output_norm(torch.cat((dna_vector, protein_vector, bio_vector), dim=1))
        return fused

    def forward(self, v_dna, v_prot, bio_features, geom_features=None,
                token_dna=None, token_dna_mask=None, token_dna_variant_index=None,
                token_protein=None, token_protein_mask=None,
                token_protein_variant_index=None):
        features = self.encode_features(
            v_dna, v_prot, bio_features, geom_features,
            token_dna, token_dna_mask, token_dna_variant_index,
            token_protein, token_protein_mask, token_protein_variant_index,
        )
        return self.classifier(features).squeeze(-1)


def _builder(dimensions, config):
    return CrossAttentionThreeModalities(
        dna_dim=dimensions["token_dna"],
        protein_dim=dimensions["token_protein"],
        bio_dim=dimensions["bio"],
        geom_dim=dimensions["geometry"],
        config=config,
    )


def run_cross_attention_3modal_optuna(
    context, base_config=None, search_space=None, n_trials=None, display_fn=None,
    experiment_key="cross_attention_3modal_seed_42",
):
    """Tune tri-modal cross-attention using Module 6's shared persistence flow."""
    if context.get("token_embedding_cache_index") is None:
        raise KeyError(
            "token_embedding_cache_index is required. Run the shared token extraction cell first."
        )
    base = dict(
        DEFAULT_CROSS_ATTENTION_3MODAL_CONFIG if base_config is None else base_config
    )
    space = dict(
        DEFAULT_CROSS_ATTENTION_3MODAL_SEARCH_SPACE
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
        "architecture": "parallel_tri_modal_cross_attention_dna_protein_bio_geometry",
        "k_semantics": "k_tokens_per_side; window_length=2*k+1",
    }
    return run_architecture_optuna(
        context=run_context,
        base_config=base,
        search_space=space,
        n_trials=(
            DEFAULT_CROSS_ATTENTION_3MODAL_N_TRIALS
            if n_trials is None else int(n_trials)
        ),
        model_builder=_builder,
        architecture="CrossAttention_ThreeModalities_VariantWindow",
        variant_builder=lambda cfg: (
            f"radius_{cfg['k_radius']}_window_{2 * int(cfg['k_radius']) + 1}_"
            f"layers_{cfg['num_layers']}_dim_{cfg['model_dim']}_pool_{cfg['pooling']}"
        ),
        experiment_key=experiment_key,
        source="module06_cross_attention_three_modalities",
        display_fn=display_fn,
        manifest_architecture="parallel_tri_modal_cross_attention_dna_protein_bio_geometry",
    )
