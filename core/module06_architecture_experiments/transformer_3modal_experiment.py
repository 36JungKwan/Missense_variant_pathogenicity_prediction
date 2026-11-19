"""Three-modality Transformer over DNA, protein, and biological feature tokens."""
from __future__ import annotations

from torch import nn
import torch

from core.module06_architecture_experiments.token_topk_experiment import (
    _VariantCenteredWindow,
)
from core.module06_architecture_experiments.architecture_experiment_runner import (
    run_architecture_optuna,
)


DEFAULT_TRANSFORMER_3MODAL_CONFIG = {
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
DEFAULT_TRANSFORMER_3MODAL_SEARCH_SPACE = {
    "k_radius": {"type": "categorical", "choices": [4, 8, 16]},
    "model_dim": {"type": "categorical", "choices": [128, 256, 512]},
    "num_heads": {"type": "categorical", "choices": [4, 8]},
    "num_layers": {"type": "int", "low": 1, "high": 4},
    "ffn_multiplier": {"type": "categorical", "choices": [2, 4, 8]},
    "pooling": {"type": "categorical", "choices": ["mean", "attention"]},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5},
    "attention_dropout": {"type": "float", "low": 0.0, "high": 0.4},
    "feature_dropout": {"type": "float", "low": 0.0, "high": 0.3},
    "learning_rate": {"type": "float", "low": 1e-6, "high": 2e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 0.2, "log": True},
}
DEFAULT_TRANSFORMER_3MODAL_N_TRIALS = 40


class _ScalarFeatureTokenizer(nn.Module):
    """Turn each normalized scalar feature into its own learned token."""

    def __init__(self, n_features, model_dim, dropout):
        super().__init__()
        self.n_features = int(n_features)
        self.value_vectors = nn.Parameter(torch.empty(self.n_features, model_dim))
        self.feature_vectors = nn.Parameter(torch.empty(self.n_features, model_dim))
        self.dropout = nn.Dropout(dropout)
        if self.n_features:
            nn.init.xavier_uniform_(self.value_vectors)
            nn.init.normal_(self.feature_vectors, std=0.02)

    def forward(self, values):
        if self.n_features == 0:
            return values.new_zeros((values.shape[0], 0, self.feature_vectors.shape[-1]))
        # x_j * W_j + e_j preserves feature identity while allowing the
        # Transformer to attend directly across bio/geometry measurements.
        tokens = values.unsqueeze(-1) * self.value_vectors.unsqueeze(0)
        tokens = tokens + self.feature_vectors.unsqueeze(0)
        return self.dropout(tokens)


class TransformerThreeModalities(nn.Module):
    """Joint self-attention over DNA, protein, and bio/geometry feature tokens."""

    def __init__(self, dna_dim, protein_dim, bio_dim, geom_dim, config):
        super().__init__()
        model_dim = int(config["model_dim"])
        dropout = float(config["dropout"])
        self.k_radius = int(config["k_radius"])
        self.dna_window = _VariantCenteredWindow(dna_dim, model_dim, dropout)
        self.protein_window = _VariantCenteredWindow(protein_dim, model_dim, dropout)
        self.dna_modality = nn.Parameter(torch.zeros(model_dim))
        self.protein_modality = nn.Parameter(torch.zeros(model_dim))
        self.bio_modality = nn.Parameter(torch.zeros(model_dim))
        self.geometry_modality = nn.Parameter(torch.zeros(model_dim))
        feature_dropout = float(config["feature_dropout"])
        self.bio_tokenizer = _ScalarFeatureTokenizer(bio_dim, model_dim, feature_dropout)
        self.geom_tokenizer = _ScalarFeatureTokenizer(geom_dim, model_dim, feature_dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=int(config["num_heads"]),
            dim_feedforward=model_dim * int(config["ffn_multiplier"]),
            dropout=float(config["attention_dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(config["num_layers"]))
        self.pooling = str(config["pooling"])
        self.pool_score = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
        self.output_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, max(model_dim // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(model_dim // 2, 32), 1),
        )

    def forward(self, v_dna, v_prot, bio_features, geom_features=None,
                token_dna=None, token_dna_mask=None, token_dna_variant_index=None,
                token_protein=None, token_protein_mask=None,
                token_protein_variant_index=None):
        if token_dna is None or token_protein is None:
            raise ValueError("TransformerThreeModalities requires both FM token caches.")
        dna, dna_mask = self.dna_window(
            token_dna, token_dna_mask, token_dna_variant_index,
            self.k_radius, self.dna_modality,
        )
        protein, protein_mask = self.protein_window(
            token_protein, token_protein_mask, token_protein_variant_index,
            self.k_radius, self.protein_modality,
        )
        bio = self.bio_tokenizer(bio_features) + self.bio_modality.view(1, 1, -1)
        if geom_features is None:
            geom = bio.new_zeros((bio.shape[0], 0, bio.shape[-1]))
        else:
            geom = self.geom_tokenizer(geom_features) + self.geometry_modality.view(1, 1, -1)

        tokens = torch.cat((dna, protein, bio, geom), dim=1)
        feature_mask = torch.ones(
            (tokens.shape[0], bio.shape[1] + geom.shape[1]),
            dtype=torch.bool,
            device=tokens.device,
        )
        valid_mask = torch.cat((dna_mask, protein_mask, feature_mask), dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=~valid_mask)
        if self.pooling == "attention":
            scores = self.pool_score(encoded).squeeze(-1).masked_fill(~valid_mask, -torch.inf)
            weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        elif self.pooling == "mean":
            valid = valid_mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        else:
            raise ValueError(f"Unsupported Transformer pooling mode: {self.pooling}")
        return self.classifier(self.output_norm(pooled)).squeeze(-1)


def _builder(dimensions, config):
    return TransformerThreeModalities(
        dna_dim=dimensions["token_dna"],
        protein_dim=dimensions["token_protein"],
        bio_dim=dimensions["bio"],
        geom_dim=dimensions["geometry"],
        config=config,
    )


def run_transformer_3modal_optuna(context, base_config=None, search_space=None,
                                  n_trials=None, display_fn=None,
                                  experiment_key="transformer_3modal_seed_42"):
    """Tune a three-modality Transformer with the common Module 6 resume/output flow."""
    if context.get("token_embedding_cache_index") is None:
        raise KeyError(
            "token_embedding_cache_index is required. Run the shared token extraction cell first."
        )
    base = dict(DEFAULT_TRANSFORMER_3MODAL_CONFIG if base_config is None else base_config)
    space = dict(
        DEFAULT_TRANSFORMER_3MODAL_SEARCH_SPACE if search_space is None else search_space
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
        "architecture": "joint_transformer_over_dna_protein_and_per_feature_bio_geometry_tokens",
    }
    return run_architecture_optuna(
        context=run_context,
        base_config=base,
        search_space=space,
        n_trials=(
            DEFAULT_TRANSFORMER_3MODAL_N_TRIALS
            if n_trials is None else int(n_trials)
        ),
        model_builder=_builder,
        architecture="Transformer_ThreeModalities",
        variant_builder=lambda cfg: (
            f"radius_{cfg['k_radius']}_window_{2 * int(cfg['k_radius']) + 1}_"
            f"layers_{cfg['num_layers']}_dim_{cfg['model_dim']}_pool_{cfg['pooling']}"
        ),
        experiment_key=experiment_key,
        source="module06_transformer_three_modalities",
        display_fn=display_fn,
        manifest_architecture="joint_transformer_dna_protein_bio_geometry_feature_tokens",
    )
