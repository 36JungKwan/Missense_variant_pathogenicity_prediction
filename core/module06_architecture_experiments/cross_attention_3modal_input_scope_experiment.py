"""Cross-attention over three modalities with center or full FM token inputs."""
from __future__ import annotations

import torch
from torch import nn

from core.module06_architecture_experiments.architecture_experiment_runner import (
    run_architecture_optuna,
)
from core.module06_architecture_experiments.cross_attention_3modal_experiment import (
    _CrossAttentionBlock,
)
from core.module06_architecture_experiments.token_topk_experiment import (
    _VariantCenteredWindow,
)
from core.module06_architecture_experiments.transformer_3modal_experiment import (
    _ScalarFeatureTokenizer,
)
from core.module06_architecture_experiments.transformer_3modal_input_scope_experiment import (
    _FullPairedTokenProjection,
)


DEFAULT_CROSS_ATTENTION_3MODAL_SCOPE_CONFIG = {
    "model_dim": 128,
    "num_heads": 4,
    "num_layers": 1,
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
DEFAULT_CROSS_ATTENTION_3MODAL_SCOPE_SEARCH_SPACE = {
    "model_dim": {"type": "categorical", "choices": [128, 256]},
    "num_heads": {"type": "categorical", "choices": [4, 8]},
    "num_layers": {"type": "int", "low": 1, "high": 3},
    "ffn_multiplier": {"type": "categorical", "choices": [2, 4, 8]},
    "pooling": {"type": "categorical", "choices": ["mean", "attention"]},
    "dropout": {"type": "float", "low": 0.05, "high": 0.5},
    "attention_dropout": {"type": "float", "low": 0.0, "high": 0.4},
    "feature_dropout": {"type": "float", "low": 0.0, "high": 0.3},
    "learning_rate": {"type": "float", "low": 1e-6, "high": 2e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 0.2, "log": True},
}
DEFAULT_CROSS_ATTENTION_3MODAL_SCOPE_N_TRIALS = 30
INPUT_SCOPES = ("center", "full_tokens")


class CrossAttentionThreeModalInputScope(nn.Module):
    """Tri-modal cross-attention with either one center token or all FM tokens."""

    def __init__(self, dna_dim, protein_dim, bio_dim, geom_dim, config, input_scope):
        super().__init__()
        if input_scope not in INPUT_SCOPES:
            raise ValueError(f"input_scope must be one of {INPUT_SCOPES}, got {input_scope!r}")
        self.input_scope = input_scope
        model_dim = int(config["model_dim"])
        dropout = float(config["dropout"])
        feature_dropout = float(config["feature_dropout"])

        if input_scope == "center":
            self.dna_projection = _VariantCenteredWindow(dna_dim, model_dim, dropout)
            self.protein_projection = _VariantCenteredWindow(protein_dim, model_dim, dropout)
        else:
            self.dna_projection = _FullPairedTokenProjection(dna_dim, model_dim, dropout)
            self.protein_projection = _FullPairedTokenProjection(protein_dim, model_dim, dropout)

        self.dna_modality = nn.Parameter(torch.zeros(model_dim))
        self.protein_modality = nn.Parameter(torch.zeros(model_dim))
        self.bio_modality = nn.Parameter(torch.zeros(model_dim))
        self.geometry_modality = nn.Parameter(torch.zeros(model_dim))
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

    def _project_sequences(self, token_dna, token_dna_mask, token_dna_variant_index,
                           token_protein, token_protein_mask,
                           token_protein_variant_index):
        if self.input_scope == "center":
            dna, dna_mask = self.dna_projection(
                token_dna, token_dna_mask, token_dna_variant_index,
                0, self.dna_modality,
            )
            protein, protein_mask = self.protein_projection(
                token_protein, token_protein_mask, token_protein_variant_index,
                0, self.protein_modality,
            )
        else:
            dna, dna_mask = self.dna_projection(
                token_dna, token_dna_mask, token_dna_variant_index,
                self.dna_modality,
            )
            protein, protein_mask = self.protein_projection(
                token_protein, token_protein_mask, token_protein_variant_index,
                self.protein_modality,
            )
        return dna, dna_mask, protein, protein_mask

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
            raise ValueError("CrossAttentionThreeModalInputScope requires both FM token caches.")
        dna, dna_mask, protein, protein_mask = self._project_sequences(
            token_dna, token_dna_mask, token_dna_variant_index,
            token_protein, token_protein_mask, token_protein_variant_index,
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


def _builder_for_scope(input_scope):
    def builder(dimensions, config):
        return CrossAttentionThreeModalInputScope(
            dna_dim=dimensions["token_dna"],
            protein_dim=dimensions["token_protein"],
            bio_dim=dimensions["bio"],
            geom_dim=dimensions["geometry"],
            config=config,
            input_scope=input_scope,
        )
    return builder


def run_cross_attention_3modal_input_scope_optuna(
    context, input_scope, base_config=None, search_space=None, n_trials=None,
    display_fn=None, experiment_key=None,
):
    """Tune center-token or full-token tri-modal cross-attention separately."""
    if input_scope not in INPUT_SCOPES:
        raise ValueError(f"input_scope must be one of {INPUT_SCOPES}, got {input_scope!r}")
    if context.get("token_embedding_cache_index") is None:
        raise KeyError(
            "token_embedding_cache_index is required. Run the shared token extraction cell first."
        )
    config = dict(
        DEFAULT_CROSS_ATTENTION_3MODAL_SCOPE_CONFIG if base_config is None else base_config
    )
    space = dict(
        DEFAULT_CROSS_ATTENTION_3MODAL_SCOPE_SEARCH_SPACE
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
        "architecture": "three_modal_cross_attention_input_scope_comparison",
        "input_scope": input_scope,
        "ref_alt_representation": "paired_ref_alt_token_embeddings; no delta-only compression",
    }
    key = experiment_key or f"cross_attention_3modal_{input_scope}_seed_{int(config['seed'])}"
    scope_label = "center_token" if input_scope == "center" else "full_tokens_unpooled"
    return run_architecture_optuna(
        context=run_context,
        base_config=config,
        search_space=space,
        n_trials=(
            DEFAULT_CROSS_ATTENTION_3MODAL_SCOPE_N_TRIALS
            if n_trials is None else int(n_trials)
        ),
        model_builder=_builder_for_scope(input_scope),
        architecture=(
            "CrossAttention_ThreeModalities_CenterToken"
            if input_scope == "center"
            else "CrossAttention_ThreeModalities_FullTokens"
        ),
        variant_builder=lambda cfg: (
            f"{scope_label}_layers_{cfg['num_layers']}_dim_{cfg['model_dim']}_"
            f"pool_{cfg['pooling']}"
        ),
        experiment_key=key,
        source="module06_cross_attention_three_modal_input_scope",
        display_fn=display_fn,
        manifest_architecture=f"three_modal_cross_attention_{input_scope}_paired_ref_alt_tokens",
    )
