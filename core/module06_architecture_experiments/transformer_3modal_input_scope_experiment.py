"""Three-modal Transformer comparing center-token and full-token FM inputs."""
from __future__ import annotations

import math

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


DEFAULT_TRANSFORMER_3MODAL_SCOPE_CONFIG = {
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
DEFAULT_TRANSFORMER_3MODAL_SCOPE_SEARCH_SPACE = {
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
DEFAULT_TRANSFORMER_3MODAL_SCOPE_N_TRIALS = 30
INPUT_SCOPES = ("center", "full_tokens")


def _relative_sinusoidal_encoding(offsets, model_dim, dtype):
    positions = offsets.to(dtype=torch.float32).unsqueeze(1)
    dimensions = torch.arange(0, model_dim, 2, device=offsets.device, dtype=torch.float32)
    scales = torch.exp(dimensions * (-math.log(10000.0) / float(model_dim)))
    angles = positions * scales.unsqueeze(0)
    encoding = torch.zeros((len(offsets), model_dim), device=offsets.device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    if model_dim > 1:
        encoding[:, 1::2] = torch.cos(angles[:, :encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class _FullPairedTokenProjection(nn.Module):
    """Project every ref/alt token pair while aligning positions to the variant."""

    def __init__(self, input_dim, model_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(int(input_dim) * 2, model_dim)
        self.variant_embedding = nn.Parameter(torch.zeros(model_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.variant_embedding, std=0.02)

    def forward(self, tokens, mask, variant_indices, modality_embedding):
        batch, alleles, length, hidden = tokens.shape
        if alleles != 2:
            raise ValueError(f"Expected ref/alt allele axis of length 2, got {alleles}")
        source_positions = torch.arange(length, device=tokens.device).view(1, 1, -1)
        min_positions = torch.where(mask, source_positions, length).amin(dim=-1)
        max_positions = torch.where(mask, source_positions, -1).amax(dim=-1)
        max_left = int((variant_indices - min_positions).clamp_min(0).max().item())
        max_right = int((max_positions - variant_indices).clamp_min(0).max().item())
        offsets = torch.arange(-max_left, max_right + 1, device=tokens.device)
        positions = variant_indices.unsqueeze(-1) + offsets.view(1, 1, -1)
        in_bounds = (positions >= 0) & (positions < length)
        safe_positions = positions.clamp(0, max(length - 1, 0))
        gather_positions = safe_positions.unsqueeze(-1).expand(-1, -1, -1, hidden)
        paired_tokens = tokens.gather(2, gather_positions)
        paired_mask = mask.gather(2, safe_positions) & in_bounds

        ref_tokens, alt_tokens = paired_tokens[:, 0], paired_tokens[:, 1]
        ref_mask, alt_mask = paired_mask[:, 0], paired_mask[:, 1]
        ref_tokens = ref_tokens * ref_mask.unsqueeze(-1).to(ref_tokens.dtype)
        alt_tokens = alt_tokens * alt_mask.unsqueeze(-1).to(alt_tokens.dtype)
        paired = torch.cat((ref_tokens, alt_tokens), dim=-1).to(self.projection.weight.dtype)
        projected = self.projection(paired)
        projected = projected + _relative_sinusoidal_encoding(
            offsets, projected.shape[-1], projected.dtype
        ).unsqueeze(0)
        projected = projected + offsets.eq(0).view(1, -1, 1).to(projected.dtype) * self.variant_embedding
        projected = projected + modality_embedding.view(1, 1, -1)
        return self.dropout(projected), ref_mask | alt_mask


class TransformerThreeModalInputScope(nn.Module):
    """Joint Transformer with center-only or unpooled full FM token input."""

    def __init__(self, dna_dim, protein_dim, bio_dim, geom_dim, config, input_scope):
        super().__init__()
        if input_scope not in INPUT_SCOPES:
            raise ValueError(f"input_scope must be one of {INPUT_SCOPES}, got {input_scope!r}")
        self.input_scope = input_scope
        model_dim = int(config["model_dim"])
        dropout = float(config["dropout"])
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
        feature_dropout = float(config["feature_dropout"])
        self.bio_tokenizer = _ScalarFeatureTokenizer(bio_dim, model_dim, feature_dropout)
        self.geom_tokenizer = _ScalarFeatureTokenizer(geom_dim, model_dim, feature_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=int(config["num_heads"]),
            dim_feedforward=model_dim * int(config["ffn_multiplier"]),
            dropout=float(config["attention_dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=int(config["num_layers"])
        )
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
            raise ValueError("Transformer3ModalInputScope requires both FM token caches.")
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
                token_dna, token_dna_mask, token_dna_variant_index, self.dna_modality
            )
            protein, protein_mask = self.protein_projection(
                token_protein, token_protein_mask, token_protein_variant_index,
                self.protein_modality,
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


def _builder_for_scope(input_scope):
    def builder(dimensions, config):
        return TransformerThreeModalInputScope(
            dna_dim=dimensions["token_dna"],
            protein_dim=dimensions["token_protein"],
            bio_dim=dimensions["bio"],
            geom_dim=dimensions["geometry"],
            config=config,
            input_scope=input_scope,
        )
    return builder


def run_transformer_3modal_input_scope_optuna(
    context, input_scope, base_config=None, search_space=None, n_trials=None,
    display_fn=None, experiment_key=None,
):
    """Tune center-token or full-token three-modal Transformer separately."""
    if input_scope not in INPUT_SCOPES:
        raise ValueError(f"input_scope must be one of {INPUT_SCOPES}, got {input_scope!r}")
    if context.get("token_embedding_cache_index") is None:
        raise KeyError(
            "token_embedding_cache_index is required. Run the shared token extraction cell first."
        )
    config = dict(
        DEFAULT_TRANSFORMER_3MODAL_SCOPE_CONFIG if base_config is None else base_config
    )
    space = dict(
        DEFAULT_TRANSFORMER_3MODAL_SCOPE_SEARCH_SPACE
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
        "architecture": "three_modal_joint_transformer_input_scope_comparison",
        "input_scope": input_scope,
        "ref_alt_representation": "paired_ref_alt_token_embeddings; no delta-only compression",
    }
    key = experiment_key or f"transformer_3modal_{input_scope}_seed_{int(config['seed'])}"
    scope_label = "center_token" if input_scope == "center" else "full_tokens_unpooled"
    return run_architecture_optuna(
        context=run_context,
        base_config=config,
        search_space=space,
        n_trials=(
            DEFAULT_TRANSFORMER_3MODAL_SCOPE_N_TRIALS
            if n_trials is None else int(n_trials)
        ),
        model_builder=_builder_for_scope(input_scope),
        architecture="Transformer_ThreeModalities_InputScope",
        variant_builder=lambda cfg: (
            f"{scope_label}_layers_{cfg['num_layers']}_dim_{cfg['model_dim']}_"
            f"pool_{cfg['pooling']}"
        ),
        experiment_key=key,
        source="module06_transformer_three_modal_input_scope",
        display_fn=display_fn,
        manifest_architecture=f"three_modal_joint_transformer_{input_scope}_paired_ref_alt_tokens",
    )
