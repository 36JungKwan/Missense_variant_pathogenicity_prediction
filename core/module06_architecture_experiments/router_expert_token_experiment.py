"""Router + Experts using variant-centered or full FM token sequences."""
from __future__ import annotations

import torch
from torch import nn

from core.module06_architecture_experiments.architecture_experiment_runner import (
    run_architecture_optuna,
)
from core.module06_architecture_experiments.router_expert_model import (
    RouterExpertCorrection,
)
from core.module06_architecture_experiments.token_topk_experiment import (
    _VariantCenteredWindow,
)


DEFAULT_ROUTER_EXPERT_TOKEN_CONFIG = {
    "token_dim": 256,
    "embedding_dim": 256,
    "expert_hidden_dim": 256,
    "router_hidden_dim": 128,
    "correction_hidden_dim": 256,
    "router_temperature": 1.0,
    "beta_init": 0.0,
    "dropout": 0.2,
    "max_epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "early_stop_patience": 6,
    "lr_factor": 0.5,
    "lr_patience": 3,
    "seed": 42,
}
DEFAULT_ROUTER_EXPERT_TOKEN_SEARCH_SPACE = {
    "token_dim": {"type": "categorical", "choices": [512]},
    "embedding_dim": {"type": "categorical", "choices": [256, 512]},
    "expert_hidden_dim": {"type": "categorical", "choices": [256, 512]},
    "router_hidden_dim": {"type": "categorical", "choices": [64, 128]},
    "correction_hidden_dim": {"type": "categorical", "choices": [128, 256]},
    "router_temperature": {"type": "float", "low": 0.1, "high": 2.0},
    "beta_init": {"type": "categorical", "choices": [0.0, 0.01, 0.05, 0.1, 0.25]},
    "dropout": {"type": "float", "low": 0.05, "high": 0.4},
    "learning_rate": {"type": "float", "low": 1e-6, "high": 2e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 0.2, "log": True},
}
DEFAULT_ROUTER_EXPERT_TOKEN_N_TRIALS = 10
TOKEN_SCOPES = ("centered_16", "full_tokens")


class _TokenExpertRouterModel(nn.Module):
    """Token-level DNA/protein experts plus the existing bio/router/correction head."""

    def __init__(self, dna_dim, protein_dim, bio_dim, geom_dim, config, scope):
        super().__init__()
        if scope not in TOKEN_SCOPES:
            raise ValueError(f"Unsupported token scope: {scope}")
        self.scope = scope
        self.local_radius = 16
        token_dim = int(config["token_dim"])
        dropout = float(config["dropout"])

        self.dna_modality = nn.Parameter(torch.zeros(token_dim))
        self.protein_modality = nn.Parameter(torch.zeros(token_dim))
        self.dna_token_projection = _VariantCenteredWindow(dna_dim, token_dim, dropout)
        self.protein_token_projection = _VariantCenteredWindow(protein_dim, token_dim, dropout)
        self.dna_pool_score = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))
        self.protein_pool_score = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))

        self.router_experts = RouterExpertCorrection(
            dna_dim=token_dim,
            protein_dim=token_dim,
            biological_dim=int(bio_dim) + int(geom_dim),
            embedding_dim=int(config["embedding_dim"]),
            expert_hidden_dim=int(config["expert_hidden_dim"]),
            router_hidden_dim=int(config["router_hidden_dim"]),
            correction_hidden_dim=int(config["correction_hidden_dim"]),
            router_temperature=float(config["router_temperature"]),
            beta_init=float(config["beta_init"]),
            dropout=dropout,
        )

    @staticmethod
    def _masked_attention_pool(tokens, mask, scorer):
        scores = scorer(tokens).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)

    def forward(self, v_dna, v_prot, bio_features, geom_features=None,
                token_dna=None, token_dna_mask=None, token_dna_variant_index=None,
                token_protein=None, token_protein_mask=None,
                token_protein_variant_index=None):
        if token_dna is None or token_protein is None:
            raise ValueError("Token Router + Experts requires both FM token caches.")

        if self.scope == "centered_16":
            dna_radius = protein_radius = self.local_radius
        else:
            # Radius L-1 covers every valid token on either side of the variant.
            dna_radius = max(int(token_dna.shape[2]) - 1, 0)
            protein_radius = max(int(token_protein.shape[2]) - 1, 0)

        dna_tokens, dna_mask = self.dna_token_projection(
            token_dna, token_dna_mask, token_dna_variant_index,
            dna_radius, self.dna_modality,
        )
        protein_tokens, protein_mask = self.protein_token_projection(
            token_protein, token_protein_mask, token_protein_variant_index,
            protein_radius, self.protein_modality,
        )
        dna_vector = self._masked_attention_pool(
            dna_tokens, dna_mask, self.dna_pool_score
        )
        protein_vector = self._masked_attention_pool(
            protein_tokens, protein_mask, self.protein_pool_score
        )
        return self.router_experts(
            dna_vector,
            protein_vector,
            bio_features,
            geom_features,
        ).squeeze(-1)


def _model_builder_for_scope(scope):
    def builder(dimensions, config):
        return _TokenExpertRouterModel(
            dna_dim=dimensions["token_dna"],
            protein_dim=dimensions["token_protein"],
            bio_dim=dimensions["bio"],
            geom_dim=dimensions["geometry"],
            config=config,
            scope=scope,
        )
    return builder


def run_router_expert_token_optuna(
    context, scope, base_config=None, search_space=None, n_trials=None,
    display_fn=None, experiment_key=None,
):
    """Tune one token scope with the shared Module 6 checkpoint/resume outputs."""
    if scope not in TOKEN_SCOPES:
        raise ValueError(f"scope must be one of {TOKEN_SCOPES}, got {scope!r}")
    if context.get("token_embedding_cache_index") is None:
        raise KeyError(
            "token_embedding_cache_index is required. Run the shared token extraction cell first."
        )
    config = dict(
        DEFAULT_ROUTER_EXPERT_TOKEN_CONFIG if base_config is None else base_config
    )
    space = dict(
        DEFAULT_ROUTER_EXPERT_TOKEN_SEARCH_SPACE
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
        "architecture": "three_expert_router_with_token_level_dna_protein_experts",
        "token_scope": scope,
        "k_semantics": "centered_16 means 16 tokens per side plus variant (33 total); full_tokens uses all valid positions",
    }
    key = experiment_key or f"router_expert_{scope}_seed_{int(config['seed'])}"
    scope_label = "k16_window33" if scope == "centered_16" else "full_tokens"
    return run_architecture_optuna(
        context=run_context,
        base_config=config,
        search_space=space,
        n_trials=(
            DEFAULT_ROUTER_EXPERT_TOKEN_N_TRIALS
            if n_trials is None else int(n_trials)
        ),
        model_builder=_model_builder_for_scope(scope),
        architecture="RouterExpert_TokenInput",
        variant_builder=lambda cfg: (
            f"{scope_label}_token_dim_{cfg['token_dim']}_"
            f"embedding_{cfg['embedding_dim']}_beta_{cfg['beta_init']}"
        ),
        experiment_key=key,
        source="module06_router_expert_token_inputs",
        display_fn=display_fn,
        manifest_architecture=f"router_expert_{scope}_token_inputs_with_three_experts_and_joint_correction",
    )
