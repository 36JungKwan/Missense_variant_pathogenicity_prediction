"""Optuna configuration and wrapper for Router + Experts (proposal 3)."""
from __future__ import annotations

from core.module06_architecture_experiments.architecture_experiment_runner import (
    run_architecture_optuna,
)
from .router_expert_model import RouterExpertCorrection


DEFAULT_ROUTER_EXPERT_CONFIG = {
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
DEFAULT_ROUTER_EXPERT_N_TRIALS = 100
DEFAULT_ROUTER_EXPERT_SEARCH_SPACE = {
    "embedding_dim": {"type": "int", "low": 128, "high": 4096, "step": 64},
    "expert_hidden_dim": {"type": "int", "low": 64, "high": 1024, "step": 64},
    "router_hidden_dim": {"type": "int", "low": 16, "high": 768, "step": 16},
    "correction_hidden_dim": {"type": "int", "low": 64, "high": 1024, "step": 64},
    "router_temperature": {"type": "float", "low": 0.05, "high": 2.0},
    "beta_init": {"type": "categorical", "choices": [0.0, 0.01, 0.05, 0.1, 0.25]},
    "dropout": {"type": "float", "low": 0.005, "high": 0.4},
    "learning_rate": {"type": "float", "low": 1e-6, "high": 2e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
}


def get_default_router_expert_config():
    return dict(DEFAULT_ROUTER_EXPERT_CONFIG)


def get_default_router_expert_search_space():
    return {key: dict(spec) for key, spec in DEFAULT_ROUTER_EXPERT_SEARCH_SPACE.items()}


def _build_router_expert_model(dimensions, config):
    return RouterExpertCorrection(
        dna_dim=dimensions["dna"],
        protein_dim=dimensions["protein"],
        biological_dim=dimensions["bio"] + dimensions["geometry"],
        embedding_dim=int(config["embedding_dim"]),
        expert_hidden_dim=int(config["expert_hidden_dim"]),
        router_hidden_dim=int(config["router_hidden_dim"]),
        correction_hidden_dim=int(config["correction_hidden_dim"]),
        router_temperature=float(config["router_temperature"]),
        beta_init=float(config["beta_init"]),
        dropout=float(config["dropout"]),
    )


def run_router_expert_optuna(context, base_config=None, search_space=None,
                             n_trials=None, display_fn=None):
    config = dict(
        DEFAULT_ROUTER_EXPERT_CONFIG if base_config is None else base_config
    )
    space = get_default_router_expert_search_space() if search_space is None else search_space
    return run_architecture_optuna(
        context=context,
        base_config=config,
        search_space=space,
        n_trials=DEFAULT_ROUTER_EXPERT_N_TRIALS if n_trials is None else int(n_trials),
        model_builder=_build_router_expert_model,
        architecture="RouterExpert",
        variant_builder=lambda _cfg: "three_expert_softmax_router_joint_logit_correction",
        experiment_key="router_expert",
        source="module06_router_expert",
        display_fn=display_fn,
    )
