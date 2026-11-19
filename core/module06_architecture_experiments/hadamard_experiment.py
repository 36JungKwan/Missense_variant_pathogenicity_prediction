"""Optuna configuration and experiment wrapper for Hadamard/RB-Net fusion."""
from __future__ import annotations

from core.module06_architecture_experiments.architecture_experiment_runner import (
    run_architecture_optuna,
)
from .hadamard_model import HadamardEmbeddingFusion


DEFAULT_HADAMARD_CONFIG = {
    "hidden_dim": 512,
    "fusion_dim": 512,
    "bio_hidden_dim": 128,
    "head_dim": 128,
    "dropout": 0.2,
    "max_epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "early_stop_patience": 6,
    "lr_factor": 0.5,
    "lr_patience": 3,
    "seed": 42,
}
DEFAULT_HADAMARD_N_TRIALS = 50
DEFAULT_HADAMARD_SEARCH_SPACE = {
    "hidden_dim": {"type": "int", "low": 64, "high": 1024, "step": 64},
    "fusion_dim": {"type": "int", "low": 64, "high": 1024, "step": 64},
    "bio_hidden_dim": {"type": "int", "low": 64, "high": 2048, "step": 64},
    "head_dim": {"type": "int", "low": 64, "high": 512, "step": 64},
    "dropout": {"type": "float", "low": 0.005, "high": 0.4},
    "learning_rate": {"type": "float", "low": 1e-6, "high": 2e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
}


def get_default_hadamard_config():
    return dict(DEFAULT_HADAMARD_CONFIG)


def get_default_hadamard_search_space():
    return {key: dict(spec) for key, spec in DEFAULT_HADAMARD_SEARCH_SPACE.items()}


def _build_hadamard_model(dimensions, config):
    return HadamardEmbeddingFusion(
        dna_dim=dimensions["dna"],
        protein_dim=dimensions["protein"],
        biological_dim=dimensions["bio"] + dimensions["geometry"],
        hidden_dim=int(config["hidden_dim"]),
        fusion_dim=int(config["fusion_dim"]),
        bio_hidden_dim=int(config["bio_hidden_dim"]),
        head_dim=int(config["head_dim"]),
        dropout=float(config["dropout"]),
    )


def run_hadamard_optuna_experiment(context, base_config=None, search_space=None,
                                   n_trials=None, display_fn=None):
    config = dict(DEFAULT_HADAMARD_CONFIG if base_config is None else base_config)
    space = get_default_hadamard_search_space() if search_space is None else search_space
    return run_architecture_optuna(
        context=context,
        base_config=config,
        search_space=space,
        n_trials=DEFAULT_HADAMARD_N_TRIALS if n_trials is None else int(n_trials),
        model_builder=_build_hadamard_model,
        architecture="HadamardFusion",
        variant_builder=lambda _cfg: "rbnet_hadamard_product_concat_projection",
        experiment_key="hadamard",
        source="module06_hadamard",
        display_fn=display_fn,
    )
