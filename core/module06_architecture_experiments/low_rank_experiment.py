"""Optuna search configuration and runner for factorized low-rank fusion."""
from __future__ import annotations

from core.module06_architecture_experiments.architecture_experiment_runner import (
    run_architecture_optuna,
    run_architecture_experiment,
)
from .low_rank_model import LowRankMultimodalFusion


DEFAULT_LOW_RANK_CONFIG = {
    "hidden_dim": 512,
    "fusion_dim": 512,
    "rank": 8,
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
DEFAULT_LOW_RANK_N_TRIALS = 50
DEFAULT_LOW_RANK_SEARCH_SPACE = {
    "hidden_dim": {"type": "int", "low": 64, "high": 2048, "step": 64},
    "fusion_dim": {"type": "int", "low": 64, "high": 2048, "step": 64},
    "rank": {"type": "int", "low": 2, "high": 64, "step": 2},
    "head_dim": {"type": "int", "low": 64, "high": 512, "step": 64},
    "dropout": {"type": "float", "low": 0.005, "high": 0.4},
    "learning_rate": {"type": "float", "low": 1e-6, "high": 2e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
}


def get_default_low_rank_config():
    """Return an independent, notebook-editable copy of low-rank settings."""
    return dict(DEFAULT_LOW_RANK_CONFIG)


def get_default_low_rank_search_space():
    return {key: dict(spec) for key, spec in DEFAULT_LOW_RANK_SEARCH_SPACE.items()}


def _build_low_rank_model(dimensions, config):
    return LowRankMultimodalFusion(
        dna_dim=dimensions["dna"],
        protein_dim=dimensions["protein"],
        biological_dim=dimensions["bio"] + dimensions["geometry"],
        hidden_dim=int(config["hidden_dim"]),
        fusion_dim=int(config["fusion_dim"]),
        rank=int(config["rank"]),
        head_dim=int(config["head_dim"]),
        dropout=float(config["dropout"]),
    )


def run_low_rank_experiment(context, low_rank_config=None, display_fn=None):
    """Train, validate, checkpoint, and test the low-rank fusion model."""
    config = dict(DEFAULT_LOW_RANK_CONFIG if low_rank_config is None else low_rank_config)
    variant = f"factorized_three_modal_rank_{int(config['rank'])}"
    return run_architecture_experiment(
        context=context,
        experiment_config=config,
        model_builder=_build_low_rank_model,
        architecture="LowRankFusion",
        variant=variant,
        experiment_key="low_rank",
        source="module06_low_rank",
        display_fn=display_fn,
    )


def run_low_rank_optuna_experiment(context, base_config=None, search_space=None,
                                   n_trials=None, display_fn=None):
    config = dict(DEFAULT_LOW_RANK_CONFIG if base_config is None else base_config)
    space = get_default_low_rank_search_space() if search_space is None else search_space
    return run_architecture_optuna(
        context=context,
        base_config=config,
        search_space=space,
        n_trials=DEFAULT_LOW_RANK_N_TRIALS if n_trials is None else int(n_trials),
        model_builder=_build_low_rank_model,
        architecture="LowRankFusion",
        variant_builder=lambda cfg: f"factorized_three_modal_rank_{int(cfg['rank'])}_dim_{int(cfg['fusion_dim'])}",
        experiment_key="low_rank",
        source="module06_low_rank",
        display_fn=display_fn,
    )
