"""Configuration and runner wrapper for factorized low-rank fusion."""
from __future__ import annotations

from core.module06_architecture_experiments.architecture_experiment_runner import (
    run_architecture_grid,
    run_architecture_experiment,
)
from .low_rank_model import LowRankMultimodalFusion


DEFAULT_LOW_RANK_CONFIG = {
    "hidden_dim": 256,
    "fusion_dim": 256,
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
DEFAULT_LOW_RANK_GRID = {
    "rank": [8, 4],
    "fusion_dim": [256, 128],
    "dropout": [0.2, 0.1],
    "learning_rate": [1e-4, 5e-5],
}


def get_default_low_rank_config():
    """Return an independent, notebook-editable copy of low-rank settings."""
    return dict(DEFAULT_LOW_RANK_CONFIG)


def get_default_low_rank_grid():
    """Return a copy of the compact low-rank grid, baseline values first."""
    return {key: list(values) for key, values in DEFAULT_LOW_RANK_GRID.items()}


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


def run_low_rank_grid_experiment(context, base_config=None, search_space=None, display_fn=None):
    """Grid search low-rank dimension and regularization settings."""
    config = dict(DEFAULT_LOW_RANK_CONFIG if base_config is None else base_config)
    grid = get_default_low_rank_grid() if search_space is None else search_space
    return run_architecture_grid(
        context=context,
        base_config=config,
        search_space=grid,
        model_builder=_build_low_rank_model,
        architecture="LowRankFusion",
        variant_builder=lambda cfg: (
            f"factorized_three_modal_rank_{int(cfg['rank'])}_dim_{int(cfg['fusion_dim'])}"
        ),
        experiment_key="low_rank",
        source="module06_low_rank",
        display_fn=display_fn,
    )
