"""Configuration and experiment wrappers for Hadamard embedding fusion."""
from __future__ import annotations

from core.module06_architecture_experiments.architecture_experiment_runner import (
    run_architecture_grid,
)
from .hadamard_model import HadamardEmbeddingFusion


DEFAULT_HADAMARD_CONFIG = {
    "hidden_dim": 512,
    "fusion_dim": 512,
    "bio_hidden_dim": 256,
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
DEFAULT_HADAMARD_GRID = {
    "hidden_dim": [512, 256],
    "fusion_dim": [512, 256],
    "dropout": [0.2, 0.1],
    "learning_rate": [1e-4, 5e-5],
}


def get_default_hadamard_config():
    return dict(DEFAULT_HADAMARD_CONFIG)


def get_default_hadamard_grid():
    return {key: list(values) for key, values in DEFAULT_HADAMARD_GRID.items()}


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


def run_hadamard_grid_experiment(context, base_config=None, search_space=None, display_fn=None):
    """Grid search dimensions, dropout, and learning rate for Hadamard fusion."""
    config = dict(DEFAULT_HADAMARD_CONFIG if base_config is None else base_config)
    grid = get_default_hadamard_grid() if search_space is None else search_space
    return run_architecture_grid(
        context=context,
        base_config=config,
        search_space=grid,
        model_builder=_build_hadamard_model,
        architecture="HadamardFusion",
        variant_builder=lambda _cfg: "projected_pairwise_hadamard_with_skip",
        experiment_key="hadamard",
        source="module06_hadamard",
        display_fn=display_fn,
    )
