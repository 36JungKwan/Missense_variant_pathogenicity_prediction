"""FiLM configuration and wrapper for the shared Module 6 trainer."""
from __future__ import annotations

from core.module06_architecture_experiments.architecture_experiment_runner import (
    run_architecture_grid,
    run_architecture_experiment,
)
from .film_model import BiologyConditionedFiLM


DEFAULT_FILM_CONFIG = {
    "hidden_dim": 512,
    "bio_hidden_dim": 512,
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
DEFAULT_FILM_GRID = {
    "hidden_dim": [512, 256],
    "dropout": [0.2, 0.1],
    "learning_rate": [1e-4, 5e-5],
    "weight_decay": [1e-3, 1e-4],
}


def get_default_film_config():
    """Return an independent, notebook-editable copy of FiLM settings."""
    return dict(DEFAULT_FILM_CONFIG)


def get_default_film_grid():
    """Return a copy of the small FiLM grid, with baseline values first."""
    return {key: list(values) for key, values in DEFAULT_FILM_GRID.items()}


def _build_film_model(dimensions, config):
    return BiologyConditionedFiLM(
        dna_dim=dimensions["dna"],
        protein_dim=dimensions["protein"],
        biological_dim=dimensions["bio"] + dimensions["geometry"],
        hidden_dim=int(config["hidden_dim"]),
        bio_hidden_dim=int(config["bio_hidden_dim"]),
        head_dim=int(config["head_dim"]),
        dropout=float(config["dropout"]),
    )


def run_film_experiment(context, film_config=None, display_fn=None):
    """Run FiLM using the shared checkpointing and evaluation workflow."""
    config = dict(DEFAULT_FILM_CONFIG if film_config is None else film_config)
    return run_architecture_experiment(
        context=context,
        experiment_config=config,
        model_builder=_build_film_model,
        architecture="FiLM",
        variant="bio_conditioned_residual_film",
        experiment_key="film",
        source="module06_film",
        display_fn=display_fn,
        # Preserve the existing completed FiLM stage signature for resume.
        manifest_architecture="biology_conditioned_film",
    )


def run_film_grid_experiment(context, base_config=None, search_space=None, display_fn=None):
    """Grid search a compact set of FiLM capacity and regularization settings."""
    config = dict(DEFAULT_FILM_CONFIG if base_config is None else base_config)
    grid = get_default_film_grid() if search_space is None else search_space
    return run_architecture_grid(
        context=context,
        base_config=config,
        search_space=grid,
        model_builder=_build_film_model,
        architecture="FiLM",
        variant_builder=lambda _config: "bio_conditioned_residual_film",
        experiment_key="film",
        source="module06_film",
        display_fn=display_fn,
        manifest_architecture="biology_conditioned_film",
    )
