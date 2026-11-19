"""Optuna studies for the four original fusion families with separate ref/alt inputs."""
from __future__ import annotations

from copy import deepcopy

from core.module05v2_fusion_classifier.fusion_model import MultiStrategyFusionModel
from core.module06_architecture_experiments.architecture_experiment_runner import (
    run_architecture_optuna,
)


EMBEDDING_REPRESENTATION = "ref_alt_concat"
FUSION_STRATEGIES = {
    "concat": "Concat",
    "cross_attention": "CrossAttention",
    "transformer": "Transformer",
    "gating": "Gating",
}

DEFAULT_REF_ALT_CONFIG = {
    "fusion_dim": 512,
    "bio_upsample_dim": 128,
    "dropout": 0.2,
    "max_epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "early_stop_patience": 6,
    "lr_factor": 0.5,
    "lr_patience": 3,
    "seed": 42,
}
DEFAULT_REF_ALT_N_TRIALS = 40

# All fusion widths are divisible by 8 because the original attention and
# Transformer implementations use eight heads.
DEFAULT_REF_ALT_SEARCH_SPACE = {
    "fusion_dim": {"type": "categorical", "choices": [128, 256, 512, 768, 1024]},
    "bio_upsample_dim": {"type": "categorical", "choices": [64, 128, 256, 512]},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5},
    "learning_rate": {"type": "float", "low": 1e-6, "high": 2e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 0.2, "log": True},
}


def get_default_ref_alt_config():
    return dict(DEFAULT_REF_ALT_CONFIG)


def get_default_ref_alt_search_spaces():
    """Return a separate editable search-space dict for every fusion family."""
    return {name: deepcopy(DEFAULT_REF_ALT_SEARCH_SPACE) for name in FUSION_STRATEGIES}


def _builder(strategy):
    def build(dimensions, config):
        return MultiStrategyFusionModel(
            dna_in_dim=dimensions["dna"],
            prot_in_dim=dimensions["protein"],
            active_modalities=dimensions["active_modalities"],
            fusion_dim=int(config["fusion_dim"]),
            bio_upsample_dim=int(config["bio_upsample_dim"]),
            bio_in_dim=dimensions["bio"],
            geom_in_dim=dimensions["geometry"],
            fusion_strategy=strategy,
            dropout=float(config["dropout"]),
        )
    return build


def run_ref_alt_fusion_optuna(context, base_config=None, search_spaces=None,
                              n_trials=None, display_fn=None):
    """Run four independent Optuna studies, each on A/B and all configured tests."""
    base = get_default_ref_alt_config() if base_config is None else dict(base_config)
    spaces = get_default_ref_alt_search_spaces() if search_spaces is None else search_spaces
    missing = sorted(set(FUSION_STRATEGIES) - set(spaces))
    if missing:
        raise KeyError(f"Missing search spaces for fusion strategies: {missing}")

    results = {}
    for strategy, label in FUSION_STRATEGIES.items():
        results[strategy] = run_ref_alt_strategy_optuna(
            context=context,
            strategy=strategy,
            base_config=base,
            search_space=spaces[strategy],
            n_trials=n_trials,
            display_fn=display_fn,
        )
    return results


def run_ref_alt_strategy_optuna(context, strategy, base_config=None,
                                search_space=None, n_trials=None, display_fn=None):
    """Run one original fusion family with concatenated reference/alternate vectors."""
    if strategy not in FUSION_STRATEGIES:
        raise KeyError(
            f"Unknown fusion strategy {strategy!r}; choose from {list(FUSION_STRATEGIES)}"
        )
    base = get_default_ref_alt_config() if base_config is None else dict(base_config)
    spaces = get_default_ref_alt_search_spaces()
    space = spaces[strategy] if search_space is None else search_space
    label = FUSION_STRATEGIES[strategy]
    return run_architecture_optuna(
        context=context,
        base_config=base,
        search_space=space,
        n_trials=DEFAULT_REF_ALT_N_TRIALS if n_trials is None else int(n_trials),
        model_builder=_builder(strategy),
        architecture=f"RefAltConcat_{label}",
        variant_builder=lambda _cfg: f"ref_alt_concat_{strategy}",
        experiment_key=f"ref_alt_{strategy}",
        source="module06_ref_alt_concat",
        display_fn=display_fn,
        manifest_architecture=f"ref_alt_concat_{strategy}",
        embedding_representation=EMBEDDING_REPRESENTATION,
    )
