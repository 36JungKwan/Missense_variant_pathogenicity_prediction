"""Hybrid search: tune top DL encoders, feature sets, and three ML classifiers."""
from __future__ import annotations

import json
import hashlib
import importlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .dl_architecture_model import FusionArchitectureVariant
from .evaluator_profiler import FusionEvaluatorProfiler
from .resume_utils import ensure_stage_signature, save_progress_csv, upsert_record
from .optuna_search import optuna_trial_plan, tell_trial
from .global_results import refresh_global_metrics_from_context


HYBRID_CLASSIFIER_SPACES = {
    "XGBoost": {
        "max_depth": {"type": "int", "low": 3, "high": 18},
        "min_child_weight": {"type": "float", "low": 0.5, "high": 50.0, "log": True},
        "learning_rate": {"type": "float", "low": 0.0001, "high": 0.5, "log": True},
        "colsample_bytree": {"type": "float", "low": 0.2, "high": 1.0},
        "colsample_bylevel": {"type": "float", "low": 0.2, "high": 1.0},
        "colsample_bynode": {"type": "float", "low": 0.2, "high": 1.0},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "gamma": {"type": "float", "low": 0.0, "high": 10.0},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 0.1, "high": 100.0, "log": True},
        # Training is balanced; include ~0.236 (val prior correction) and 1.0 in this range.
        "scale_pos_weight": {"type": "float", "low": 0.15, "high": 5.0, "log": True},
        "max_bin": {"type": "int", "low": 128, "high": 512, "step": 64},
    },
    "LightGBM": {
        "max_depth": {"type": "int", "low": 3, "high": 18},
        "num_leaves": {"type": "int", "low": 4, "high": 256, "high_power_of_two": "max_depth"},
        "min_child_samples": {"type": "int", "low": 5, "high": 500, "log": True},
        "min_child_weight": {"type": "float", "low": 1e-4, "high": 50.0, "log": True},
        "learning_rate": {"type": "float", "low": 0.0001, "high": 0.5, "log": True},
        "colsample_bytree": {"type": "float", "low": 0.2, "high": 1.0},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "min_split_gain": {"type": "float", "low": 0.0, "high": 5.0},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 20.0, "log": True},
        "reg_lambda": {"type": "float", "low": 0.1, "high": 100.0, "log": True},
        "scale_pos_weight": {"type": "float", "low": 0.15, "high": 5.0, "log": True},
    },
    "CatBoost": {
        "depth": {"type": "int", "low": 3, "high": 18},
        "learning_rate": {"type": "float", "low": 0.0005, "high": 0.5, "log": True},
        "l2_leaf_reg": {"type": "float", "low": 0.1, "high": 100.0, "log": True},
        "rsm": {"type": "float", "low": 0.25, "high": 1.0},
        "random_strength": {"type": "float", "low": 0.0, "high": 10.0},
        "leaf_estimation_iterations": {"type": "int", "low": 1, "high": 30},
        "scale_pos_weight": {"type": "float", "low": 0.15, "high": 5.0, "log": True},
        "bootstrap_type": {"type": "categorical", "choices": ["Bayesian", "Bernoulli", "MVS"]},
        "bagging_temperature": {"type": "float", "low": 0.0, "high": 10.0,
                                "when": {"bootstrap_type": "Bayesian"}},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0,
                      "when": {"bootstrap_type": "Bernoulli"}},
    },
}
HYBRID_ENCODER_SPACE = {
    # Model capacity and regularization.
    "fusion_dim": {"type": "categorical", "choices": [128, 192, 256, 384, 512, 768, 1024]},
    "dropout": {"type": "float", "low": 0.0, "high": 0.65},
    "head_scale": {"type": "float", "low": 0.4, "high": 2.0},
    # Optimizer dynamics.
    "learning_rate": {"type": "float", "low": 1e-5, "high": 2e-2, "log": True},
    "weight_decay": {"type": "float", "low": 1e-8, "high": 0.2, "log": True},
    "optimizer": {"type": "categorical", "choices": ["adamw", "adam", "radam"]},
    "beta1": {"type": "float", "low": 0.80, "high": 0.99},
    "beta2": {"type": "float", "low": 0.90, "high": 0.9999},
    "optimizer_eps": {"type": "float", "low": 1e-9, "high": 1e-6, "log": True},
    "lr_scheduler": {"type": "categorical", "choices": ["plateau", "cosine"]},
    "lr_factor": {"type": "float", "low": 0.1, "high": 0.8},
    "lr_patience": {"type": "int", "low": 2, "high": 8},
    "min_learning_rate": {"type": "float", "low": 1e-7, "high": 1e-5, "log": True},
    "batch_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
    "max_epochs": {"type": "int", "low": 30, "high": 100},
    "early_stop_patience": {"type": "int", "low": 6, "high": 20},
    "grad_clip_norm": {"type": "categorical", "choices": [0.5, 1.0, 2.0, 5.0]},
    # Train data are balanced, while validation/test retain lower pathogenic prevalence.
    "pos_weight_scale": {"type": "float", "low": 0.01, "high": 5.0, "log": True},
    "loss_type": {"type": "categorical", "choices": ["bce", "weighted_bce", "focal"]},
    "focal_alpha": {"type": "float", "low": 0.02, "high": 0.8},
    "focal_gamma": {"type": "float", "low": 0.5, "high": 4.0},
    "label_smoothing": {"type": "float", "low": 0.0, "high": 0.10},
    # Noise/feature-level regularization on the DNA/protein embedding inputs.
    "feature_dropout": {"type": "float", "low": 0.0, "high": 0.40},
    "modality_dropout": {"type": "float", "low": 0.0, "high": 0.40},
    "gaussian_noise": {"type": "float", "low": 0.0, "high": 0.08},
}
DEFAULT_HYBRID_SEARCH_CONFIG = {
    "encoder_n_trials": 32,
    "classifier_n_trials": 48,
    "encoder_max_epochs": 100,
    "encoder_early_stop_patience": 20,
    "encoder_space": HYBRID_ENCODER_SPACE,
    "classifier_spaces": HYBRID_CLASSIFIER_SPACES,
    "feature_sets": ["latent_only", "latent_plus_bio", "latent_plus_bio_geom"],
    "confirmation_seeds": [42, 123, 2026],
}


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _hyperparameters(value):
    try:
        raw = json.loads(value) if isinstance(value, str) else (value or {})
    except (TypeError, json.JSONDecodeError):
        raw = {}
    aliases = {
        "fusion_dim": ("fusion_dim", "fusiondim"),
        "dropout": ("dropout",),
        "weight_decay": ("weight_decay", "weightdecay"),
        "learning_rate": ("learning_rate", "learningrate"),
        "head_scale": ("head_scale", "headscale"),
    }
    normalized = {re.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in raw.items()} if isinstance(raw, dict) else {}
    defaults = {"fusion_dim": 512, "dropout": 0.2, "weight_decay": 1e-3, "learning_rate": 1e-4, "head_scale": 1.0}
    for name, candidates in aliases.items():
        for candidate in candidates:
            key = re.sub(r"[^a-z0-9]", "", candidate.lower())
            if key in normalized:
                defaults[name] = normalized[key]
                break
    defaults["fusion_dim"] = int(float(defaults["fusion_dim"]))
    for key in ("dropout", "weight_decay", "learning_rate", "head_scale"):
        defaults[key] = float(defaults[key])
    # Preserve the family-specific architecture settings found by the expanded
    # pure-DL search when rebuilding the Hybrid encoder.
    fusion_parameter_types = {
        "concat_bilinear_rank": int,
        "cross_attention_dropout": float,
        "cross_attention_heads": int,
        "transformer_ffn_expansion": int,
        "transformer_num_heads": int,
        "transformer_depth": int,
        "gating_temperature": float,
    }
    defaults["fusion_hparams"] = {
        name: cast(normalized[re.sub(r"[^a-z0-9]", "", name.lower())])
        for name, cast in fusion_parameter_types.items()
        if re.sub(r"[^a-z0-9]", "", name.lower()) in normalized
        and pd.notna(normalized[re.sub(r"[^a-z0-9]", "", name.lower())])
    }
    return defaults


def select_top_dl_fusion_variants(result_path, target_configs, top_n=5):
    """Choose unique structures by mean validation MCC, never by test metrics."""
    results = pd.read_csv(result_path)
    required = {"Architecture", "Variant", "Target_Config", "Candidate_Level", "Profile", "Val_MCC", "Hyperparameters", "Source"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"DL result table lacks required columns: {sorted(missing)}")
    data = results[
        results["Architecture"].astype(str).str.startswith("PyTorch_")
        & results["Candidate_Level"].astype(str).ne("individual_seed")
        & ~results["Source"].astype(str).str.contains("baseline_batchrun|baseline_control", case=False, na=False)
        & results["Target_Config"].isin([c["name"] for c in target_configs])
    ].copy()
    data["Val_MCC"] = pd.to_numeric(data["Val_MCC"], errors="coerce")
    data = data.dropna(subset=["Val_MCC"])
    if data.empty:
        raise ValueError(f"No validation-scored DL fusion candidates in {result_path}")

    # Collapse duplicate stage records by candidate, then average test-set validation
    # scores. Pick the best DL training profile per structure/configuration.
    candidate_keys = ["Target_Config", "Architecture", "Variant", "Profile", "Candidate_Level", "Hyperparameters"]
    candidate = data.groupby(candidate_keys, dropna=False, as_index=False).agg(Val_MCC=("Val_MCC", "mean"))
    candidate = candidate.sort_values("Val_MCC", ascending=False).drop_duplicates(
        ["Target_Config", "Architecture", "Variant"]
    )
    coverage = candidate.groupby(["Architecture", "Variant"])["Target_Config"].nunique()
    complete = coverage[coverage == len(target_configs)].reset_index()[["Architecture", "Variant"]]
    candidate = candidate.merge(complete, on=["Architecture", "Variant"], how="inner")
    if candidate.empty:
        raise ValueError("No DL fusion variants have validation results in both configurations A and B")
    ranking = candidate.groupby(["Architecture", "Variant"], as_index=False).agg(
        Mean_Val_MCC=("Val_MCC", "mean"), Config_Count=("Target_Config", "nunique")
    ).sort_values(["Mean_Val_MCC", "Architecture", "Variant"], ascending=[False, True, True]).head(top_n)
    if len(ranking) < top_n:
        raise ValueError(f"Need {top_n} complete fusion variants, found {len(ranking)}")
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    chosen = candidate.merge(ranking[["Architecture", "Variant"]], on=["Architecture", "Variant"], how="inner")
    chosen = chosen.merge(ranking, on=["Architecture", "Variant"], how="inner")
    seed_trials_path = Path(result_path).parent.parent / "pure_dl_seed_confirmation_v1" / "seed_confirmation_trials.csv"
    seed_trials = pd.read_csv(seed_trials_path) if seed_trials_path.exists() else pd.DataFrame()
    resolved = []
    for _, row in chosen.iterrows():
        params = _hyperparameters(row["Hyperparameters"])
        try:
            raw = json.loads(row["Hyperparameters"]) if isinstance(row["Hyperparameters"], str) else row["Hyperparameters"]
        except (TypeError, json.JSONDecodeError):
            raw = {}
        if not isinstance(raw, dict) or not raw:
            if not seed_trials.empty:
                matches = seed_trials[
                    seed_trials["Target_Config"].astype(str).eq(str(row["Target_Config"]))
                    & seed_trials["Architecture"].astype(str).eq(str(row["Architecture"]))
                    & seed_trials["Variant"].astype(str).eq(str(row["Variant"]))
                    & seed_trials["Profile"].astype(str).eq(str(row["Profile"]))
                ]
                if not matches.empty:
                    source = matches.iloc[0].to_dict()
                    for key in ("fusion_dim", "dropout", "weight_decay", "learning_rate", "head_scale"):
                        for alias in (key, key.title(), key.replace("_", " ").title()):
                            if alias in source and pd.notna(source[alias]):
                                params[key] = int(source[alias]) if key == "fusion_dim" else float(source[alias])
                                break
        resolved.append(params)
    chosen["Selected_Hyperparameters"] = resolved
    return chosen.sort_values(["Rank", "Target_Config"]), ranking


def run_hybrid_top5_optuna(context, dl_results_path=None, search_config=None, display_fn=None):
    """Tune selected fusion encoders and classifiers with validation-only Optuna.

    For each selected structure/configuration, the encoder is tuned once and shared
    by the three classifier studies. Each classifier study jointly searches its
    hyperparameters and the latent/bio/geometry feature recipe. Final classifier
    candidates are confirmed across seeds; test sets are used only afterward.
    """
    pipeline = context["pipeline"]
    configs = context["TARGET_CONFIGS"]
    datasets = context["datasets"]
    dna_model, protein_model = context["DNA_MODEL"], context["PROTEIN_MODEL"]
    train_config = context["CONFIG"]
    device = context["DEVICE"]
    display = display_fn or print
    from . import xgboost_model as xgboost_model_module
    importlib.reload(xgboost_model_module)
    root = Path(pipeline.batch_run_dir)
    # Keep the existing resumable output location stable so rerunning this cell
    # reuses the completed Hybrid study instead of starting training again.
    out_dir = root / "hybrid_optuna_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    dl_results_path = Path(dl_results_path or root / "pure_dl_final_v1" / "all_dl_test_results.csv")
    selected, top_variants = select_top_dl_fusion_variants(dl_results_path, configs, 5)
    selection_path = out_dir / "top5_dl_fusion_variants.csv"
    save_progress_csv(selected.to_dict("records"), selection_path)

    search_config = search_config or DEFAULT_HYBRID_SEARCH_CONFIG
    spaces = search_config.get("classifier_spaces", HYBRID_CLASSIFIER_SPACES)
    encoder_n_trials = int(search_config.get("encoder_n_trials", DEFAULT_HYBRID_SEARCH_CONFIG["encoder_n_trials"]))
    classifier_n_trials = int(search_config.get("classifier_n_trials", DEFAULT_HYBRID_SEARCH_CONFIG["classifier_n_trials"]))
    encoder_space = search_config.get("encoder_space", HYBRID_ENCODER_SPACE)
    feature_sets = list(search_config.get("feature_sets", DEFAULT_HYBRID_SEARCH_CONFIG["feature_sets"]))
    confirmation_seeds = [int(s) for s in search_config.get("confirmation_seeds", DEFAULT_HYBRID_SEARCH_CONFIG["confirmation_seeds"])]
    available_cpus = os.cpu_count() or 1
    classifier_n_jobs = max(
        1,
        min(int(search_config.get("classifier_n_jobs", min(8, available_cpus))), available_cpus),
    )
    if set(spaces) != set(HYBRID_CLASSIFIER_SPACES):
        raise ValueError(f"Hybrid Optuna spaces must define exactly {list(HYBRID_CLASSIFIER_SPACES)}")
    valid_feature_sets = {"latent_only", "latent_plus_bio", "latent_plus_bio_geom"}
    if not feature_sets or not set(feature_sets).issubset(valid_feature_sets):
        raise ValueError(f"feature_sets must be selected from {sorted(valid_feature_sets)}")
    print(f"[HYBRID RESOURCES] encoder trials={encoder_n_trials}/structure-config; classifier trials={classifier_n_trials}/classifier-feature-config; classifier threads={classifier_n_jobs}")
    signature = {
        "top5": selected[["Rank", "Target_Config", "Architecture", "Variant", "Hyperparameters", "Selected_Hyperparameters", "Val_MCC"]].to_dict("records"),
        "sampler": "TPE", "encoder_space": encoder_space, "encoder_n_trials": encoder_n_trials,
        "classifier_spaces": spaces, "classifier_n_trials": classifier_n_trials,
        "feature_sets": feature_sets, "confirmation_seeds": confirmation_seeds,
        "training_config": train_config,
        "target_configs": configs,
        "datasets": datasets,
        "dna_model": dna_model,
        "protein_model": protein_model,
        "encoder_checkpoint_rule": "validation_mcc_optimized_threshold",
        "classifier_selection_rule": "validation_mcc_optimized_threshold",
        "classifier_common": {"n_estimators_or_iterations": 3000, "early_stopping_rounds": 150},
    }
    ensure_stage_signature(out_dir, "hybrid_expanded_optuna_v2", signature)

    trials_path = out_dir / "hybrid_optuna_trials.csv"
    validation_path = out_dir / "hybrid_validation_metrics.csv"
    test_path = out_dir / "hybrid_test_metrics.csv"
    confirmation_path = out_dir / "hybrid_seed_confirmation_metrics.csv"
    encoder_history_path = out_dir / "hybrid_encoder_epoch_history.csv"
    encoder_trials_path = out_dir / "hybrid_encoder_optuna_trials.csv"
    trial_rows = pd.read_csv(trials_path).to_dict("records") if trials_path.exists() else []
    validation_rows = pd.read_csv(validation_path).to_dict("records") if validation_path.exists() else []
    test_rows = pd.read_csv(test_path).to_dict("records") if test_path.exists() else []
    confirmation_rows = pd.read_csv(confirmation_path).to_dict("records") if confirmation_path.exists() else []
    history_rows = pd.read_csv(encoder_history_path).to_dict("records") if encoder_history_path.exists() else []
    encoder_trial_rows = pd.read_csv(encoder_trials_path).to_dict("records") if encoder_trials_path.exists() else []

    loader_cache = {}

    def loader(split, target, train=False):
        key = (str(split), str(target["name"]), bool(train))
        if key not in loader_cache:
            loader_cache[key] = pipeline._get_dataloader(
                split_name=split, pooling="center", dna_name=dna_model["name"], prot_name=protein_model["name"],
                active_mods=target["active_modalities"], is_train=train,
                geom_feature_names=target["geom_feature_names"], geometry_context="train3",
                loader_seed=int(train_config["seed"]),
            )
        return loader_cache[key]

    def move_batch(batch, key):
        value = batch.get(key)
        return value.to(device, non_blocking=True) if value is not None else None

    def forward(model, batch):
        return model(move_batch(batch, "v_dna"), move_batch(batch, "v_prot"),
                     move_batch(batch, "bio_features"), move_batch(batch, "geom_features"))

    def forward_train(model, batch, params, input_stds):
        dna, protein = move_batch(batch, "v_dna"), move_batch(batch, "v_prot")
        bio, geom = move_batch(batch, "bio_features"), move_batch(batch, "geom_features")
        feature_dropout = float(params.get("feature_dropout", 0.0))
        if feature_dropout:
            dna = torch.nn.functional.dropout(dna, p=feature_dropout, training=True)
            protein = torch.nn.functional.dropout(protein, p=feature_dropout, training=True)
        modality_dropout = float(params.get("modality_dropout", 0.0))
        if modality_dropout:
            keep_dna = torch.rand((dna.shape[0], 1), device=device) >= modality_dropout
            keep_protein = torch.rand((protein.shape[0], 1), device=device) >= modality_dropout
            both_missing = ~(keep_dna | keep_protein)
            if both_missing.any():
                restore_dna = torch.rand((dna.shape[0], 1), device=device) < 0.5
                keep_dna |= both_missing & restore_dna
                keep_protein |= both_missing & ~restore_dna
            dna, protein = dna * keep_dna, protein * keep_protein
        noise = float(params.get("gaussian_noise", 0.0))
        if noise:
            dna = dna + torch.randn_like(dna) * input_stds[0] * noise
            protein = protein + torch.randn_like(protein) * input_stds[1] * noise
        return model(dna, protein, bio, geom)

    def make_encoder_optimizer(model, params):
        optimizer_name = str(params.get("optimizer", "adamw")).lower()
        optimizer_class = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam,
                           "radam": torch.optim.RAdam}[optimizer_name]
        return optimizer_class(
            model.parameters(), lr=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
            betas=(float(params.get("beta1", 0.9)), float(params.get("beta2", 0.999))),
            eps=float(params.get("optimizer_eps", 1e-8)),
        )

    def make_encoder_loss(params, pos_weight):
        loss_type = str(params.get("loss_type", "weighted_bce"))
        smoothing = float(params.get("label_smoothing", 0.0))
        def loss_fn(logits, labels):
            targets = labels * (1.0 - smoothing) + 0.5 * smoothing if smoothing else labels
            per_item = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            if loss_type == "focal":
                p_correct = torch.exp(-per_item)
                alpha_pos = float(params.get("focal_alpha", 0.5))
                alpha = targets * alpha_pos + (1.0 - targets) * (1.0 - alpha_pos)
                return (alpha * (1.0 - p_correct).pow(float(params.get("focal_gamma", 2.0))) * per_item).mean()
            if loss_type == "weighted_bce":
                per_item = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=pos_weight, reduction="none"
                )
            return per_item.mean()
        return loss_fn

    def probabilities(model, dataloader, criterion=None):
        model.eval()
        labels, probs = [], []
        loss_sum = sample_count = 0
        with torch.no_grad():
            for batch in dataloader:
                targets = batch["label"].to(device, non_blocking=True)
                logits = forward(model, batch)
                labels.extend(targets.reshape(-1).cpu().numpy())
                probs.extend(torch.sigmoid(logits).reshape(-1).cpu().numpy())
                if criterion is not None:
                    loss_sum += float(criterion(logits, targets).item()) * len(targets)
                    sample_count += len(targets)
        mean_loss = loss_sum / max(sample_count, 1)
        return np.asarray(labels, dtype=np.int32), np.asarray(probs, dtype=np.float32), mean_loss

    def encode(model, dataloader):
        model.eval()
        features, labels, bio_values, geom_values = [], [], [], []
        with torch.no_grad():
            for batch in dataloader:
                latent = model(move_batch(batch, "v_dna"), move_batch(batch, "v_prot"),
                               move_batch(batch, "bio_features"), move_batch(batch, "geom_features"),
                               return_features=True)
                features.append(latent.cpu().numpy().astype(np.float32, copy=False))
                labels.append(batch["label"].reshape(-1).cpu().numpy().astype(np.int32, copy=False))
                if batch.get("bio_features") is not None:
                    bio_values.append(batch["bio_features"].cpu().numpy().astype(np.float32, copy=False))
                if batch.get("geom_features") is not None:
                    geom_values.append(batch["geom_features"].cpu().numpy().astype(np.float32, copy=False))
        n_samples = sum(len(values) for values in labels)
        side = {
            "bio": np.concatenate(bio_values) if bio_values else np.empty((n_samples, 0), dtype=np.float32),
            "geom": np.concatenate(geom_values) if geom_values else np.empty((n_samples, 0), dtype=np.float32),
        }
        return np.concatenate(features), np.concatenate(labels), side

    def compose_features(latent, side, feature_set):
        parts = [latent]
        if feature_set in {"latent_plus_bio", "latent_plus_bio_geom"} and side["bio"].shape[1]:
            parts.append(side["bio"])
        if feature_set == "latent_plus_bio_geom" and side["geom"].shape[1]:
            parts.append(side["geom"])
        return np.concatenate(parts, axis=1).astype(np.float32, copy=False)

    def fit_encoder(target, architecture, variant, params, tr_loader, val_loader, dims, checkpoint, history_key, seed):
        resume_path = checkpoint.with_suffix(".training.pt")
        best_path = checkpoint.with_suffix(".best.pth")
        if checkpoint.exists():
            model = FusionArchitectureVariant(
                dims["dna"], dims["prot"], dims["bio"], dims["geom"], target["active_modalities"],
                variant, params["fusion_dim"], params["dropout"], params["head_scale"],
                fusion_hparams=params.get("fusion_hparams", {}),
            ).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            return model
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        model = FusionArchitectureVariant(
            dims["dna"], dims["prot"], dims["bio"], dims["geom"], target["active_modalities"],
            variant, params["fusion_dim"], params["dropout"], params["head_scale"],
            fusion_hparams=params.get("fusion_hparams", {}),
        ).to(device)
        pos_weight = torch.tensor([float(params.get("pos_weight", 1.0))], device=device)
        criterion = make_encoder_loss(params, pos_weight)
        optimizer = make_encoder_optimizer(model, params)
        max_epochs = int(params.get("max_epochs", search_config.get("encoder_max_epochs", train_config["epochs"])))
        scheduler_name = str(params.get("lr_scheduler", "plateau"))
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=float(params.get("min_learning_rate", 1e-6))
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=float(params.get("lr_factor", train_config["lr_factor"])),
                patience=int(params.get("lr_patience", train_config["lr_patience"])),
                min_lr=float(params.get("min_learning_rate", 1e-6)),
            )
        from torch.utils.data import DataLoader
        encoder_train_loader = DataLoader(
            tr_loader.dataset, batch_size=int(params.get("batch_size", tr_loader.batch_size)),
            shuffle=True, num_workers=tr_loader.num_workers,
            pin_memory=getattr(tr_loader, "pin_memory", False), drop_last=False,
            generator=torch.Generator().manual_seed(int(seed)), collate_fn=tr_loader.collate_fn,
        )
        best_mcc, best_epoch, wait, start_epoch = -float("inf"), 0, 0, 0
        if resume_path.exists():
            state = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            best_mcc, best_epoch, wait = state["best_mcc"], state["best_epoch"], state["wait"]
            start_epoch = state["next_epoch"]
            torch.set_rng_state(state["torch_rng_state"])
            if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng_state"])
            if state.get("loader_rng_state") is not None:
                encoder_train_loader.generator.set_state(state["loader_rng_state"])
            print(f"[RESUME] Hybrid encoder {target['name']}/{variant} from epoch {start_epoch + 1}")
        patience = int(params.get("early_stop_patience", search_config.get("encoder_early_stop_patience", train_config["early_stop_patience"])))
        dataset = tr_loader.dataset
        input_stds = (
            float(dataset.v_dna_tensor.float().std(unbiased=False).item()),
            float(dataset.v_prot_tensor.float().std(unbiased=False).item()),
        )
        for epoch in range(start_epoch, max_epochs):
            if wait >= patience:
                break
            model.train()
            loss_sum = sample_count = 0
            for batch in encoder_train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = forward_train(model, batch, params, input_stds)
                labels = batch["label"].to(device, non_blocking=True)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(params.get("grad_clip_norm", 1.0)))
                optimizer.step()
                n = int(labels.shape[0]); loss_sum += float(loss.item()) * n; sample_count += n
            train_loss = loss_sum / max(sample_count, 1)
            y_val, p_val, val_loss = probabilities(model, val_loader, criterion)
            threshold, val_mcc = pipeline._select_validation_threshold(y_val, p_val)
            scheduler.step() if scheduler_name == "cosine" else scheduler.step(val_mcc)
            history_rows.append({
                "Target_Config": target["name"], "Architecture": architecture, "Variant": variant,
                "Trial_Key": history_key, "Training_Seed": int(seed),
                "Epoch": epoch + 1, "Learning_Rate": optimizer.param_groups[0]["lr"],
                "Train_Loss": train_loss, "Val_Loss": val_loss,
                "Val_MCC_Optimized_Threshold": val_mcc, "Val_Threshold": threshold,
            })
            save_progress_csv(history_rows, encoder_history_path)
            if val_mcc > best_mcc:
                best_mcc, best_epoch, wait = val_mcc, epoch + 1, 0
                temporary_best = best_path.with_name(best_path.name + ".tmp")
                torch.save(model.state_dict(), temporary_best)
                temporary_best.replace(best_path)
            else:
                wait += 1
            state = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "next_epoch": epoch + 1,
                "best_mcc": best_mcc, "best_epoch": best_epoch, "wait": wait,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "loader_rng_state": encoder_train_loader.generator.get_state(),
            }
            temporary_state = resume_path.with_name(resume_path.name + ".tmp")
            torch.save(state, temporary_state)
            temporary_state.replace(resume_path)
            if wait >= patience:
                break
        if not best_path.exists():
            raise RuntimeError(f"No Hybrid encoder checkpoint for {target['name']}/{variant}")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        temporary_checkpoint = checkpoint.with_name(checkpoint.name + ".tmp")
        torch.save(model.state_dict(), temporary_checkpoint)
        temporary_checkpoint.replace(checkpoint)
        best_path.unlink(missing_ok=True)
        resume_path.unlink(missing_ok=True)
        print(f"[HYBRID ENCODER] {target['name']}/{variant}: best_epoch={best_epoch}, val_MCC={best_mcc:.4f}")
        return model

    def new_classifier(name, params, seed, scale_pos_weight=1.0):
        params = dict(params)
        params.pop("feature_set", None)
        if name == "XGBoost":
            from .xgboost_model import XGBoostFusionManager
            model = XGBoostFusionManager(random_state=seed, n_jobs=classifier_n_jobs)
            classifier_params = dict(params)
            weight_scale = float(classifier_params.pop("class_weight_scale", 1.0))
            sampled_scale_pos_weight = classifier_params.pop("scale_pos_weight", None)
            model.base_params.update({"n_estimators": 3000, "early_stopping_rounds": 150})
            model.base_params.update(classifier_params)
            model.base_params["scale_pos_weight"] = float(
                sampled_scale_pos_weight if sampled_scale_pos_weight is not None
                else float(scale_pos_weight) * weight_scale
            )
            return model
        if name == "LightGBM":
            import lightgbm as lgb
            classifier_params = dict(params)
            weight_scale = float(classifier_params.pop("class_weight_scale", 1.0))
            sampled_scale_pos_weight = classifier_params.pop("scale_pos_weight", None)
            effective_params = {
                "n_estimators": 3000, "max_depth": -1, "subsample_freq": 1,
                "objective": "binary", "verbosity": -1,
                "n_jobs": classifier_n_jobs, "random_state": seed,
                "scale_pos_weight": float(
                    sampled_scale_pos_weight if sampled_scale_pos_weight is not None
                    else float(scale_pos_weight) * weight_scale
                ),
            }
            effective_params.update(classifier_params)
            return lgb.LGBMClassifier(**effective_params)
        from catboost import CatBoostClassifier
        classifier_params = dict(params)
        sampled_scale_pos_weight = classifier_params.pop("scale_pos_weight", None)
        class_weight_mode = classifier_params.pop("class_weight_mode", "Balanced")
        effective_params = {
            "iterations": 3000, "loss_function": "Logloss", "eval_metric": "Logloss",
            "auto_class_weights": None if sampled_scale_pos_weight is not None or class_weight_mode == "None" else class_weight_mode,
            "random_seed": seed, "thread_count": classifier_n_jobs, "verbose": False,
        }
        if sampled_scale_pos_weight is not None:
            effective_params["scale_pos_weight"] = float(sampled_scale_pos_weight)
        effective_params.update(classifier_params)
        return CatBoostClassifier(**effective_params)

    def effective_classifier_parameters(name, params, seed):
        """Persist tuned and fixed estimator settings together for reproducibility."""
        recorded = dict(params)
        recorded.pop("feature_set", None)
        if name == "XGBoost":
            fixed = {"n_estimators": 3000, "early_stopping_rounds": 150,
                     "objective": "binary:logistic", "eval_metric": "logloss",
                     "tree_method": "hist", "n_jobs": classifier_n_jobs, "random_state": seed}
        elif name == "LightGBM":
            fixed = {"n_estimators": 3000, "early_stopping_rounds": 150,
                     "objective": "binary", "n_jobs": classifier_n_jobs, "random_state": seed}
        else:
            fixed = {"iterations": 3000, "early_stopping_rounds": 150,
                     "loss_function": "Logloss", "eval_metric": "Logloss",
                     "auto_class_weights": None if "scale_pos_weight" in recorded else "Balanced",
                     "thread_count": classifier_n_jobs, "random_seed": seed}
        return {**fixed, **recorded}

    def fit_classifier(name, model, xtr, ytr, xval, yval):
        if name == "XGBoost":
            model.train_hybrid(xtr, ytr, xval, yval)
            probs, _, _ = model.predict_hybrid(xval)
        elif name == "LightGBM":
            import lightgbm as lgb
            model.fit(xtr, ytr, eval_X=xval, eval_y=yval, eval_metric="binary_logloss",
                      callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
            probs = model.predict_proba(xval, num_iteration=model.best_iteration_)[:, 1]
        else:
            model.fit(xtr, ytr, eval_set=(xval, yval), early_stopping_rounds=150, verbose=False)
            probs = model.predict_proba(xval)[:, 1]
        threshold, val_mcc = pipeline._select_validation_threshold(yval, probs)
        metrics = FusionEvaluatorProfiler.compute_metrics(yval, probs, pipeline._threshold_predictions(probs, threshold))
        return model, threshold, val_mcc, metrics

    def save_classifier(name, model, path):
        if name == "XGBoost":
            model.save_model(str(path.parent), prefix=path.name.removesuffix("_xgboost.json"))
        elif name == "LightGBM":
            model.booster_.save_model(str(path))
        else:
            model.save_model(str(path))

    def load_classifier(name, path, params, seed):
        if name == "XGBoost":
            model = new_classifier(name, params, seed); model.load_model(str(path.parent), prefix=path.name.removesuffix("_xgboost.json")); return model
        if name == "LightGBM":
            import lightgbm as lgb
            return lgb.Booster(model_file=str(path))
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(); model.load_model(str(path)); return model

    def predict_classifier(name, model, x):
        if name == "XGBoost": return model.predict_hybrid(x)[0]
        if name == "LightGBM": return model.predict(x)
        return model.predict_proba(x)[:, 1]

    # Iterate config -> structure. Each encoder is trained once and feeds all ML studies.
    for target in configs:
        config_name = target["name"]
        target_candidates = selected[selected["Target_Config"].eq(config_name)]
        for _, candidate in target_candidates.iterrows():
            architecture, variant = str(candidate["Architecture"]), str(candidate["Variant"])
            params = candidate["Selected_Hyperparameters"]
            model_tag = f"{config_name}__{architecture}__{variant}"
            model_dir = out_dir / "encoders" / _safe_name(model_tag)
            model_dir.mkdir(parents=True, exist_ok=True)
            train_loader = loader("train3", target, True)
            val_loader = loader("val", target, False)
            first = next(iter(train_loader))
            dims = {"dna": int(first["v_dna"].shape[1]), "prot": int(first["v_prot"].shape[1]),
                    "bio": int(first["bio_features"].shape[1]),
                    "geom": int(first["geom_features"].shape[1]) if "geom_features" in first else 0}
            if getattr(train_loader, "generator", None) is not None:
                train_loader.generator.manual_seed(int(train_config["seed"]))
            train_labels = train_loader.dataset.labels.detach().cpu().numpy().astype(np.int32)
            pos_ratio = float(np.sum(train_labels == 0) / max(np.sum(train_labels == 1), 1))
            encoder_combo = f"{model_tag}__encoder"
            encoder_study_dir = out_dir / "optuna_encoder" / _safe_name(encoder_combo)
            allowed_dims = [value for value in encoder_space["fusion_dim"]["choices"] if int(value) % 8 == 0]
            if not allowed_dims:
                raise ValueError("Encoder fusion_dim choices must be divisible by the 8-head/token constraint")
            local_encoder_space = dict(encoder_space)
            local_encoder_space["fusion_dim"] = {"type": "categorical", "choices": allowed_dims}
            encoder_plan = optuna_trial_plan(
                encoder_study_dir, encoder_combo, encoder_n_trials, local_encoder_space,
                int(train_config["seed"]),
            )
            best_encoder = None
            for enc_study, enc_trial, enc_hp, is_new_encoder_trial in encoder_plan:
                encoder_trial_id = f"{encoder_combo}__trial_{enc_trial.number:04d}"
                parameter_hash = hashlib.sha1(json.dumps(enc_hp, sort_keys=True).encode("utf-8")).hexdigest()[:12]
                enc_checkpoint = model_dir / "encoder_trials" / f"encoder_params_{parameter_hash}.pth"
                enc_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                cached_encoder = next((row for row in encoder_trial_rows if row.get("Trial_ID") == encoder_trial_id and enc_checkpoint.exists()), None)
                trial_params = dict(params)
                trial_params.update(enc_hp)
                trial_params["fusion_dim"] = int(trial_params["fusion_dim"])
                trial_params["dropout"] = float(trial_params["dropout"])
                trial_params["learning_rate"] = float(trial_params["learning_rate"])
                trial_params["weight_decay"] = float(trial_params["weight_decay"])
                trial_params["pos_weight"] = pos_ratio * float(enc_hp["pos_weight_scale"])
                if cached_encoder is not None:
                    val_mcc = float(cached_encoder["Val_MCC"])
                    val_auprc = float(cached_encoder["Val_AUPRC"])
                    torch_model = FusionArchitectureVariant(
                        dims["dna"], dims["prot"], dims["bio"], dims["geom"], target["active_modalities"],
                        variant, trial_params["fusion_dim"], trial_params["dropout"], trial_params["head_scale"],
                        fusion_hparams=trial_params.get("fusion_hparams", {}),
                    ).to(device)
                    torch_model.load_state_dict(torch.load(enc_checkpoint, map_location=device, weights_only=True))
                else:
                    if getattr(train_loader, "generator", None) is not None:
                        train_loader.generator.manual_seed(int(train_config["seed"]))
                    torch_model = fit_encoder(
                        target, architecture, variant, trial_params, train_loader, val_loader,
                        dims, enc_checkpoint, encoder_trial_id, int(train_config["seed"]),
                    )
                    y_val_enc, p_val_enc, _ = probabilities(torch_model, val_loader)
                    enc_threshold, val_mcc = pipeline._select_validation_threshold(y_val_enc, p_val_enc)
                    val_auprc = FusionEvaluatorProfiler.compute_metrics(
                        y_val_enc, p_val_enc,
                        pipeline._threshold_predictions(p_val_enc, enc_threshold),
                    )["AUPRC"]
                    encoder_trial_rows = upsert_record(encoder_trial_rows, {
                        "Trial_ID": encoder_trial_id, "Target_Config": config_name,
                        "Architecture": architecture, "Variant": variant, "Trial": enc_trial.number,
                        "Parameters": json.dumps(trial_params, sort_keys=True), "Val_MCC": float(val_mcc),
                        "Val_AUPRC": float(val_auprc), "Checkpoint": str(enc_checkpoint),
                    }, ("Trial_ID",))
                    save_progress_csv(encoder_trial_rows, encoder_trials_path)
                    tell_trial(enc_study, enc_trial, val_mcc) if is_new_encoder_trial else None
                score = (val_mcc, float(val_auprc))
                if best_encoder is None or score > best_encoder["score"]:
                    best_encoder = {"score": score, "model": torch_model, "params": trial_params,
                                    "checkpoint": enc_checkpoint, "trial": enc_trial.number,
                                    "val_mcc": val_mcc, "val_auprc": val_auprc}
            if best_encoder is None:
                raise RuntimeError(f"No encoder trial completed for {encoder_combo}")
            params = best_encoder["params"]
            encoder = best_encoder["model"]
            encoder_trial_number = best_encoder["trial"]

            feature_path = model_dir / "latent_features_optuna_v2.npz"
            if feature_path.exists():
                cached = np.load(feature_path, allow_pickle=False)
                x_train, y_train, x_val, y_val = cached["x_train"], cached["y_train"], cached["x_val"], cached["y_val"]
                x_tests = {d["name"]: cached[f"x_test_{i}"] for i, d in enumerate(datasets)}
                y_tests = {d["name"]: cached[f"y_test_{i}"] for i, d in enumerate(datasets)}
                side_train = {key: cached[f"{key}_train"] for key in ("bio", "geom")}
                side_val = {key: cached[f"{key}_val"] for key in ("bio", "geom")}
                side_tests = {d["name"]: {key: cached[f"{key}_test_{i}"] for key in ("bio", "geom")} for i, d in enumerate(datasets)}
            else:
                x_train, y_train, side_train = encode(encoder, train_loader)
                x_val, y_val, side_val = encode(encoder, val_loader)
                x_tests, y_tests, side_tests = {}, {}, {}
                for i, dataset in enumerate(datasets):
                    test_loader = loader(dataset["test"], target, False)
                    x_tests[dataset["name"]], y_tests[dataset["name"]], side_tests[dataset["name"]] = encode(encoder, test_loader)
                    del test_loader
                temporary_features = feature_path.with_name(feature_path.name + ".tmp")
                with temporary_features.open("wb") as handle:
                    np.savez_compressed(
                        handle, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                        bio_train=side_train["bio"], geom_train=side_train["geom"],
                        bio_val=side_val["bio"], geom_val=side_val["geom"],
                        **{f"x_test_{i}": x_tests[d["name"]] for i, d in enumerate(datasets)},
                        **{f"y_test_{i}": y_tests[d["name"]] for i, d in enumerate(datasets)},
                        **{f"{key}_test_{i}": side_tests[d["name"]][key] for i, d in enumerate(datasets) for key in ("bio", "geom")},
                    )
                temporary_features.replace(feature_path)
            available_feature_sets = [
                feature_set for feature_set in feature_sets
                if feature_set != "latent_plus_bio_geom" or side_train["geom"].shape[1] > 0
            ]
            if side_train["bio"].shape[1] == 0:
                available_feature_sets = ["latent_only"]
            feature_data = {
                feature_set: {
                    "x_train": compose_features(x_train, side_train, feature_set),
                    "y_train": y_train,
                    "x_val": compose_features(x_val, side_val, feature_set),
                    "y_val": y_val,
                    "x_tests": {name: compose_features(x, side_tests[name], feature_set) for name, x in x_tests.items()},
                }
                for feature_set in available_feature_sets
            }
            feature_data_by_seed = {int(train_config["seed"]): feature_data}
            for confirmation_seed in confirmation_seeds:
                if confirmation_seed == int(train_config["seed"]):
                    continue
                seed_encoder_dir = model_dir / "encoder_confirmations"
                seed_encoder_dir.mkdir(parents=True, exist_ok=True)
                seed_checkpoint = seed_encoder_dir / f"encoder_seed_{confirmation_seed}.pth"
                seed_feature_path = seed_encoder_dir / f"latent_features_seed_{confirmation_seed}.npz"
                if seed_feature_path.exists():
                    seed_cache = np.load(seed_feature_path, allow_pickle=False)
                    seed_x_train, seed_y_train = seed_cache["x_train"], seed_cache["y_train"]
                    seed_x_val, seed_y_val = seed_cache["x_val"], seed_cache["y_val"]
                    seed_x_tests = {d["name"]: seed_cache[f"x_test_{i}"] for i, d in enumerate(datasets)}
                    seed_y_tests = {d["name"]: seed_cache[f"y_test_{i}"] for i, d in enumerate(datasets)}
                    seed_side_train = {key: seed_cache[f"{key}_train"] for key in ("bio", "geom")}
                    seed_side_val = {key: seed_cache[f"{key}_val"] for key in ("bio", "geom")}
                    seed_side_tests = {d["name"]: {key: seed_cache[f"{key}_test_{i}"] for key in ("bio", "geom")} for i, d in enumerate(datasets)}
                    del seed_cache
                else:
                    if getattr(train_loader, "generator", None) is not None:
                        train_loader.generator.manual_seed(int(confirmation_seed))
                    seed_encoder = fit_encoder(
                        target, architecture, variant, params, train_loader, val_loader,
                        dims, seed_checkpoint, f"confirmation_seed_{confirmation_seed}", confirmation_seed,
                    )
                    seed_x_train, seed_y_train, seed_side_train = encode(seed_encoder, train_loader)
                    seed_x_val, seed_y_val, seed_side_val = encode(seed_encoder, val_loader)
                    seed_x_tests, seed_y_tests, seed_side_tests = {}, {}, {}
                    for i, dataset in enumerate(datasets):
                        test_loader = loader(dataset["test"], target, False)
                        seed_x_tests[dataset["name"]], seed_y_tests[dataset["name"]], seed_side_tests[dataset["name"]] = encode(seed_encoder, test_loader)
                        del test_loader
                    temp_seed_features = seed_feature_path.with_name(seed_feature_path.name + ".tmp")
                    with temp_seed_features.open("wb") as handle:
                        np.savez_compressed(
                            handle, x_train=seed_x_train, y_train=seed_y_train,
                            x_val=seed_x_val, y_val=seed_y_val,
                            bio_train=seed_side_train["bio"], geom_train=seed_side_train["geom"],
                            bio_val=seed_side_val["bio"], geom_val=seed_side_val["geom"],
                            **{f"x_test_{i}": seed_x_tests[d["name"]] for i, d in enumerate(datasets)},
                            **{f"y_test_{i}": seed_y_tests[d["name"]] for i, d in enumerate(datasets)},
                            **{f"{key}_test_{i}": seed_side_tests[d["name"]][key] for i, d in enumerate(datasets) for key in ("bio", "geom")},
                        )
                    temp_seed_features.replace(seed_feature_path)
                    del seed_encoder
                    torch.cuda.empty_cache()
                feature_data_by_seed[int(confirmation_seed)] = {
                    feature_set: {
                        "x_train": compose_features(seed_x_train, seed_side_train, feature_set),
                        "y_train": seed_y_train,
                        "x_val": compose_features(seed_x_val, seed_side_val, feature_set),
                        "y_val": seed_y_val,
                        "x_tests": {name: compose_features(x, seed_side_tests[name], feature_set) for name, x in seed_x_tests.items()},
                    }
                    for feature_set in available_feature_sets
                }
                if not np.array_equal(np.sort(seed_y_train), np.sort(y_train)) or not np.array_equal(seed_y_val, y_val):
                    raise ValueError(f"Labels changed while confirming encoder seed {confirmation_seed}")
                for dataset in datasets:
                    dataset_name = dataset["name"]
                    if not np.array_equal(seed_y_tests[dataset_name], y_tests[dataset_name]):
                        raise ValueError(f"Test labels/order changed while confirming encoder seed {confirmation_seed}/{dataset_name}")
            del encoder, train_loader, val_loader
            torch.cuda.empty_cache()

            for classifier_name, search_space in spaces.items():
                combo = f"{model_tag}__{classifier_name}"
                class_ratio = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
                best = None
                study_dir = out_dir / "optuna" / _safe_name(combo)
                joint_space = dict(search_space)
                joint_space["feature_set"] = {"type": "categorical", "choices": available_feature_sets}
                trial_plan = optuna_trial_plan(study_dir, combo, classifier_n_trials, joint_space, int(train_config["seed"]))
                for trial_index, (study, opt_trial, hp, is_new_trial) in enumerate(trial_plan, 1):
                    trial_id = f"{combo}__trial_{opt_trial.number:04d}"
                    model_path = out_dir / "classifiers" / _safe_name(trial_id)
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    extension = {"XGBoost": ".json", "LightGBM": ".txt", "CatBoost": ".cbm"}[classifier_name]
                    model_path = model_path.with_suffix(extension)
                    if classifier_name == "XGBoost":
                        model_path = model_path.with_name(model_path.stem + "_xgboost.json")
                    cached = next((r for r in trial_rows if r.get("Trial_ID") == trial_id and model_path.exists()), None)
                    if cached is not None:
                        model = load_classifier(classifier_name, model_path, hp, int(train_config["seed"]))
                        val_mcc, threshold = float(cached["Val_MCC"]), float(cached["Decision_Threshold"])
                        val_metrics = {m: float(cached[f"Val_{m}"]) for m in ("Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "MCC", "AUROC", "AUPRC")}
                        tell_trial(study, opt_trial, val_mcc) if is_new_trial else None
                    else:
                        feature_set = hp["feature_set"]
                        model = new_classifier(classifier_name, hp, int(train_config["seed"]), class_ratio)
                        model, threshold, val_mcc, val_metrics = fit_classifier(
                            classifier_name, model, feature_data[feature_set]["x_train"], y_train,
                            feature_data[feature_set]["x_val"], y_val,
                        )
                        save_classifier(classifier_name, model, model_path)
                        trial_row = {
                            "Trial_ID": trial_id, "Target_Config": config_name, "Architecture": architecture,
                            "Variant": variant, "Classifier": classifier_name,
                            "Feature_Set": hp["feature_set"], "Encoder_Trial": encoder_trial_number,
                            "Trial": opt_trial.number,
                            "Parameters": json.dumps(
                                effective_classifier_parameters(classifier_name, hp, int(train_config["seed"])),
                                sort_keys=True,
                            ), "Decision_Threshold": threshold,
                            "Val_MCC": val_mcc, **{f"Val_{k}": v for k, v in val_metrics.items()},
                            "Model_Path": str(model_path),
                        }
                        trial_rows = upsert_record(trial_rows, trial_row, ("Trial_ID",))
                        save_progress_csv(trial_rows, trials_path)
                        tell_trial(study, opt_trial, val_mcc) if is_new_trial else None
                    score = (val_mcc, float(val_metrics["AUPRC"]))
                    if best is None or score > best["score"]:
                        best = {"score": score, "model": model, "path": model_path, "params": hp,
                                "threshold": threshold, "val_mcc": val_mcc, "val_metrics": val_metrics,
                                "trial": opt_trial.number}
                if best is None:
                    raise RuntimeError(f"No classifier trial completed for {combo}")

                selected_feature_set = str(best["params"]["feature_set"])
                network = f"Hybrid_{classifier_name}_{architecture}_{variant}__{selected_feature_set}__encoder_trial_{encoder_trial_number:02d}"
                seed_results = []
                for confirmation_seed in confirmation_seeds:
                    existing_seed_rows = [
                        row for row in confirmation_rows
                        if str(row.get("Target_Config")) == config_name
                        and str(row.get("Architecture")) == architecture
                        and str(row.get("Variant")) == variant
                        and str(row.get("Classifier")) == classifier_name
                        and str(row.get("Feature_Set")) == selected_feature_set
                        and str(row.get("Seed")) == str(confirmation_seed)
                    ]
                    if len(existing_seed_rows) == len(datasets):
                        seed_results.extend(existing_seed_rows)
                        continue
                    seed_model = new_classifier(classifier_name, best["params"], confirmation_seed, class_ratio)
                    seed_feature_data = feature_data_by_seed.get(confirmation_seed, feature_data)
                    seed_features = seed_feature_data[selected_feature_set]
                    seed_model, seed_threshold, seed_val_mcc, seed_val_metrics = fit_classifier(
                        classifier_name, seed_model, seed_features["x_train"], seed_features["y_train"],
                        seed_features["x_val"], seed_features["y_val"],
                    )
                    for dataset in datasets:
                        ds_name = dataset["name"]
                        seed_probs = predict_classifier(
                            classifier_name, seed_model, seed_feature_data[selected_feature_set]["x_tests"][ds_name],
                        )
                        seed_preds = pipeline._threshold_predictions(seed_probs, seed_threshold)
                        seed_metrics = FusionEvaluatorProfiler.compute_metrics(y_tests[ds_name], seed_probs, seed_preds)
                        seed_row = {
                            "Target_Config": config_name, "Dataset": ds_name, "Architecture": architecture,
                            "Variant": variant, "Classifier": classifier_name, "Feature_Set": selected_feature_set,
                            "Encoder_Trial": encoder_trial_number, "Seed": confirmation_seed, "Network": network,
                            "Parameters": json.dumps(
                                effective_classifier_parameters(classifier_name, best["params"], confirmation_seed),
                                sort_keys=True,
                            ),
                            "Fusion_Hyperparameters": json.dumps(params, sort_keys=True),
                            "Decision_Threshold": seed_threshold, "Val_MCC": seed_val_mcc,
                            "Val_AUPRC": seed_val_metrics["AUPRC"], **seed_metrics,
                        }
                        confirmation_rows = upsert_record(
                            confirmation_rows, seed_row,
                            ("Target_Config", "Dataset", "Architecture", "Variant", "Classifier", "Feature_Set", "Seed"),
                        )
                        seed_results.append(seed_row)
                    save_progress_csv(confirmation_rows, confirmation_path)
                    del seed_model
                seed_frame = pd.DataFrame(seed_results)
                seed_summary = seed_frame.groupby("Dataset", as_index=False).agg(
                    MCC=("MCC", "mean"), MCC_Seed_STD=("MCC", "std"),
                    AUROC=("AUROC", "mean"), AUROC_Seed_STD=("AUROC", "std"),
                    AUPRC=("AUPRC", "mean"), AUPRC_Seed_STD=("AUPRC", "std"),
                )
                per_seed_validation = seed_frame.groupby("Seed", as_index=False).first()
                mean_val_mcc = float(per_seed_validation["Val_MCC"].mean())
                mean_val_auprc = float(per_seed_validation["Val_AUPRC"].mean())
                mean_threshold = float(per_seed_validation["Decision_Threshold"].mean())
                validation_row = {
                    "Target_Config": config_name, "Ablation": target["ablation"], "Architecture": architecture,
                    "Variant": variant, "Classifier": classifier_name, "Feature_Set": selected_feature_set,
                    "Encoder_Trial": encoder_trial_number, "Network": network,
                    "Model": network, "Selected_Trial": best["trial"],
                    "Parameters": json.dumps(
                        effective_classifier_parameters(classifier_name, best["params"], int(train_config["seed"])),
                        sort_keys=True,
                    ),
                    "Fusion_Hyperparameters": json.dumps(params, sort_keys=True),
                    "Decision_Threshold": mean_threshold, "Val_MCC": mean_val_mcc,
                    "Val_AUPRC": mean_val_auprc, "Confirmation_Seeds": len(confirmation_seeds),
                    "Optuna_Val_MCC": best["val_mcc"],
                    "DL_Fusion_Mean_Val_MCC": float(candidate["Mean_Val_MCC"]),
                }
                validation_rows = upsert_record(validation_rows, validation_row,
                    ("Target_Config", "Architecture", "Variant", "Classifier", "Feature_Set"))
                save_progress_csv(validation_rows, validation_path)
                for dataset in datasets:
                    ds_name = dataset["name"]
                    metrics = seed_summary.loc[seed_summary["Dataset"].eq(ds_name)].iloc[0].to_dict()
                    test_row = {
                        "Target_Config": config_name, "Ablation": target["ablation"], "Dataset": ds_name,
                        "Architecture": architecture, "Variant": variant, "Classifier": classifier_name,
                        "Feature_Set": selected_feature_set, "Encoder_Trial": encoder_trial_number,
                        "Network": network, "Model": network, "Selected_Trial": best["trial"],
                        "Parameters": json.dumps(
                            effective_classifier_parameters(classifier_name, best["params"], int(train_config["seed"])),
                            sort_keys=True,
                        ),
                        "Fusion_Hyperparameters": json.dumps(params, sort_keys=True),
                        "Decision_Threshold": mean_threshold, "Val_MCC": mean_val_mcc,
                        "Val_AUPRC": mean_val_auprc, "Confirmation_Seeds": len(confirmation_seeds), **metrics,
                        "DL_Fusion_Mean_Val_MCC": float(candidate["Mean_Val_MCC"]),
                    }
                    test_rows = upsert_record(test_rows, test_row,
                        ("Target_Config", "Dataset", "Architecture", "Variant", "Classifier", "Feature_Set"))
                    save_progress_csv(test_rows, test_path)
            print(f"[HYBRID] Finished {config_name}/{architecture}/{variant}")

    result = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    validation = pd.read_csv(validation_path) if validation_path.exists() else pd.DataFrame()
    expected = len(top_variants) * len(configs) * len(HYBRID_CLASSIFIER_SPACES) * len(datasets)
    print(f"[HYBRID COMPLETE] {len(result)}/{expected} test evaluations.")
    if len(result) != expected:
        raise RuntimeError(f"Hybrid result table is incomplete: expected {expected} rows, found {len(result)}")
    display(top_variants)
    if not validation.empty:
        display(validation.sort_values(["Target_Config", "Val_MCC"], ascending=[True, False]))
    if not result.empty:
        best_networks = validation.sort_values(
            ["Target_Config", "Val_MCC", "Val_AUPRC"], ascending=[True, False, False]
        ).drop_duplicates("Target_Config")[["Target_Config", "Network", "Val_MCC", "Val_AUPRC"]]
        selected_v2 = result.merge(best_networks, on=["Target_Config", "Network", "Val_MCC", "Val_AUPRC"], how="inner", suffixes=("", "_Validation"))
        print("Best Hybrid model per A/B, selected by validation MCC:")
        display(selected_v2.sort_values(["Target_Config", "Dataset"]))
        print(f"Hybrid results saved in the current batch output directory.")
    global_metrics = refresh_global_metrics_from_context(context, display_fn=display, show=False)
    print(f"[GLOBAL RESULTS] Updated {global_metrics['path']}")
    return {"top_variants": top_variants, "validation": validation, "results": result,
            "path": test_path, "output_dir": out_dir}
