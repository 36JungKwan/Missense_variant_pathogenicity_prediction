"""Fixed-parameter fusion × classifier benchmark with resumable artifacts."""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from core.module05v2_fusion_classifier.evaluator_profiler import FusionEvaluatorProfiler
from core.module05v2_fusion_classifier.resume_utils import (
    ensure_stage_signature,
    replace_file_with_retry,
    save_progress_csv,
    upsert_record,
)
from core.module06_architecture_experiments.architecture_experiment_runner import (
    _architecture_loader,
    _attach_token_cache,
    _atomic_torch_save,
    _move,
    _get_test_profile_metrics,
    run_architecture_experiment,
)
from core.module06_architecture_experiments.concat_3modal_model import ConcatThreeModalities
from core.module06_architecture_experiments.film_experiment import _build_film_model
from core.module06_architecture_experiments.gating_3modal_model import GatingThreeModalities
from core.module06_architecture_experiments.low_rank_experiment import _build_low_rank_model
from core.module06_architecture_experiments.router_expert_experiment import _build_router_expert_model
from core.module06_architecture_experiments.transformer_3modal_experiment import (
    _builder as _build_transformer_3modal,
)
from core.module06_architecture_experiments.cross_attention_3modal_experiment import (
    _builder as _build_cross_attention_3modal,
)


FIXED_CLASSIFIER_NAMES = ("MLP", "XGBoost", "CatBoost", "LightGBM")
FIXED_FUSION_NAMES = (
    "Concat_ThreeModalities",
    "LowRank",
    "Gating_ThreeModalities",
    "FiLM",
    "Transformer_ThreeModalities",
    "CrossAttention_ThreeModalities",
    "Router_Expert",
)
FIXED_DATASET_NAMES = (
    "Train3Val_Test", "Train3Val_ClinVarHQ", "Train3Val_ProteinGym", "Train3Val_UniProt",
)
METRIC_NAMES = (
    "Accuracy", "Precision", "Recall", "Specificity", "F1_Score",
    "MCC", "AUROC", "AUPRC",
)
PROFILE_NAMES = (
    "FM_GFLOPs/Sample", "FM_Latency_ms/Sample",
    "Geom_GFLOPs/Sample", "Geom_Latency_ms/Sample",
    "Classifier_GFLOPs/Sample", "Classifier_Latency_ms/Sample",
    "Classifier_Total_Params", "Classifier_Param_Mem_MB",
    "Classifier_XGB_Model_Mem_MB", "Classifier_Model_Mem_MB",
    "Classifier_Peak_VRAM_MB", "E2E_GFLOPs/Sample",
    "E2E_Latency_ms/Sample", "E2E_Total_Params", "E2E_Param_Mem_MB",
    "Max_Peak_VRAM_MB",
)
PROFILE_EXTRA_NAMES = (
    "Classifier_Tree_Leaves", "Classifier_GFLOPs_Note", "Classifier_Model_Mem_Scope",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 ** 2), 4) if path.is_file() else float("nan")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json_atomic(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable), encoding="utf-8")
        replace_file_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _make_architecture_config(fusion: str, shared: dict, seed: int) -> dict:
    """Fill only architecture-specific dimensions absent from the shared profile."""
    cfg = dict(shared)
    dim = int(cfg["model_dim"])
    cfg["seed"] = int(seed)
    if fusion == "LowRank":
        cfg.update(hidden_dim=dim, fusion_dim=dim, rank=8, head_dim=max(32, dim // 4))
    elif fusion == "FiLM":
        cfg.update(hidden_dim=dim, bio_hidden_dim=dim, head_dim=max(32, dim // 4))
    elif fusion == "Router_Expert":
        cfg.update(
            embedding_dim=dim,
            expert_hidden_dim=dim,
            router_hidden_dim=max(32, dim // 4),
            correction_hidden_dim=dim,
            router_temperature=1.0,
            beta_init=0.0,
        )
    return cfg


def _model_builder(fusion: str):
    if fusion == "Concat_ThreeModalities":
        return lambda dims, cfg: ConcatThreeModalities(
            dims["dna"], dims["protein"], dims["bio"], dims["geometry"], cfg
        )
    if fusion == "LowRank":
        return _build_low_rank_model
    if fusion == "Gating_ThreeModalities":
        return lambda dims, cfg: GatingThreeModalities(
            dims["dna"], dims["protein"], dims["bio"], dims["geometry"], cfg
        )
    if fusion == "FiLM":
        return _build_film_model
    if fusion == "Transformer_ThreeModalities":
        return _build_transformer_3modal
    if fusion == "CrossAttention_ThreeModalities":
        return _build_cross_attention_3modal
    if fusion == "Router_Expert":
        return _build_router_expert_model
    raise KeyError(f"Unknown fixed fusion strategy: {fusion}")


def load_legacy_best_classifier_params(ml_batch_dir, target_config="B_with_geometry"):
    """Load each legacy tree model's highest-validation-MCC no-PCA trial."""
    root = Path(ml_batch_dir)
    candidates = {
        "XGBoost": root / "xgb_no_pca" / "xgb_no_pca_grid_trials.csv",
        "LightGBM": root / "lightgbm" / "lightgbm_grid_trials.csv",
        "CatBoost": root / "catboost" / "catboost_grid_trials.csv",
    }
    selected = {}
    for classifier, csv_path in candidates.items():
        if not csv_path.is_file():
            raise FileNotFoundError(f"Legacy {classifier} trial CSV is missing: {csv_path}")
        frame = pd.read_csv(csv_path)
        if "Target_Config" not in frame or "Val_MCC" not in frame or "Parameters" not in frame:
            raise ValueError(f"Legacy trial CSV has an unexpected schema: {csv_path}")
        subset = frame[frame["Target_Config"].astype(str).eq(target_config)].copy()
        if "PCA" in subset:
            subset = subset[~subset["PCA"].astype(str).str.lower().isin({"true", "1"})]
        if "Branch" in subset:
            subset = subset[subset["Branch"].astype(str).str.lower().isin({"nopca", "nan"})]
        subset["Val_MCC"] = pd.to_numeric(subset["Val_MCC"], errors="coerce")
        subset = subset.dropna(subset=["Val_MCC", "Parameters"])
        if subset.empty:
            raise ValueError(f"No B/no-PCA completed trials found for {classifier}: {csv_path}")
        row = subset.sort_values(["Val_MCC", "Trial"], ascending=[False, True]).iloc[0]
        params = json.loads(str(row["Parameters"]))
        selected[classifier] = {
            "params": params,
            "source_csv": str(csv_path.resolve()),
            "source_trial": str(row["Trial"]),
            "source_val_mcc": float(row["Val_MCC"]),
            "source_pca": False,
            "source_target_config": target_config,
        }
    return selected


def _feature_args(model, batch, device, token_aware):
    common = [
        _move(batch, "v_dna", device, required=True),
        _move(batch, "v_prot", device, required=True),
        _move(batch, "bio_features", device, required=True),
        _move(batch, "geom_features", device),
    ]
    if not token_aware:
        return common
    common.extend(_move(batch, key, device, required=True) for key in (
        "token_dna", "token_dna_mask", "token_dna_variant_index",
        "token_protein", "token_protein_mask", "token_protein_variant_index",
    ))
    return common


def _extract_features(model, loader, device, token_aware):
    model.eval()
    features, labels, variant_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            encoded = model.encode_features(*_feature_args(model, batch, device, token_aware))
            features.append(encoded.detach().float().cpu())
            labels.append(batch["label"].reshape(-1).cpu().to(torch.int64))
            variant_ids.extend([str(value) for value in batch["variant_id"]])
    return {
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "variant_ids": variant_ids,
    }


def _fit_fixed_tree(classifier, params, seed, x_train, y_train, x_val, y_val):
    effective = dict(params)
    effective.pop("random_state", None)
    effective.pop("random_seed", None)
    if classifier == "XGBoost":
        from xgboost import XGBClassifier
        model = XGBClassifier(**effective, random_state=int(seed))
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    elif classifier == "LightGBM":
        import lightgbm as lgb
        effective["random_state"] = int(seed)
        model = lgb.LGBMClassifier(**effective)
        model.fit(
            x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
    elif classifier == "CatBoost":
        from catboost import CatBoostClassifier
        effective["random_seed"] = int(seed)
        model = CatBoostClassifier(**effective)
        model.fit(x_train, y_train, eval_set=(x_val, y_val),
                  early_stopping_rounds=120, verbose=False)
    else:
        raise KeyError(f"Unsupported fixed tree classifier: {classifier}")
    return model


def _predict_proba(classifier, model, features):
    if classifier == "LightGBM" and hasattr(model, "predict_proba") and getattr(model, "best_iteration_", None):
        return model.predict_proba(features, num_iteration=model.best_iteration_)[:, 1]
    if classifier == "LightGBM" and not hasattr(model, "predict_proba"):
        return np.asarray(model.predict(features), dtype=np.float32)
    return model.predict_proba(features)[:, 1]


def _save_tree_model(classifier, model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if classifier == "XGBoost":
        model.save_model(str(temporary))
    elif classifier == "LightGBM":
        model.booster_.save_model(str(temporary))
    else:
        model.save_model(str(temporary))
    replace_file_with_retry(temporary, path)


def _load_tree_model(classifier, path):
    if classifier == "XGBoost":
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(str(path))
        return model
    if classifier == "LightGBM":
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path))
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def _tree_predict_proba(classifier, model, features):
    if classifier == "LightGBM":
        return model.predict(features)
    return model.predict_proba(features)[:, 1]


def _compute_metrics(pipeline, labels, probabilities, threshold):
    predictions = pipeline._threshold_predictions(probabilities, threshold)
    return FusionEvaluatorProfiler.compute_metrics(labels, probabilities, predictions)


def _tree_node_count(classifier, model):
    try:
        if classifier == "XGBoost":
            table = model.get_booster().trees_to_dataframe()
            return int(table["Feature"].astype(str).eq("Leaf").sum())
        if classifier == "LightGBM":
            return int(sum(int(tree["num_leaves"]) for tree in model.dump_model()["tree_info"]))
        return int(sum(int(value) for value in model.get_tree_leaf_counts()))
    except Exception:
        return np.nan


def _profile_tree_pipeline(context, model, loader, tree, classifier, encoder_checkpoint, tree_path,
                           dimensions, device, token_aware, active_modalities, split_name):
    """Profile the neural feature encoder plus tree predictor; tree FLOPs are not fabricated."""
    model.eval()
    elapsed = 0.0
    n_samples = 0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch in loader:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            z = model.encode_features(*_feature_args(model, batch, device, token_aware))
            z_np = z.detach().float().cpu().numpy()
            _tree_predict_proba(classifier, tree, z_np)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            n_samples += int(z.shape[0])

    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )
    params = sum(int(parameter.numel()) for parameter in model.parameters())
    parameter_memory = sum(
        int(parameter.numel()) * int(parameter.element_size())
        for parameter in model.parameters()
    ) / (1024 ** 2)
    fm_profiler = FusionEvaluatorProfiler(
        tensorboard_dir=None,
        fm_profile_json=getattr(context["pipeline"], "fm_profile_json", None),
        geom_profile_json=getattr(context["pipeline"], "geom_profile_json", None),
        device=device,
    )
    fm_profiler.fusion_metrics.update({
        "gflops_per_sample": 0.0,
        "inference_time_ms": round(elapsed * 1000 / max(n_samples, 1), 4),
        "num_parameters": params,
        "param_memory_mb": round(parameter_memory, 4),
        "peak_memory_mb": round(peak_vram, 4),
        # This measurement is specific to XGBoost; keep other classifiers N/A
        # instead of reporting a misleading zero-sized XGBoost artifact.
        "xgb_model_memory_mb": _size_mb(tree_path) if classifier == "XGBoost" else np.nan,
    })
    stats = fm_profiler.get_e2e_profiling(
        dna_model_name=context["DNA_MODEL"]["name"],
        prot_model_name=context["PROTEIN_MODEL"]["name"],
        active_modalities=active_modalities,
        split_name=split_name,
        pooling="center",
    )
    fm_profiler.close()
    tree_memory = _size_mb(tree_path)
    stats.update({
        "Classifier_GFLOPs/Sample": np.nan,
        "Classifier_Latency_ms/Sample": round(elapsed * 1000 / max(n_samples, 1), 4),
        "Classifier_Total_Params": params,
        "Classifier_Param_Mem_MB": round(parameter_memory, 4),
        "Classifier_XGB_Model_Mem_MB": tree_memory if classifier == "XGBoost" else np.nan,
        "Classifier_Model_Mem_MB": round(_size_mb(encoder_checkpoint) + tree_memory, 4),
        "Classifier_Peak_VRAM_MB": round(peak_vram, 4),
        "E2E_GFLOPs/Sample": np.nan,
        "E2E_Latency_ms/Sample": round(
            float(stats.get("FM_Latency_ms/Sample", 0.0))
            + float(stats.get("Geom_Latency_ms/Sample", 0.0))
            + elapsed * 1000 / max(n_samples, 1), 4,
        ),
        "E2E_Total_Params": int(stats.get("E2E_Total_Params", 0)),
        "E2E_Param_Mem_MB": round(float(stats.get("E2E_Param_Mem_MB", 0.0)), 4),
        "Max_Peak_VRAM_MB": round(max(float(stats.get("Max_Peak_VRAM_MB", 0.0)), peak_vram), 4),
        "Classifier_Tree_Leaves": _tree_node_count(classifier, tree),
        "Classifier_GFLOPs_Note": "Tree operations are not represented as neural-network FLOPs.",
        "Classifier_Model_Mem_Scope": "fusion encoder checkpoint + serialized tree model",
    })
    return {name: stats.get(name, np.nan) for name in PROFILE_NAMES} | {
        "Classifier_Tree_Leaves": stats["Classifier_Tree_Leaves"],
        "Classifier_GFLOPs_Note": stats["Classifier_GFLOPs_Note"],
        "Classifier_Model_Mem_Scope": stats["Classifier_Model_Mem_Scope"],
    }


def _load_or_extract_features(context, fusion, seed, model, builder, config, token_aware,
                              target, split_name, is_train, device, feature_root):
    pipeline = context["pipeline"]
    cache_dir = feature_root / fusion / f"seed_{seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split_name}.pt"
    signature_path = cache_dir / f"{split_name}_signature.json"
    token_index = context.get("token_embedding_cache_index")
    signature = {
        "fusion": fusion,
        "seed": int(seed),
        "split": split_name,
        "target": target,
        "embedding_representation": "ref_alt_concat",
        "model_config": config,
        "token_cache_index": token_index.get("index_path") if token_index is not None else None,
    }
    ensure_stage_signature(
        cache_dir / f"{split_name}_manifest",
        f"features_{fusion}_{split_name}",
        signature,
        mutable_paths=(("model_config", "batch_size"),)
        if context.get("allow_batch_size_resume") else (),
    )
    if cache_path.is_file():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    loader = _architecture_loader(
        pipeline, context, target, split_name, "center", context["DNA_MODEL"],
        context["PROTEIN_MODEL"], False, "train3", int(seed), "ref_alt_concat",
    )
    loader = _attach_token_cache(
        loader, context, split_name, is_train=False,
        window_radius=context.get("token_window_radius") if token_aware else None,
    )
    bundle = _extract_features(model, loader, device, token_aware)
    _atomic_torch_save(bundle, cache_path)
    return bundle


def _load_head_results(path):
    return pd.read_csv(path).to_dict("records") if Path(path).is_file() else []


def _seed_key(value):
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def _prediction_group_counts(rows):
    return Counter((
        str(row.get("Fusion")), str(row.get("Classifier")),
        _seed_key(row.get("Seed")), str(row.get("Dataset")),
    ) for row in rows)


def _incoming_prediction_counts(frame, fusion, classifier, seed):
    if frame.empty or "Dataset" not in frame:
        return {}
    return {
        (str(fusion), str(classifier), _seed_key(seed), str(dataset)): int(count)
        for dataset, count in frame.groupby("Dataset", dropna=False).size().items()
    }


def _prediction_groups_present(existing_counts, incoming_counts):
    return bool(incoming_counts) and all(
        existing_counts.get(key, 0) >= count
        for key, count in incoming_counts.items()
    )


def _bulk_upsert_rows(existing_rows, incoming_rows, keys):
    """Vectorized replacement for per-row upsert on large prediction CSVs."""
    incoming = pd.DataFrame(incoming_rows)
    if incoming.empty:
        return existing_rows
    existing = pd.DataFrame(existing_rows)
    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    return combined.drop_duplicates(subset=keys, keep="last").to_dict("records")


def _saved_head_is_complete(head_dir, target_name, datasets):
    """Check persisted tree-head artifacts without loading feature tensors."""
    head_dir = Path(head_dir)
    model_files = [
        head_dir / "best_model.json",
        head_dir / "best_model.txt",
        head_dir / "best_model.cbm",
    ]
    if not any(path.is_file() for path in model_files):
        return False
    val_rows = _load_head_results(head_dir / "validation_metrics.csv")
    if not any(str(row.get("Target_Config")) == str(target_name) for row in val_rows):
        return False
    test_rows = _load_head_results(head_dir / "test_metrics.csv")
    prediction_rows = _load_head_results(head_dir / "test_predictions.csv")
    for dataset_cfg in datasets:
        dataset_name = str(dataset_cfg["name"])
        metric = next((row for row in test_rows if
                       str(row.get("Target_Config")) == str(target_name)
                       and str(row.get("Dataset")) == dataset_name), None)
        if metric is None or pd.isna(metric.get("MCC")):
            return False
        prediction_count = sum(
            1 for row in prediction_rows
            if str(row.get("Target_Config")) == str(target_name)
            and str(row.get("Dataset")) == dataset_name
        )
        expected_count = (
            int(float(metric["N_Test_Samples"]))
            if pd.notna(metric.get("N_Test_Samples")) else 1
        )
        if prediction_count < expected_count:
            return False
    return True


def _run_tree_head(context, fusion, seed, classifier, legacy_record, encoder_model,
                   encoder_dir, feature_root, token_aware, target, feature_bundles=None,
                   display_fn=None):
    pipeline = context["pipeline"]
    device = torch.device(context.get("DEVICE", pipeline.device))
    params = dict(legacy_record["params"])
    if classifier == "XGBoost":
        params["random_state"] = int(seed)
        extension = ".json"
    elif classifier == "LightGBM":
        params["random_state"] = int(seed)
        extension = ".txt"
    else:
        params["random_seed"] = int(seed)
        extension = ".cbm"

    target_name = str(target["name"])
    head_dir = Path(pipeline.batch_run_dir) / "module06_architecture_experiments" / "fusion_classifier_benchmark" / "heads" / fusion / classifier / f"seed_{seed}"
    head_dir.mkdir(parents=True, exist_ok=True)
    ensure_stage_signature(head_dir, f"fixed_head_{fusion}_{classifier}_seed_{seed}", {
        "fusion": fusion, "classifier": classifier, "seed": int(seed),
        "target_config": target, "classifier_parameters": params,
        "legacy_parameter_source": legacy_record,
        "encoder_checkpoint": str((encoder_dir / "checkpoints" / target_name / "best_model.pth").resolve()),
        "feature_representation": "frozen_fusion_encode_features_ref_alt_concat",
    })
    model_path = head_dir / f"best_model{extension}"
    val_path = head_dir / "validation_metrics.csv"
    test_path = head_dir / "test_metrics.csv"
    predictions_path = head_dir / "test_predictions.csv"
    val_rows = _load_head_results(val_path)
    test_rows = _load_head_results(test_path)
    prediction_rows = _load_head_results(predictions_path)

    # A finished classifier needs neither its latent feature cache nor a
    # profiling loader on resume. Return its saved records directly.
    if _saved_head_is_complete(head_dir, target_name, context["datasets"]):
        print(f"[RESUME] {fusion}/{classifier}/seed_{seed} is complete; skipping feature and dataset loading.")
        return {
            "validation_metrics": pd.DataFrame(val_rows),
            "test_metrics": pd.DataFrame(test_rows),
            "model_path": str(model_path),
            "output_dir": str(head_dir),
        }

    if feature_bundles is None:
        feature_bundles = {}
    train_data = feature_bundles.get("train3")
    if train_data is None:
        train_data = torch.load(
            feature_root / fusion / f"seed_{seed}" / "train3.pt",
            map_location="cpu", weights_only=False,
        )
    val_data = feature_bundles.get("val")
    if val_data is None:
        val_data = torch.load(
            feature_root / fusion / f"seed_{seed}" / "val.pt",
            map_location="cpu", weights_only=False,
        )
    x_train = train_data["features"].numpy()
    y_train = train_data["labels"].numpy()
    x_val = val_data["features"].numpy()
    y_val = val_data["labels"].numpy()

    val_record = next((r for r in val_rows if r.get("Target_Config") == target_name), None)
    if val_record is None or not model_path.is_file():
        _seed_everything(int(seed))
        tree = _fit_fixed_tree(classifier, params, seed, x_train, y_train, x_val, y_val)
        _save_tree_model(classifier, tree, model_path)
        p_val = _predict_proba(classifier, tree, x_val)
        threshold, val_mcc = pipeline._select_validation_threshold(y_val, p_val)
        val_metrics = _compute_metrics(pipeline, y_val, p_val, threshold)
        val_record = {
            "Target_Config": target_name,
            "Fusion": fusion,
            "Classifier": classifier,
            "Seed": int(seed),
            "Batch_Size": int(pipeline.config.get("batch_size", 0)),
            "Input_Representation": "ref_alt_concat",
            "Decision_Threshold": float(threshold),
            "Val_MCC": float(val_mcc),
            "Best_Iteration": int(getattr(tree, "best_iteration", getattr(tree, "best_iteration_", -1)) or -1),
            "Hyperparameters": json.dumps(params, sort_keys=True),
            "Legacy_Parameter_Source": legacy_record["source_csv"],
            "Legacy_Parameter_Trial": legacy_record["source_trial"],
            "Legacy_Parameter_Val_MCC": legacy_record["source_val_mcc"],
            **{f"Val_{key}": value for key, value in val_metrics.items()},
        }
        val_rows = upsert_record(val_rows, val_record, ["Target_Config"])
        save_progress_csv(val_rows, val_path)
    else:
        tree = _load_tree_model(classifier, model_path)
        print(f"[RESUME] Reusing fixed {classifier} head: {fusion}/seed_{seed}")

    threshold = float(val_record["Decision_Threshold"])
    profile_metrics = None
    profile_dataset = next((ds for ds in context["datasets"] if ds["name"] == "Train3Val_Test"), context["datasets"][0])
    profile_loader = _architecture_loader(
        pipeline, context, target, profile_dataset["test"], "center", context["DNA_MODEL"],
        context["PROTEIN_MODEL"], False, "train3", int(seed), "ref_alt_concat",
    )
    profile_loader = _attach_token_cache(
        profile_loader, context, profile_dataset["test"], is_train=False,
        window_radius=context.get("token_window_radius") if token_aware else None,
    )
    encoder_checkpoint = encoder_dir / "checkpoints" / target_name / "best_model.pth"
    existing_profile_row = next(
        (row for row in test_rows if row.get("Dataset") == profile_dataset["name"]), None
    )
    if existing_profile_row is None or pd.isna(existing_profile_row.get("Classifier_Latency_ms/Sample")):
        profile_metrics = _profile_tree_pipeline(
            context, encoder_model, profile_loader, tree, classifier, encoder_checkpoint,
            model_path, {}, device, token_aware, target["active_modalities"], profile_dataset["test"],
        )
    else:
        profile_metrics = {key: existing_profile_row.get(key, np.nan) for key in PROFILE_NAMES}
        profile_metrics.update({key: existing_profile_row.get(key, np.nan) for key in (
            "Classifier_Tree_Leaves", "Classifier_GFLOPs_Note", "Classifier_Model_Mem_Scope")})

    for dataset_cfg in context["datasets"]:
        dataset_name = str(dataset_cfg["name"])
        split = str(dataset_cfg["test"])
        test_data = feature_bundles.get(split)
        if test_data is None:
            test_data = torch.load(
                feature_root / fusion / f"seed_{seed}" / f"{split}.pt",
                map_location="cpu", weights_only=False,
            )
        x_test = test_data["features"].numpy()
        y_test = test_data["labels"].numpy()
        row_exists = any(
            row.get("Dataset") == dataset_name and row.get("Target_Config") == target_name
            for row in test_rows
        )
        existing_prediction_ids = {
            str(row.get("Variant_ID")) for row in prediction_rows
            if row.get("Dataset") == dataset_name and row.get("Target_Config") == target_name
        }
        if row_exists and len(existing_prediction_ids) >= len(y_test):
            continue
        p_test = _predict_proba(classifier, tree, x_test)
        if not row_exists:
            metrics = _compute_metrics(pipeline, y_test, p_test, threshold)
            row = {
                "Run_ID": Path(pipeline.batch_run_dir).name,
                "Branch": "Fusion_Classifier_Benchmark",
                "Target_Config": target_name,
                "Ablation": target["ablation"],
                "Dataset": dataset_name,
                "Fusion": fusion,
                "Architecture": fusion,
                "Classifier": classifier,
                "Seed": int(seed),
                "Batch_Size": int(pipeline.config.get("batch_size", 0)),
                "Input_Representation": "ref_alt_concat",
                "Decision_Threshold": threshold,
                "N_Test_Samples": len(y_test),
                "Hyperparameters": json.dumps(params, sort_keys=True),
                "Legacy_Parameter_Source": legacy_record["source_csv"],
                "Legacy_Parameter_Trial": legacy_record["source_trial"],
                "Legacy_Parameter_Val_MCC": legacy_record["source_val_mcc"],
                **metrics,
                **(profile_metrics or {}),
            }
            test_rows = upsert_record(test_rows, row, ["Target_Config", "Dataset"])
            save_progress_csv(test_rows, test_path)
        prediction_records = [{
            "Target_Config": target_name, "Dataset": dataset_name,
            "Fusion": fusion, "Classifier": classifier, "Seed": int(seed),
            "Variant_ID": str(variant_id), "Label": int(label),
            "Probability_Pathogenic": float(probability),
            "Prediction": int(probability >= threshold),
        } for variant_id, label, probability in zip(test_data["variant_ids"], y_test, p_test)]
        for prediction_record in prediction_records:
            prediction_rows = upsert_record(
                prediction_rows, prediction_record,
                ["Target_Config", "Dataset", "Fusion", "Classifier", "Seed", "Variant_ID"],
            )
        save_progress_csv(prediction_rows, predictions_path)
        if display_fn:
            display_fn(f"[TEST] {fusion}/{classifier}/seed_{seed} {dataset_name}: MCC={metrics['MCC']:.4f}")
    return {"validation_metrics": pd.DataFrame(val_rows), "test_metrics": pd.DataFrame(test_rows),
            "model_path": str(model_path), "output_dir": str(head_dir)}


def _summary(frame, group_columns, validation=False):
    if frame.empty:
        return pd.DataFrame()
    numeric = frame.copy()
    value_columns = {}
    for metric in METRIC_NAMES:
        source = f"Val_{metric}" if validation and f"Val_{metric}" in numeric.columns else metric
        if source in numeric.columns:
            value_columns[source] = f"Val_{metric}" if validation else metric
    if not validation:
        for metric in (*PROFILE_NAMES, "Classifier_Tree_Leaves"):
            if metric in numeric.columns:
                value_columns[metric] = metric
    if not value_columns:
        return numeric[group_columns].drop_duplicates().reset_index(drop=True)
    for column in value_columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    grouped = numeric.groupby(group_columns, dropna=False)[list(value_columns)].agg(["mean", "std", "count"])
    grouped.columns = [f"{value_columns[metric]}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def run_fixed_fusion_classifier_benchmark(context, shared_config, seeds,
                                          legacy_classifier_params,
                                          display_fn=None):
    """Run 7 fixed fusion encoders × 4 classifiers over the configured seeds."""
    targets = context.get("TARGET_CONFIGS", [])
    if len(targets) != 1 or targets[0].get("name") != "B_with_geometry":
        raise ValueError("This benchmark is intentionally limited to B_with_geometry.")
    if not targets[0].get("geom_feature_names"):
        raise ValueError("B_with_geometry must include the configured geometry features.")
    if context.get("token_embedding_cache_index") is None:
        raise KeyError("A complete token_embedding_cache_index is required for Transformer/Cross-Attention.")
    if set(legacy_classifier_params) != {"XGBoost", "LightGBM", "CatBoost"}:
        raise ValueError("Pass fixed legacy best parameters for XGBoost, LightGBM, and CatBoost.")

    pipeline = context["pipeline"]
    device = torch.device(context.get("DEVICE", pipeline.device))
    display = display_fn or print
    output_root = Path(pipeline.batch_run_dir) / "module06_architecture_experiments" / "fusion_classifier_benchmark"
    output_root.mkdir(parents=True, exist_ok=True)
    run_signature = {
        "target_configs": targets,
        "datasets": context["datasets"],
        "dna_model": context["DNA_MODEL"],
        "protein_model": context["PROTEIN_MODEL"],
        "seeds": [int(seed) for seed in seeds],
        "fixed_model_config": shared_config,
        "legacy_classifier_params": legacy_classifier_params,
        "input_data_signature": context.get("input_data_signature"),
        "pretrained_encoder_checkpoints": context.get("pretrained_encoder_checkpoints"),
        "embedding_representation": "ref_alt_concat",
        "optuna_enabled": False,
    }
    ensure_stage_signature(
        output_root,
        "module06_fixed_fusion_classifier_benchmark",
        run_signature,
        mutable_paths=(("fixed_model_config", "batch_size"),),
    )
    run_configuration_path = output_root / "run_configuration.json"
    previous_configuration = (
        json.loads(run_configuration_path.read_text(encoding="utf-8"))
        if run_configuration_path.is_file() else {}
    )
    batch_size = int(pipeline.config.get("batch_size", shared_config.get("batch_size", 0)))
    batch_size_history = list(previous_configuration.get("batch_size_history", []))
    if not batch_size_history:
        previous_batch_size = previous_configuration.get("current_batch_size")
        if previous_batch_size is None:
            previous_batch_size = previous_configuration.get("fixed_model_config", {}).get("batch_size")
        if previous_batch_size is not None:
            batch_size_history.append({
                "batch_size": int(previous_batch_size),
                "recorded_at_utc": "recorded_before_batch_size_history",
            })
    if not batch_size_history or int(batch_size_history[-1]["batch_size"]) != batch_size:
        batch_size_history.append({
            "batch_size": batch_size,
            "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    run_configuration = dict(run_signature)
    run_configuration["current_batch_size"] = batch_size
    run_configuration["batch_size_history"] = batch_size_history
    run_configuration["batch_size_change_allowed"] = True
    _write_json_atomic(run_configuration, run_configuration_path)
    feature_root = output_root / "feature_cache"
    combined_val_path = output_root / "seed_level_validation_metrics.csv"
    combined_test_path = output_root / "seed_level_test_metrics.csv"
    combined_predictions_path = output_root / "seed_level_test_predictions.csv"
    val_rows = _load_head_results(combined_val_path)
    test_rows = _load_head_results(combined_test_path)
    prediction_rows = _load_head_results(combined_predictions_path)
    prediction_counts = _prediction_group_counts(prediction_rows)

    for fusion in FIXED_FUSION_NAMES:
        builder = _model_builder(fusion)
        token_aware = fusion in {"Transformer_ThreeModalities", "CrossAttention_ThreeModalities"}
        fusion_context = dict(context)
        if token_aware:
            fusion_context["token_window_radius"] = int(shared_config["k_radius"])
        else:
            # Vector fusion models use center-pooled ref/alt vectors; no token payload is attached.
            fusion_context["token_embedding_cache_index"] = None
            fusion_context.pop("token_window_radius", None)
        for seed in [int(value) for value in seeds]:
            config = _make_architecture_config(fusion, shared_config, seed)
            architecture_key = f"fusion_classifier_benchmark/encoders/{fusion}/seed_{seed}"
            pretrained = (
                context.get("pretrained_encoder_checkpoints", {}).get(seed)
                if fusion == "CrossAttention_ThreeModalities" else None
            )
            if fusion == "CrossAttention_ThreeModalities" and pretrained is None:
                raise KeyError(f"No pretrained Cross-Attention checkpoint was provided for seed {seed}.")
            run_result = run_architecture_experiment(
                context=fusion_context,
                experiment_config=config,
                model_builder=builder,
                architecture=fusion,
                variant="fixed_ref_alt_concat_b_geometry",
                experiment_key=architecture_key,
                source="module06_fusion_classifier_benchmark",
                display_fn=lambda *_args, **_kwargs: None,
                manifest_architecture=f"fixed_{fusion}_ref_alt_concat_B_geometry_v1",
                embedding_representation="ref_alt_concat",
                evaluate_test=True,
                pretrained_checkpoint=pretrained["checkpoint"] if pretrained else None,
                pretrained_best_epoch=pretrained.get("best_epoch") if pretrained else None,
            )
            encoder_dir = Path(run_result["output_dir"])
            target = targets[0]
            target_name = str(target["name"])
            checkpoint = encoder_dir / "checkpoints" / target_name / "best_model.pth"

            mlp_rows = run_result["test_metrics"].copy()
            if not mlp_rows.empty:
                mlp_rows["Fusion"] = fusion
                mlp_rows["Architecture"] = fusion
                mlp_rows["Classifier"] = "MLP"
                mlp_rows["Seed"] = seed
                mlp_rows["Input_Representation"] = "ref_alt_concat"
                mlp_rows["Classifier_Model_Mem_MB"] = _size_mb(checkpoint)
                for _, row in mlp_rows.iterrows():
                    test_rows = upsert_record(
                        test_rows, row.to_dict(), ["Fusion", "Classifier", "Seed", "Dataset"]
                    )
                save_progress_csv(test_rows, combined_test_path)
            mlp_predictions = run_result["test_predictions"].copy()
            if not mlp_predictions.empty:
                mlp_predictions["Fusion"] = fusion
                mlp_predictions["Classifier"] = "MLP"
                mlp_predictions["Seed"] = seed
                incoming_counts = _incoming_prediction_counts(
                    mlp_predictions, fusion, "MLP", seed
                )
                if _prediction_groups_present(prediction_counts, incoming_counts):
                    print(f"[RESUME] {fusion}/MLP/seed_{seed}: global test predictions already synchronized.")
                else:
                    print(f"[SYNC] Merging saved MLP test predictions for {fusion}/seed_{seed}...")
                    prediction_rows = _bulk_upsert_rows(
                        prediction_rows, mlp_predictions.to_dict("records"),
                        ["Fusion", "Classifier", "Seed", "Dataset", "Variant_ID"],
                    )
                    prediction_counts = _prediction_group_counts(prediction_rows)
                    save_progress_csv(prediction_rows, combined_predictions_path)
            mlp_val = run_result["validation_metrics"].copy()
            if not mlp_val.empty:
                mlp_val["Fusion"] = fusion
                mlp_val["Architecture"] = fusion
                mlp_val["Classifier"] = "MLP"
                mlp_val["Seed"] = seed
                for _, row in mlp_val.iterrows():
                    val_rows = upsert_record(
                        val_rows, row.to_dict(), ["Fusion", "Classifier", "Seed", "Target_Config"]
                    )
                save_progress_csv(val_rows, combined_val_path)

            head_root = output_root / "heads"
            tree_classifiers = ("XGBoost", "CatBoost", "LightGBM")
            incomplete_tree_classifiers = [
                classifier for classifier in tree_classifiers
                if not _saved_head_is_complete(
                    head_root / fusion / classifier / f"seed_{seed}",
                    target_name, context["datasets"],
                )
            ]
            model = None
            feature_bundles = {}
            if incomplete_tree_classifiers:
                dimensions_path = encoder_dir / "checkpoints" / target_name / "input_dimensions.json"
                if dimensions_path.is_file():
                    dimensions = json.loads(dimensions_path.read_text(encoding="utf-8"))
                else:
                    # Compatibility fallback for checkpoints created before the
                    # input dimensions were saved beside the encoder checkpoint.
                    train_loader = _architecture_loader(
                        pipeline, fusion_context, target, "train3", "center", context["DNA_MODEL"],
                        context["PROTEIN_MODEL"], False, "train3", seed, "ref_alt_concat",
                    )
                    train_loader = _attach_token_cache(
                        train_loader, fusion_context, "train3", is_train=False,
                        window_radius=shared_config.get("k_radius") if token_aware else None,
                    )
                    train_dataset = train_loader.dataset
                    dimensions = {
                        "dna": int(train_dataset.v_dna_tensor.shape[1]),
                        "protein": int(train_dataset.v_prot_tensor.shape[1]),
                        "bio": int(train_dataset.bio_tensor.shape[1]),
                        "geometry": int(train_dataset.geom_tensor.shape[1]) if train_dataset.has_geom else 0,
                        "active_modalities": list(target["active_modalities"]),
                    }
                    if token_aware:
                        dimensions.update({
                            "token_dna": int(train_dataset.token_hidden_sizes["dna"]),
                            "token_protein": int(train_dataset.token_hidden_sizes["protein"]),
                            "token_cache": True,
                        })
                    dimensions_path.parent.mkdir(parents=True, exist_ok=True)
                    dimensions_path.write_text(
                        json.dumps(dimensions, indent=2, sort_keys=True), encoding="utf-8"
                    )
                    del train_loader

                model = builder(dimensions, config).to(device)
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                model.eval()
                for split_name, is_train in [("train3", False), ("val", False)] + [
                    (str(item["test"]), False) for item in context["datasets"]
                ]:
                    feature_bundles[split_name] = _load_or_extract_features(
                        fusion_context, fusion, seed, model, builder, config, token_aware,
                        target, split_name, is_train, device, feature_root,
                    )

            for classifier in ("XGBoost", "CatBoost", "LightGBM"):
                legacy_record = legacy_classifier_params[classifier]
                head_result = _run_tree_head(
                    fusion_context, fusion, seed, classifier, legacy_record, model,
                    encoder_dir, feature_root, token_aware, target,
                    feature_bundles=feature_bundles, display_fn=display,
                )
                for _, row in head_result["validation_metrics"].iterrows():
                    row_data = row.to_dict()
                    row_data.update({"Fusion": fusion, "Architecture": fusion,
                                     "Classifier": classifier, "Seed": seed})
                    val_rows = upsert_record(
                        val_rows, row_data, ["Fusion", "Classifier", "Seed", "Target_Config"]
                    )
                save_progress_csv(val_rows, combined_val_path)
                for _, row in head_result["test_metrics"].iterrows():
                    row_data = row.to_dict()
                    row_data.update({"Fusion": fusion, "Architecture": fusion,
                                     "Classifier": classifier, "Seed": seed})
                    test_rows = upsert_record(
                        test_rows, row_data, ["Fusion", "Classifier", "Seed", "Dataset"]
                    )
                save_progress_csv(test_rows, combined_test_path)
                head_predictions = _load_head_results(
                    Path(head_result["output_dir"]) / "test_predictions.csv"
                )
                head_prediction_frame = pd.DataFrame(head_predictions)
                incoming_counts = _incoming_prediction_counts(
                    head_prediction_frame, fusion, classifier, seed
                )
                if _prediction_groups_present(prediction_counts, incoming_counts):
                    print(f"[RESUME] {fusion}/{classifier}/seed_{seed}: global test predictions already synchronized.")
                elif not head_prediction_frame.empty:
                    print(f"[SYNC] Merging saved {classifier} test predictions for {fusion}/seed_{seed}...")
                    prediction_rows = _bulk_upsert_rows(
                        prediction_rows, head_predictions,
                        ["Fusion", "Classifier", "Seed", "Dataset", "Variant_ID"],
                    )
                    prediction_counts = _prediction_group_counts(prediction_rows)
                    save_progress_csv(prediction_rows, combined_predictions_path)

    val_frame = pd.DataFrame(val_rows)
    test_frame = pd.DataFrame(test_rows)
    summary_root = output_root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    val_summary = _summary(val_frame, ["Target_Config", "Fusion", "Classifier"], validation=True)
    test_summary = _summary(test_frame, ["Target_Config", "Fusion", "Classifier", "Dataset"])
    profile_columns = [c for c in (*PROFILE_NAMES, *PROFILE_EXTRA_NAMES) if c in test_frame.columns]
    profile_frame = test_frame[[c for c in ["Target_Config", "Fusion", "Classifier", "Seed", *profile_columns] if c in test_frame.columns]]
    profile_frame = profile_frame.drop_duplicates(["Target_Config", "Fusion", "Classifier", "Seed"])
    profile_summary = _summary(profile_frame, ["Target_Config", "Fusion", "Classifier"])
    save_progress_csv(val_summary.to_dict("records"), summary_root / "validation_mean_std_by_fusion_classifier.csv")
    save_progress_csv(test_summary.to_dict("records"), summary_root / "test_mean_std_by_fusion_classifier_dataset.csv")
    save_progress_csv(profile_frame.to_dict("records"), summary_root / "profiling_by_seed.csv")
    save_progress_csv(profile_summary.to_dict("records"), summary_root / "profiling_mean_std_by_fusion_classifier.csv")

    inventory = _build_inventory(test_frame, seeds, [item["name"] for item in context["datasets"]])
    save_progress_csv(inventory.to_dict("records"), summary_root / "completion_inventory.csv")
    expected = len(FIXED_FUSION_NAMES) * len(FIXED_CLASSIFIER_NAMES) * len(seeds)
    display(f"Fixed benchmark has {len(test_frame)} test rows; expected complete combinations: {expected} model/seed pairs.")
    if display_fn:
        display_fn(test_summary)
        display_fn(profile_summary)
        display_fn(inventory)
    return {
        "output_root": str(output_root),
        "validation_metrics": val_frame,
        "test_metrics": test_frame,
        "validation_mean_std": val_summary,
        "test_mean_std": test_summary,
        "profiling_by_seed": profile_frame,
        "profiling_mean_std": profile_summary,
        "completion_inventory": inventory,
    }


def _build_inventory(test_frame, seeds, dataset_names=FIXED_DATASET_NAMES):
    expected_seeds = len(seeds)
    keys = ["Fusion", "Classifier", "Dataset"]
    observed = (
        test_frame.groupby(keys, dropna=False)["Seed"].nunique().to_dict()
        if not test_frame.empty else {}
    )
    inventory = pd.DataFrame([
        {"Fusion": fusion, "Classifier": classifier, "Dataset": dataset,
         "Completed_Seeds": int(observed.get((fusion, classifier, dataset), 0)),
         "Expected_Seeds": expected_seeds}
        for fusion in FIXED_FUSION_NAMES
        for classifier in FIXED_CLASSIFIER_NAMES
        for dataset in dataset_names
    ])
    inventory["Status"] = np.where(
        inventory["Completed_Seeds"].eq(expected_seeds), "complete",
        np.where(inventory["Completed_Seeds"].eq(0), "not_started", "incomplete"),
    )
    return inventory


def summarize_saved_fusion_classifier_benchmark(output_root, seeds=(5, 6, 7, 37, 42)):
    """Rebuild tables from partial or complete saved rows without training."""
    output_root = Path(output_root)
    test_path = output_root / "seed_level_test_metrics.csv"
    val_path = output_root / "seed_level_validation_metrics.csv"
    if not test_path.is_file():
        raise FileNotFoundError(f"No saved benchmark test rows found: {test_path}")
    test_frame = pd.read_csv(test_path)
    val_frame = pd.read_csv(val_path) if val_path.is_file() else pd.DataFrame()
    summary_root = output_root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    val_summary = _summary(val_frame, ["Target_Config", "Fusion", "Classifier"], validation=True)
    test_summary = _summary(test_frame, ["Target_Config", "Fusion", "Classifier", "Dataset"])
    profile_columns = [c for c in (*PROFILE_NAMES, *PROFILE_EXTRA_NAMES) if c in test_frame.columns]
    profile_frame = test_frame[[c for c in ["Target_Config", "Fusion", "Classifier", "Seed", *profile_columns] if c in test_frame.columns]]
    if not profile_frame.empty:
        profile_frame = profile_frame.drop_duplicates(["Target_Config", "Fusion", "Classifier", "Seed"])
    profile_summary = _summary(profile_frame, ["Target_Config", "Fusion", "Classifier"])
    inventory = _build_inventory(test_frame, seeds)
    save_progress_csv(val_summary.to_dict("records"), summary_root / "validation_mean_std_by_fusion_classifier.csv")
    save_progress_csv(test_summary.to_dict("records"), summary_root / "test_mean_std_by_fusion_classifier_dataset.csv")
    save_progress_csv(profile_frame.to_dict("records"), summary_root / "profiling_by_seed.csv")
    save_progress_csv(profile_summary.to_dict("records"), summary_root / "profiling_mean_std_by_fusion_classifier.csv")
    save_progress_csv(inventory.to_dict("records"), summary_root / "completion_inventory.csv")
    return {
        "validation_metrics": val_frame,
        "test_metrics": test_frame,
        "validation_mean_std": val_summary,
        "test_mean_std": test_summary,
        "profiling_by_seed": profile_frame,
        "profiling_mean_std": profile_summary,
        "completion_inventory": inventory,
    }
