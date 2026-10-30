from .resume_utils import ensure_stage_signature
from .optuna_search import optuna_trial_plan, prune_trial, tell_trial
from .global_results import refresh_global_metrics_from_context
"""Sequential ML experiments for the fixed A/B model configurations."""
from copy import deepcopy
import pandas as pd
from pathlib import Path

CATBOOST_MAX_DEPTH = 16

DEFAULT_ML_SEARCH_CONFIG = {
    "n_trials": {"xgb_no_pca": 150, "lightgbm": 150, "catboost": 150},
    "xgb_no_pca_space": {
        # Suggested ranges; edit low/high here before running if domain knowledge suggests it.
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
    "xgb_no_pca_common": {
        "n_estimators": 5000, "early_stopping_rounds": 200,
        "tree_method": "hist",
        "device":"cuda"
    },
    "lightgbm_space": {
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
    "catboost_space": {
        "depth": {"type": "int", "low": 3, "high": CATBOOST_MAX_DEPTH},
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

def get_default_ml_search_config():
    """Return an independent, notebook-editable copy of ML search defaults."""
    return deepcopy(DEFAULT_ML_SEARCH_CONFIG)

def _persist_rows(rows, path):
    """Atomically checkpoint completed experiment rows for interrupted resumes."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)

def _metrics_from_trial_row(row):
    return {key: row[f"Val_{key}"] for key in (
        "Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "MCC", "AUROC", "AUPRC"
    ) if f"Val_{key}" in row}


def run_xgb_no_pca(context, search_config=None, display_fn=None):
    """Run or reuse the XGBoost no-PCA Optuna stage and return its result tables."""
    pipeline = context["pipeline"]
    TARGET_CONFIGS = context["TARGET_CONFIGS"]
    datasets = context["datasets"]
    DNA_MODEL = context["DNA_MODEL"]
    PROTEIN_MODEL = context["PROTEIN_MODEL"]
    CONFIG = context["CONFIG"]
    display = display_fn or print
    import json
    import time
    import pandas as pd
    from pathlib import Path

    import numpy as np
    import torch

    from core.module05v2_fusion_classifier.evaluator_profiler import FusionEvaluatorProfiler
    from core.module05v2_fusion_classifier.xgboost_model import XGBoostFusionManager

    # The fixed control below keeps
    # the original PCA baseline's XGBoost hyperparameters for an apples-to-apples check.
    XGB_NO_PCA_SPACE = search_config.get("xgb_no_pca_space", DEFAULT_ML_SEARCH_CONFIG["xgb_no_pca_space"]) if search_config is not None else DEFAULT_ML_SEARCH_CONFIG["xgb_no_pca_space"]
    XGB_N_TRIALS = int(search_config.get("n_trials", DEFAULT_ML_SEARCH_CONFIG["n_trials"]).get("xgb_no_pca", 100)) if search_config is not None else DEFAULT_ML_SEARCH_CONFIG["n_trials"]["xgb_no_pca"]
    XGB_FIXED_PARAMS = search_config.get("xgb_no_pca_common", DEFAULT_ML_SEARCH_CONFIG["xgb_no_pca_common"]) if search_config is not None else DEFAULT_ML_SEARCH_CONFIG["xgb_no_pca_common"]
    ML_OUTPUT_DIR = Path(pipeline.batch_run_dir)
    XGB_NO_PCA_DIR = ML_OUTPUT_DIR / "xgb_no_pca"
    XGB_NO_PCA_DIR.mkdir(parents=True, exist_ok=True)


    def collect_raw_fusion_matrix(split_name, target_config, is_train=False):
        """Concatenate uncompressed FM deltas and selected tabular features."""
        loader = pipeline._get_dataloader(
            split_name=split_name,
            pooling="center",
            dna_name=DNA_MODEL["name"],
            prot_name=PROTEIN_MODEL["name"],
            active_mods=target_config["active_modalities"],
            is_train=is_train,
            geom_feature_names=target_config["geom_feature_names"],
            geometry_context="train3",
            loader_seed=CONFIG["seed"],
        )
        feature_parts, labels, variant_ids = [], [], []
        with torch.no_grad():
            for batch in loader:
                parts = [batch["v_dna"], batch["v_prot"], batch["bio_features"]]
                if "geom_features" in batch:
                    parts.append(batch["geom_features"])
                feature_parts.append(torch.cat(parts, dim=-1).cpu().numpy().astype(np.float32, copy=False))
                labels.append(batch["label"].reshape(-1).cpu().numpy().astype(np.int32, copy=False))
                variant_ids.extend(batch["variant_id"])
        del loader
        return np.concatenate(feature_parts), np.concatenate(labels), variant_ids


    def fit_no_pca_xgb(params=None, original_control=False):
        manager = XGBoostFusionManager(
            random_state=CONFIG["seed"],
            n_jobs=CONFIG["xgb_n_jobs"],
        )
        if not original_control:
            manager.base_params.update(XGB_FIXED_PARAMS)
            manager.base_params.update(params or {})
        started = time.perf_counter()
        manager.train_hybrid(X_train, y_train, X_val, y_val)
        fit_seconds = time.perf_counter() - started
        val_prob, _, _ = manager.predict_hybrid(X_val)
        threshold, val_mcc_at_threshold = pipeline._select_validation_threshold(y_val, val_prob)
        val_pred = pipeline._threshold_predictions(val_prob, threshold)
        val_metrics = FusionEvaluatorProfiler.compute_metrics(y_val, val_prob, val_pred)
        return manager, threshold, val_mcc_at_threshold, val_metrics, fit_seconds


    # Reuse the completed stage outputs on subsequent Run All executions.
    XGB_NO_PCA_TRIALS_PATH = XGB_NO_PCA_DIR / "xgb_no_pca_grid_trials.csv"
    XGB_NO_PCA_VALIDATION_PATH = XGB_NO_PCA_DIR / "xgb_no_pca_validation_metrics.csv"
    XGB_NO_PCA_TEST_PATH = XGB_NO_PCA_DIR / "xgb_no_pca_test_metrics.csv"
    XGB_NO_PCA_COMPARISON_PATH = XGB_NO_PCA_DIR / "xgb_no_pca_vs_pca_baseline.csv"
    ensure_stage_signature(XGB_NO_PCA_DIR, "xgb_no_pca", {
        "sampler": "TPE", "search_space": XGB_NO_PCA_SPACE, "n_trials": XGB_N_TRIALS,
        "fixed_params": XGB_FIXED_PARAMS, "config": CONFIG,
        "targets": TARGET_CONFIGS, "datasets": datasets, "dna_model": DNA_MODEL, "protein_model": PROTEIN_MODEL,
    })
    XGB_NO_PCA_STAGE_COMPLETE = all(path.exists() for path in (
        XGB_NO_PCA_TRIALS_PATH, XGB_NO_PCA_VALIDATION_PATH,
        XGB_NO_PCA_TEST_PATH, XGB_NO_PCA_COMPARISON_PATH,
    ))

    if XGB_NO_PCA_STAGE_COMPLETE:
        print(f"[REUSE] Completed XGBoost no-PCA results: {XGB_NO_PCA_DIR}")
        trials_df = pd.read_csv(XGB_NO_PCA_TRIALS_PATH)
        validation_df = pd.read_csv(XGB_NO_PCA_VALIDATION_PATH)
        test_df = pd.read_csv(XGB_NO_PCA_TEST_PATH)
        comparison_df = pd.read_csv(XGB_NO_PCA_COMPARISON_PATH)
        display(validation_df)
        display(comparison_df.sort_values(["Target_Config", "Dataset", "Model"]))
    else:
        trial_rows = (pd.read_csv(XGB_NO_PCA_TRIALS_PATH).to_dict("records")
                      if XGB_NO_PCA_TRIALS_PATH.exists() else [])
        validation_rows, test_rows = [], []
        for target_config in TARGET_CONFIGS:
            config_name = target_config["name"]
            ablation = target_config["ablation"]
            print(f"\n[Pure XGBoost, no PCA] {config_name}")
            config_dir = XGB_NO_PCA_DIR / config_name
            config_dir.mkdir(parents=True, exist_ok=True)

            X_train, y_train, _ = collect_raw_fusion_matrix("train3", target_config, is_train=True)
            X_val, y_val, _ = collect_raw_fusion_matrix("val", target_config, is_train=False)
            print(f"[Features] train={X_train.shape} | validation={X_val.shape} | PCA=off")

            # No-PCA control is a resumable unit with its own model and metric record.
            control_row = next((row for row in trial_rows if row.get("Target_Config") == config_name
                                and row.get("Trial") == "no_pca_control_original_params"), None)
            control_model = XGBoostFusionManager(random_state=CONFIG["seed"], n_jobs=CONFIG["xgb_n_jobs"])
            if control_row is not None and (config_dir / "xgb_no_pca_control_xgboost.json").exists():
                control_model.load_model(str(config_dir), prefix="xgb_no_pca_control")
                control_val_prob, _, _ = control_model.predict_hybrid(X_val)
                control_threshold, control_val_mcc = float(control_row["Decision_Threshold"]), float(control_row["Val_MCC_at_Threshold"])
                control_val_metrics = FusionEvaluatorProfiler.compute_metrics(
                    y_val, control_val_prob, pipeline._threshold_predictions(control_val_prob, control_threshold)
                )
                control_seconds = float(control_row["Fit_Seconds"])
                print(f"[RESUME] {config_name}/XGBoost control")
            else:
                control_model, control_threshold, control_val_mcc, control_val_metrics, control_seconds = fit_no_pca_xgb(
                    original_control=True
                )
                control_model.save_model(str(config_dir), prefix="xgb_no_pca_control")
                control_params = dict(control_model.base_params)
                control_params["scale_pos_weight"] = control_model._calc_scale_pos_weight(y_train)
                control_row = {
                    "Target_Config": config_name, "Ablation": ablation,
                    "Trial": "no_pca_control_original_params", "PCA": False,
                    "Fit_Seconds": round(control_seconds, 3),
                    "Best_Iteration": int(getattr(control_model.model, "best_iteration", -1)),
                    "Decision_Threshold": float(control_threshold),
                    "Val_MCC_at_Threshold": float(control_val_mcc),
                    **{f"Val_{key}": value for key, value in control_val_metrics.items()},
                    "Parameters": json.dumps(control_params, sort_keys=True),
                }
                trial_rows = [row for row in trial_rows if not (row.get("Target_Config") == config_name
                               and row.get("Trial") == "no_pca_control_original_params")]
                trial_rows.append(control_row)
                _persist_rows(trial_rows, XGB_NO_PCA_TRIALS_PATH)
            control_params = json.loads(control_row["Parameters"])
            control_val_prob, _, _ = control_model.predict_hybrid(X_val)
            control_val_metrics = FusionEvaluatorProfiler.compute_metrics(
                y_val, control_val_prob, pipeline._threshold_predictions(control_val_prob, control_threshold)
            )
            validation_rows.append({
                "Target_Config": config_name,
                "Ablation": ablation,
                "Model": "XGBoost_NoPCA_Control",
                "Decision_Threshold": float(control_threshold),
                **control_val_metrics,
                "Fit_Seconds": round(control_seconds, 3),
                "Parameters": json.dumps(control_params, sort_keys=True),
            })
            control = control_model

            XGB_BEST_GRID_PREFIX = "xgb_no_pca_grid_best"
            best_grid_score = (-float("inf"), -float("inf"))
            best_grid_info = None
            optuna_trials = optuna_trial_plan(config_dir, f"xgb_no_pca_{config_name}", XGB_N_TRIALS, XGB_NO_PCA_SPACE, CONFIG["seed"])
            for trial_index, (study, opt_trial, params, is_new_trial) in enumerate(optuna_trials, start=1):
                trial_name = f"optuna_{opt_trial.number:04d}"
                trial_prefix = f"xgb_no_pca_{trial_name}"
                cached = next((row for row in trial_rows if row.get("Target_Config") == config_name
                               and row.get("Trial") == trial_name), None)
                manager = XGBoostFusionManager(random_state=CONFIG["seed"], n_jobs=CONFIG["xgb_n_jobs"])
                trial_model_exists = (config_dir / f"{trial_prefix}_xgboost.json").exists()
                if cached is not None and trial_model_exists:
                    manager.load_model(str(config_dir), prefix=trial_prefix)
                    val_prob, _, _ = manager.predict_hybrid(X_val)
                    threshold = float(cached["Decision_Threshold"])
                    val_metrics = FusionEvaluatorProfiler.compute_metrics(
                        y_val, val_prob, pipeline._threshold_predictions(val_prob, threshold)
                    )
                    val_mcc = float(cached["Val_MCC_at_Threshold"])
                    fit_seconds = float(cached["Fit_Seconds"])
                    print(f"[RESUME] {config_name}/XGBoost/{trial_name}")
                else:
                    manager, threshold, val_mcc, val_metrics, fit_seconds = fit_no_pca_xgb(params=params)
                    manager.save_model(str(config_dir), prefix=trial_prefix)
                score = (float(val_mcc), float(val_metrics["AUPRC"]))
                recorded_params = dict(manager.base_params)
                row = {
                    "Target_Config": config_name,
                    "Ablation": ablation,
                    "Trial": trial_name,
                    "PCA": False,
                    "Fit_Seconds": round(fit_seconds, 3),
                    "Best_Iteration": int(getattr(manager.model, "best_iteration", -1)),
                    "Decision_Threshold": float(threshold),
                    "Val_MCC_at_Threshold": float(val_mcc),
                    **{f"Val_{key}": value for key, value in val_metrics.items()},
                    "Parameters": json.dumps(recorded_params, sort_keys=True),
                }
                trial_rows = [r for r in trial_rows if not (r.get("Target_Config") == config_name
                               and r.get("Trial") == trial_name)]
                trial_rows.append(row)
                _persist_rows(trial_rows, XGB_NO_PCA_TRIALS_PATH)
                tell_trial(study, opt_trial, val_mcc) if is_new_trial else None
                if score > best_grid_score:
                    best_grid_score = score
                    best_grid_info = {
                        "threshold": threshold,
                        "val_mcc": val_mcc,
                        "val_metrics": val_metrics,
                        "fit_seconds": fit_seconds,
                        "params": recorded_params,
                        "trial": trial_name,
                    }
                    manager.save_model(str(config_dir), prefix=XGB_BEST_GRID_PREFIX)
                del manager
                print(
                    f"  [{trial_index:02d}/{XGB_N_TRIALS}] "
                    f"Val_MCC={val_mcc:.4f} | Val_AUPRC={val_metrics['AUPRC']:.4f} | {params}"
                )

            if best_grid_info is None:
                raise RuntimeError(f"No valid grid result for {config_name}")
            validation_rows.append({
                "Target_Config": config_name,
                "Ablation": ablation,
                "Model": "XGBoost_NoPCA_GridBest",
                "Decision_Threshold": float(best_grid_info["threshold"]),
                **best_grid_info["val_metrics"],
                "Fit_Seconds": round(best_grid_info["fit_seconds"], 3),
                "Parameters": json.dumps(best_grid_info["params"], sort_keys=True),
                "Selected_Trial": best_grid_info["trial"],
            })

            best_grid = XGBoostFusionManager(
                random_state=CONFIG["seed"],
                n_jobs=CONFIG["xgb_n_jobs"],
            )
            best_grid.load_model(str(config_dir), prefix=XGB_BEST_GRID_PREFIX)
            control = XGBoostFusionManager(
                random_state=CONFIG["seed"],
                n_jobs=CONFIG["xgb_n_jobs"],
            )
            control.load_model(str(config_dir), prefix="xgb_no_pca_control")

            for dataset_config in datasets:
                split_name = dataset_config["test"]
                X_test, y_test, variant_ids = collect_raw_fusion_matrix(
                    split_name, target_config, is_train=False
                )
                for model_name, manager, threshold, params, trial_name in [
                    ("XGBoost_NoPCA_Control", control, control_threshold, control_params, "control"),
                    ("XGBoost_NoPCA_GridBest", best_grid, best_grid_info["threshold"], best_grid_info["params"], best_grid_info["trial"]),
                ]:
                    probabilities, _, _ = manager.predict_hybrid(X_test)
                    predictions = pipeline._threshold_predictions(probabilities, threshold)
                    metrics = FusionEvaluatorProfiler.compute_metrics(y_test, probabilities, predictions)
                    test_rows.append({
                        "Target_Config": config_name,
                        "Ablation": ablation,
                        "Dataset": dataset_config["name"],
                        "Model": model_name,
                        "Selected_Trial": trial_name,
                        "Decision_Threshold": float(threshold),
                        **metrics,
                        "Parameters": json.dumps(params, sort_keys=True),
                    })
                    pred_dir = config_dir / "tests" / dataset_config["name"]
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({
                        "Variant_ID": variant_ids,
                        "True_Label": y_test,
                        "Predicted_Probability": probabilities,
                        "Decision_Threshold": threshold,
                        "Prediction_Class": predictions,
                    }).to_csv(pred_dir / f"{model_name}_predictions.csv", index=False)
                del X_test, y_test, variant_ids

            del X_train, y_train, X_val, y_val, best_grid, control

        # Persist complete trial, validation, and test tables after both configurations finish.
        trials_df = pd.DataFrame(trial_rows)
        validation_df = pd.DataFrame(validation_rows)
        test_df = pd.DataFrame(test_rows)
        trials_path = XGB_NO_PCA_DIR / "xgb_no_pca_grid_trials.csv"
        validation_path = XGB_NO_PCA_DIR / "xgb_no_pca_validation_metrics.csv"
        test_path = XGB_NO_PCA_DIR / "xgb_no_pca_test_metrics.csv"
        trials_df.to_csv(trials_path, index=False)
        validation_df.to_csv(validation_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Compare the fixed no-PCA control and tuned no-PCA model with the PCA baseline.
        pca_baseline = pd.read_csv(Path(pipeline.batch_run_dir) / "fixed_config_baseline.csv")
        pca_baseline = pca_baseline.loc[pca_baseline["Network"].eq("Pure_XGBoost_Concat")].copy()
        pca_baseline["Target_Config"] = pca_baseline["Ablation"].map({
            cfg["ablation"]: cfg["name"] for cfg in TARGET_CONFIGS
        })
        pca_baseline["Model"] = "Pure_XGBoost_PCA_Baseline"
        comparison_df = pd.concat([
            pca_baseline[["Target_Config", "Ablation", "Dataset", "Model", "Decision_Threshold", "MCC", "AUROC", "AUPRC"]],
            test_df[["Target_Config", "Ablation", "Dataset", "Model", "Decision_Threshold", "MCC", "AUROC", "AUPRC"]],
        ], ignore_index=True)
        comparison_path = XGB_NO_PCA_DIR / "xgb_no_pca_vs_pca_baseline.csv"
        comparison_df.sort_values(["Target_Config", "Dataset", "Model"]).to_csv(comparison_path, index=False)

        print(f"\n[COMPLETE] XGBoost without PCA")
        print(f"Trials: {trials_path}")
        print(f"Validation metrics: {validation_path}")
        print(f"Test metrics: {test_path}")
        print(f"PCA comparison: {comparison_path}")
        display(validation_df)
        display(comparison_df.sort_values(["Target_Config", "Dataset", "Model"]))
    global_metrics = refresh_global_metrics_from_context(context, display_fn=display, show=False)
    print(f"[GLOBAL RESULTS] Updated {global_metrics['path']}")
    return {"trials": trials_df, "validation": validation_df, "test": test_df, "comparison": comparison_df}


def run_lightgbm(context, search_config=None, display_fn=None):
    """Run or reuse the LightGBM Optuna stage and return its result tables."""
    pipeline = context["pipeline"]
    TARGET_CONFIGS = context["TARGET_CONFIGS"]
    datasets = context["datasets"]
    DNA_MODEL = context["DNA_MODEL"]
    PROTEIN_MODEL = context["PROTEIN_MODEL"]
    CONFIG = context["CONFIG"]
    display = display_fn or print
    import importlib.util
    import json
    import time
    from pathlib import Path

    import joblib
    import numpy as np
    import torch
    from sklearn.decomposition import PCA

    if importlib.util.find_spec("lightgbm") is None:
        raise ImportError("Ch?a c?i LightGBM trong kernel n?y. C?i b?ng `pip install lightgbm`, sau ?? ch?y l?i cell.")
    import lightgbm as lgb
    from core.module05v2_fusion_classifier.evaluator_profiler import FusionEvaluatorProfiler

    ML_OUTPUT_DIR = Path(pipeline.batch_run_dir)
    LIGHTGBM_DIR = ML_OUTPUT_DIR / "lightgbm"
    LIGHTGBM_DIR.mkdir(parents=True, exist_ok=True)
    LIGHTGBM_SPACE = search_config.get("lightgbm_space", DEFAULT_ML_SEARCH_CONFIG["lightgbm_space"]) if search_config is not None else DEFAULT_ML_SEARCH_CONFIG["lightgbm_space"]
    LIGHTGBM_N_TRIALS = int(search_config.get("n_trials", DEFAULT_ML_SEARCH_CONFIG["n_trials"]).get("lightgbm", 100)) if search_config is not None else DEFAULT_ML_SEARCH_CONFIG["n_trials"]["lightgbm"]
    LIGHTGBM_COMMON = {
        "n_estimators": 5000,
        "subsample_freq": 1,
        "objective": "binary",
        "verbosity": -1,
        "n_jobs": CONFIG["xgb_n_jobs"],
    }
    LIGHTGBM_OUTPUTS = [
        LIGHTGBM_DIR / "lightgbm_grid_trials.csv",
        LIGHTGBM_DIR / "lightgbm_validation_metrics.csv",
        LIGHTGBM_DIR / "lightgbm_test_metrics.csv",
        LIGHTGBM_DIR / "lightgbm_vs_xgboost.csv",
    ]

    ensure_stage_signature(LIGHTGBM_DIR, "lightgbm", {
        "sampler": "TPE", "search_space": LIGHTGBM_SPACE, "n_trials": LIGHTGBM_N_TRIALS,
        "common": LIGHTGBM_COMMON, "config": CONFIG,
        "targets": TARGET_CONFIGS, "datasets": datasets, "dna_model": DNA_MODEL, "protein_model": PROTEIN_MODEL,
    })
    if all(path.exists() for path in LIGHTGBM_OUTPUTS):
        print(f"[REUSE] Completed LightGBM results: {LIGHTGBM_DIR}")
        lgb_trials_df = pd.read_csv(LIGHTGBM_OUTPUTS[0])
        lgb_validation_df = pd.read_csv(LIGHTGBM_OUTPUTS[1])
        lgb_test_df = pd.read_csv(LIGHTGBM_OUTPUTS[2])
        lgb_comparison_df = pd.read_csv(LIGHTGBM_OUTPUTS[3])
        display(lgb_validation_df)
        display(lgb_comparison_df)
    else:
        def collect_lgb_inputs(split_name, target_config, is_train=False):
            loader = pipeline._get_dataloader(
                split_name=split_name,
                pooling="center",
                dna_name=DNA_MODEL["name"],
                prot_name=PROTEIN_MODEL["name"],
                active_mods=target_config["active_modalities"],
                is_train=is_train,
                geom_feature_names=target_config["geom_feature_names"],
                geometry_context="train3",
                loader_seed=CONFIG["seed"],
            )
            rows, labels, variant_ids = [], [], []
            dna_dim = prot_dim = None
            with torch.no_grad():
                for batch in loader:
                    dna = batch["v_dna"].cpu().numpy().astype(np.float32, copy=False)
                    prot = batch["v_prot"].cpu().numpy().astype(np.float32, copy=False)
                    bio = batch["bio_features"].cpu().numpy().astype(np.float32, copy=False)
                    parts = [dna, prot, bio]
                    if "geom_features" in batch:
                        parts.append(batch["geom_features"].cpu().numpy().astype(np.float32, copy=False))
                    rows.append(np.concatenate(parts, axis=1))
                    labels.append(batch["label"].reshape(-1).cpu().numpy().astype(np.int32, copy=False))
                    variant_ids.extend(batch["variant_id"])
                    dna_dim, prot_dim = dna.shape[1], prot.shape[1]
            del loader
            return np.concatenate(rows), np.concatenate(labels), variant_ids, dna_dim, prot_dim

        def make_lgb_features(X, dna_dim, prot_dim, pca_dna=None, pca_prot=None):
            dna = X[:, :dna_dim]
            prot = X[:, dna_dim:dna_dim + prot_dim]
            tabular = X[:, dna_dim + prot_dim:]
            if pca_dna is not None:
                dna = pca_dna.transform(dna)
                prot = pca_prot.transform(prot)
            return np.concatenate([dna, prot, tabular], axis=1).astype(np.float32, copy=False)

        def fit_lgb_trial(Xtr, ytr, Xvl, yvl, params):
            model_params = dict(params)
            scale_pos_weight = float(model_params.pop("scale_pos_weight"))
            model = lgb.LGBMClassifier(
                **LIGHTGBM_COMMON,
                **model_params,
                random_state=CONFIG["seed"],
                scale_pos_weight=scale_pos_weight,
            )
            started = time.perf_counter()
            model.fit(
                Xtr, ytr,
                eval_X=Xvl,
                eval_y=yvl,
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            )
            elapsed = time.perf_counter() - started
            probs = model.predict_proba(Xvl, num_iteration=model.best_iteration_)[:, 1]
            threshold, val_mcc = pipeline._select_validation_threshold(yvl, probs)
            preds = pipeline._threshold_predictions(probs, threshold)
            metrics = FusionEvaluatorProfiler.compute_metrics(yvl, probs, preds)
            return model, threshold, val_mcc, metrics, elapsed

        lgb_trial_rows = (pd.read_csv(LIGHTGBM_OUTPUTS[0]).to_dict("records")
                          if LIGHTGBM_OUTPUTS[0].exists() else [])
        lgb_validation_rows, lgb_test_rows = [], []
        for target_config in TARGET_CONFIGS:
            config_name = target_config["name"]
            ablation = target_config["ablation"]
            print(f"\\n[LightGBM] {config_name}")
            config_dir = LIGHTGBM_DIR / config_name
            config_dir.mkdir(parents=True, exist_ok=True)
            X_train_raw, y_train, _, dna_dim, prot_dim = collect_lgb_inputs("train3", target_config, is_train=True)
            X_val_raw, y_val, _, _, _ = collect_lgb_inputs("val", target_config, is_train=False)
            X_test_raw = {}
            y_test_by_set = {}
            ids_by_set = {}
            for dataset_config in datasets:
                split_name = dataset_config["test"]
                X_test_raw[dataset_config["name"]], y_test_by_set[dataset_config["name"]], ids_by_set[dataset_config["name"]], _, _ = collect_lgb_inputs(split_name, target_config, is_train=False)

            for use_pca in (True, False):
                branch = "PCA" if use_pca else "NoPCA"
                branch_dir = config_dir / branch.lower()
                branch_dir.mkdir(parents=True, exist_ok=True)
                pca_dna = pca_prot = None
                if use_pca:
                    pca_dna = PCA(n_components=min(256, X_train_raw.shape[0], dna_dim), svd_solver="randomized", random_state=CONFIG["seed"])
                    pca_prot = PCA(n_components=min(256, X_train_raw.shape[0], prot_dim), svd_solver="randomized", random_state=CONFIG["seed"])
                    pca_dna.fit(X_train_raw[:, :dna_dim])
                    pca_prot.fit(X_train_raw[:, dna_dim:dna_dim + prot_dim])
                    joblib.dump(pca_dna, branch_dir / "pca_dna.pkl")
                    joblib.dump(pca_prot, branch_dir / "pca_prot.pkl")
                X_train = make_lgb_features(X_train_raw, dna_dim, prot_dim, pca_dna, pca_prot)
                X_val = make_lgb_features(X_val_raw, dna_dim, prot_dim, pca_dna, pca_prot)
                print(f"  {branch}: train={X_train.shape}, val={X_val.shape}")

                best_score = (-float("inf"), -float("inf"))
                best_model_info = None
                optuna_trials = optuna_trial_plan(branch_dir, f"lightgbm_{config_name}_{branch}", LIGHTGBM_N_TRIALS, LIGHTGBM_SPACE, CONFIG["seed"])
                for trial_index, (study, opt_trial, params, is_new_trial) in enumerate(optuna_trials, start=1):
                    trial_name = f"optuna_{opt_trial.number:04d}"
                    trial_model_path = branch_dir / f"lightgbm_{trial_name}.txt"
                    cached = next((row for row in lgb_trial_rows
                                   if row.get("Target_Config") == config_name
                                   and row.get("Branch") == branch
                                   and row.get("Trial") == trial_name), None)
                    if cached is not None and trial_model_path.exists():
                        val_metrics = _metrics_from_trial_row(cached)
                        val_mcc = float(cached["Val_MCC_at_Threshold"])
                        score = (val_mcc, float(val_metrics["AUPRC"]))
                        if score > best_score:
                            best_score = score
                            best_model_info = {
                                "threshold": float(cached["Decision_Threshold"]), "metrics": val_metrics,
                                "val_mcc": val_mcc,
                                "params": dict(params),
                                "trial": trial_name, "fit_seconds": float(cached["Fit_Seconds"]),
                                "best_iteration": int(cached["Best_Iteration"]),
                            }
                            import shutil
                            shutil.copy2(trial_model_path, branch_dir / "lightgbm_best.txt")
                        print(f"    [RESUME] {config_name}/{branch}/{trial_name}")
                        tell_trial(study, opt_trial, val_mcc) if is_new_trial else None
                        continue
                    model, threshold, val_mcc, val_metrics, fit_seconds = fit_lgb_trial(X_train, y_train, X_val, y_val, params)
                    score = (float(val_mcc), float(val_metrics["AUPRC"]))
                    row = {
                        "Target_Config": config_name, "Ablation": ablation,
                        "Branch": branch, "Trial": trial_name,
                        "PCA": use_pca, "Fit_Seconds": round(fit_seconds, 3),
                        "Best_Iteration": int(model.best_iteration_ or -1),
                        "Decision_Threshold": float(threshold),
                        "Val_MCC_at_Threshold": float(val_mcc),
                        **{f"Val_{key}": value for key, value in val_metrics.items()},
                        "Parameters": json.dumps({**LIGHTGBM_COMMON, **params}, sort_keys=True),
                    }
                    model.booster_.save_model(str(trial_model_path))
                    lgb_trial_rows = [r for r in lgb_trial_rows if not (
                        r.get("Target_Config") == config_name and r.get("Branch") == branch
                        and r.get("Trial") == trial_name
                    )]
                    lgb_trial_rows.append(row)
                    _persist_rows(lgb_trial_rows, LIGHTGBM_OUTPUTS[0])
                    tell_trial(study, opt_trial, val_mcc) if is_new_trial else None
                    if score > best_score:
                        best_score = score
                        best_model_info = {
                            "threshold": float(threshold), "metrics": val_metrics,
                            "val_mcc": float(val_mcc), "params": dict(params),
                            "trial": row["Trial"], "fit_seconds": fit_seconds,
                            "best_iteration": int(model.best_iteration_ or -1),
                        }
                        model.booster_.save_model(str(branch_dir / "lightgbm_best.txt"))
                    del model
                    print(f"    [{trial_index:02d}/{LIGHTGBM_N_TRIALS}] Val_MCC={val_mcc:.4f} | Val_AUPRC={val_metrics['AUPRC']:.4f} | {params}")

                if best_model_info is None:
                    raise RuntimeError(f"No LightGBM Optuna result for {config_name}/{branch}")
                lgb_validation_rows.append({
                    "Target_Config": config_name, "Ablation": ablation,
                    "Model": f"LightGBM_{branch}_GridBest",
                    "Decision_Threshold": best_model_info["threshold"],
                    **best_model_info["metrics"],
                    "Val_MCC_at_Threshold": best_model_info["val_mcc"],
                    "Fit_Seconds": round(best_model_info["fit_seconds"], 3),
                    "Best_Iteration": best_model_info["best_iteration"],
                    "Selected_Trial": best_model_info["trial"],
                    "Parameters": json.dumps({**LIGHTGBM_COMMON, **best_model_info["params"]}, sort_keys=True),
                })
                best_model = lgb.Booster(model_file=str(branch_dir / "lightgbm_best.txt"))
                for dataset_config in datasets:
                    test_name = dataset_config["name"]
                    X_test = make_lgb_features(X_test_raw[test_name], dna_dim, prot_dim, pca_dna, pca_prot)
                    probabilities = best_model.predict(X_test, num_iteration=best_model_info["best_iteration"] if best_model_info["best_iteration"] > 0 else None)
                    threshold = best_model_info["threshold"]
                    predictions = pipeline._threshold_predictions(probabilities, threshold)
                    metrics = FusionEvaluatorProfiler.compute_metrics(y_test_by_set[test_name], probabilities, predictions)
                    lgb_test_rows.append({
                        "Target_Config": config_name, "Ablation": ablation,
                        "Dataset": test_name, "Model": f"LightGBM_{branch}_GridBest",
                        "Selected_Trial": best_model_info["trial"],
                        "Decision_Threshold": threshold, **metrics,
                        "Parameters": json.dumps({**LIGHTGBM_COMMON, **best_model_info["params"]}, sort_keys=True),
                    })
                    pred_dir = branch_dir / "tests" / test_name
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({
                        "Variant_ID": ids_by_set[test_name],
                        "True_Label": y_test_by_set[test_name],
                        "Predicted_Probability": probabilities,
                        "Decision_Threshold": threshold,
                        "Prediction_Class": predictions,
                    }).to_csv(pred_dir / "predictions.csv", index=False)
                del best_model, X_train, X_val
            del X_train_raw, X_val_raw, X_test_raw, y_test_by_set, ids_by_set

        lgb_trials_df = pd.DataFrame(lgb_trial_rows)
        lgb_validation_df = pd.DataFrame(lgb_validation_rows)
        lgb_test_df = pd.DataFrame(lgb_test_rows)
        lgb_trials_df.to_csv(LIGHTGBM_OUTPUTS[0], index=False)
        lgb_validation_df.to_csv(LIGHTGBM_OUTPUTS[1], index=False)
        lgb_test_df.to_csv(LIGHTGBM_OUTPUTS[2], index=False)
        xgb_no_pca = pd.read_csv(ML_OUTPUT_DIR / "xgb_no_pca" / "xgb_no_pca_test_metrics.csv")
        xgb_pca = pd.read_csv(Path(pipeline.batch_run_dir) / "fixed_config_baseline.csv")
        xgb_pca = xgb_pca.loc[xgb_pca["Network"].eq("Pure_XGBoost_Concat")].copy()
        xgb_pca["Target_Config"] = xgb_pca["Ablation"].map({cfg["ablation"]: cfg["name"] for cfg in TARGET_CONFIGS})
        xgb_pca["Model"] = "Pure_XGBoost_PCA_Baseline"
        compare_cols = ["Target_Config", "Ablation", "Dataset", "Model", "Decision_Threshold", "MCC", "AUROC", "AUPRC"]
        lgb_comparison_df = pd.concat([
            xgb_pca[compare_cols],
            xgb_no_pca[compare_cols],
            lgb_test_df[compare_cols],
        ], ignore_index=True).sort_values(["Target_Config", "Dataset", "Model"])
        lgb_comparison_df.to_csv(LIGHTGBM_OUTPUTS[3], index=False)
        print(f"\\n[COMPLETE] LightGBM results saved in {LIGHTGBM_DIR}")
        display(lgb_validation_df)
        display(lgb_comparison_df)
    global_metrics = refresh_global_metrics_from_context(context, display_fn=display, show=False)
    print(f"[GLOBAL RESULTS] Updated {global_metrics['path']}")
    return {"trials": lgb_trials_df, "validation": lgb_validation_df, "test": lgb_test_df, "comparison": lgb_comparison_df}


def run_catboost(context, search_config=None, display_fn=None):
    """Run or reuse the CatBoost Optuna stage and return its result tables."""
    pipeline = context["pipeline"]
    TARGET_CONFIGS = context["TARGET_CONFIGS"]
    datasets = context["datasets"]
    DNA_MODEL = context["DNA_MODEL"]
    PROTEIN_MODEL = context["PROTEIN_MODEL"]
    CONFIG = context["CONFIG"]
    display = display_fn or print
    import importlib.util
    import json
    import os
    import time
    from pathlib import Path

    import joblib
    import numpy as np
    import torch
    from sklearn.decomposition import PCA

    if importlib.util.find_spec("catboost") is None:
        raise ImportError("Ch?a c?i CatBoost trong kernel n?y. C?i b?ng `pip install catboost`, sau ?? ch?y l?i cell.")
    from catboost import CatBoostClassifier
    from core.module05v2_fusion_classifier.evaluator_profiler import FusionEvaluatorProfiler

    ML_OUTPUT_DIR = Path(pipeline.batch_run_dir)
    CATBOOST_DIR = ML_OUTPUT_DIR / "catboost"
    CATBOOST_DIR.mkdir(parents=True, exist_ok=True)
    CATBOOST_SPACE = search_config.get("catboost_space", DEFAULT_ML_SEARCH_CONFIG["catboost_space"]) if search_config is not None else DEFAULT_ML_SEARCH_CONFIG["catboost_space"]
    CATBOOST_N_TRIALS = int(search_config.get("n_trials", DEFAULT_ML_SEARCH_CONFIG["n_trials"]).get("catboost", 100)) if search_config is not None else DEFAULT_ML_SEARCH_CONFIG["n_trials"]["catboost"]
    CATBOOST_COMMON = {
        "iterations": 5000,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": min(8, max(1, (os.cpu_count() or 1) - 1)),
    }
    CATBOOST_OUTPUTS = [
        CATBOOST_DIR / "catboost_grid_trials.csv",
        CATBOOST_DIR / "catboost_validation_metrics.csv",
        CATBOOST_DIR / "catboost_test_metrics.csv",
        CATBOOST_DIR / "catboost_vs_previous_models.csv",
    ]

    ensure_stage_signature(CATBOOST_DIR, "catboost", {
        "sampler": "TPE", "search_space": CATBOOST_SPACE, "n_trials": CATBOOST_N_TRIALS,
        "common": CATBOOST_COMMON, "config": CONFIG,
        "targets": TARGET_CONFIGS, "datasets": datasets, "dna_model": DNA_MODEL, "protein_model": PROTEIN_MODEL,
    })
    if all(path.exists() for path in CATBOOST_OUTPUTS):
        print(f"[REUSE] Completed CatBoost results: {CATBOOST_DIR}")
        cat_trials_df = pd.read_csv(CATBOOST_OUTPUTS[0])
        cat_validation_df = pd.read_csv(CATBOOST_OUTPUTS[1])
        cat_test_df = pd.read_csv(CATBOOST_OUTPUTS[2])
        cat_comparison_df = pd.read_csv(CATBOOST_OUTPUTS[3])
        display(cat_validation_df)
        display(cat_comparison_df)
    else:
        def collect_catboost_inputs(split_name, target_config, is_train=False):
            loader = pipeline._get_dataloader(
                split_name=split_name,
                pooling="center",
                dna_name=DNA_MODEL["name"],
                prot_name=PROTEIN_MODEL["name"],
                active_mods=target_config["active_modalities"],
                is_train=is_train,
                geom_feature_names=target_config["geom_feature_names"],
                geometry_context="train3",
                loader_seed=CONFIG["seed"],
            )
            rows, labels, variant_ids = [], [], []
            dna_dim = prot_dim = None
            with torch.no_grad():
                for batch in loader:
                    dna = batch["v_dna"].cpu().numpy().astype(np.float32, copy=False)
                    prot = batch["v_prot"].cpu().numpy().astype(np.float32, copy=False)
                    bio = batch["bio_features"].cpu().numpy().astype(np.float32, copy=False)
                    parts = [dna, prot, bio]
                    if "geom_features" in batch:
                        parts.append(batch["geom_features"].cpu().numpy().astype(np.float32, copy=False))
                    rows.append(np.concatenate(parts, axis=1))
                    labels.append(batch["label"].reshape(-1).cpu().numpy().astype(np.int32, copy=False))
                    variant_ids.extend(batch["variant_id"])
                    dna_dim, prot_dim = dna.shape[1], prot.shape[1]
            del loader
            return np.concatenate(rows), np.concatenate(labels), variant_ids, dna_dim, prot_dim

        def make_catboost_features(X, dna_dim, prot_dim, pca_dna=None, pca_prot=None):
            dna = X[:, :dna_dim]
            prot = X[:, dna_dim:dna_dim + prot_dim]
            tabular = X[:, dna_dim + prot_dim:]
            if pca_dna is not None:
                dna = pca_dna.transform(dna)
                prot = pca_prot.transform(prot)
            return np.concatenate([dna, prot, tabular], axis=1).astype(np.float32, copy=False)

        cat_trial_rows = (pd.read_csv(CATBOOST_OUTPUTS[0]).to_dict("records")
                          if CATBOOST_OUTPUTS[0].exists() else [])
        cat_validation_rows, cat_test_rows = [], []
        for target_config in TARGET_CONFIGS:
            config_name = target_config["name"]
            ablation = target_config["ablation"]
            print(f"\\n[CatBoost] {config_name}")
            config_dir = CATBOOST_DIR / config_name
            config_dir.mkdir(parents=True, exist_ok=True)
            X_train_raw, y_train, _, dna_dim, prot_dim = collect_catboost_inputs("train3", target_config, is_train=True)
            X_val_raw, y_val, _, _, _ = collect_catboost_inputs("val", target_config, is_train=False)

            for use_pca in (True, False):
                branch = "PCA" if use_pca else "NoPCA"
                branch_dir = config_dir / branch.lower()
                branch_dir.mkdir(parents=True, exist_ok=True)
                pca_dna = pca_prot = None
                if use_pca:
                    pca_dna = PCA(n_components=min(256, X_train_raw.shape[0], dna_dim), svd_solver="randomized", random_state=CONFIG["seed"])
                    pca_prot = PCA(n_components=min(256, X_train_raw.shape[0], prot_dim), svd_solver="randomized", random_state=CONFIG["seed"])
                    pca_dna.fit(X_train_raw[:, :dna_dim])
                    pca_prot.fit(X_train_raw[:, dna_dim:dna_dim + prot_dim])
                    joblib.dump(pca_dna, branch_dir / "pca_dna.pkl")
                    joblib.dump(pca_prot, branch_dir / "pca_prot.pkl")
                X_train = make_catboost_features(X_train_raw, dna_dim, prot_dim, pca_dna, pca_prot)
                X_val = make_catboost_features(X_val_raw, dna_dim, prot_dim, pca_dna, pca_prot)
                print(f"  {branch}: train={X_train.shape}, val={X_val.shape}")

                best_score = (-float("inf"), -float("inf"))
                best_info = None
                optuna_trials = optuna_trial_plan(branch_dir, f"catboost_{config_name}_{branch}", CATBOOST_N_TRIALS, CATBOOST_SPACE, CONFIG["seed"])
                for trial_index, (study, opt_trial, params, is_new_trial) in enumerate(optuna_trials, start=1):
                    trial_name = f"optuna_{opt_trial.number:04d}"
                    trial_model_path = branch_dir / f"catboost_{trial_name}.cbm"
                    cached = next((row for row in cat_trial_rows
                                   if row.get("Target_Config") == config_name
                                   and row.get("Branch") == branch
                                   and row.get("Trial") == trial_name), None)
                    if cached is not None and trial_model_path.exists():
                        val_metrics = _metrics_from_trial_row(cached)
                        val_mcc = float(cached["Val_MCC_at_Threshold"])
                        score = (val_mcc, float(val_metrics["AUPRC"]))
                        if score > best_score:
                            best_score = score
                            best_info = {
                                "threshold": float(cached["Decision_Threshold"]), "metrics": val_metrics,
                                "val_mcc": val_mcc,
                                "params": dict(params),
                                "trial": trial_name, "fit_seconds": float(cached["Fit_Seconds"]),
                                "best_iteration": int(cached["Best_Iteration"]),
                            }
                            import shutil
                            shutil.copy2(trial_model_path, branch_dir / "catboost_best.cbm")
                        print(f"    [RESUME] {config_name}/{branch}/{trial_name}")
                        tell_trial(study, opt_trial, val_mcc) if is_new_trial else None
                        continue
                    depth = int(params.get("depth", 6))
                    if depth > CATBOOST_MAX_DEPTH:
                        if is_new_trial:
                            prune_trial(study, opt_trial)
                        print(
                            f"    [SKIP] {config_name}/{branch}/{trial_name}: "
                            f"CatBoost depth {depth} exceeds supported maximum {CATBOOST_MAX_DEPTH}"
                        )
                        continue
                    cat_params = dict(params)
                    scale_pos_weight = float(cat_params.pop("scale_pos_weight"))
                    model = CatBoostClassifier(
                        **CATBOOST_COMMON,
                        **cat_params,
                        random_seed=CONFIG["seed"],
                        scale_pos_weight=scale_pos_weight,
                    )
                    started = time.perf_counter()
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=120,
                        use_best_model=True,
                        verbose=False,
                    )
                    fit_seconds = time.perf_counter() - started
                    val_prob = model.predict_proba(X_val)[:, 1]
                    threshold, val_mcc = pipeline._select_validation_threshold(y_val, val_prob)
                    val_pred = pipeline._threshold_predictions(val_prob, threshold)
                    val_metrics = FusionEvaluatorProfiler.compute_metrics(y_val, val_prob, val_pred)
                    score = (float(val_mcc), float(val_metrics["AUPRC"]))
                    row = {
                        "Target_Config": config_name, "Ablation": ablation,
                        "Branch": branch, "Trial": trial_name,
                        "PCA": use_pca, "Fit_Seconds": round(fit_seconds, 3),
                        "Best_Iteration": int(model.get_best_iteration()),
                        "Decision_Threshold": float(threshold),
                        "Val_MCC_at_Threshold": float(val_mcc),
                        **{f"Val_{key}": value for key, value in val_metrics.items()},
                        "Parameters": json.dumps({**CATBOOST_COMMON, **params, "random_seed": CONFIG["seed"]}, sort_keys=True),
                    }
                    model.save_model(str(trial_model_path))
                    cat_trial_rows = [r for r in cat_trial_rows if not (
                        r.get("Target_Config") == config_name and r.get("Branch") == branch
                        and r.get("Trial") == trial_name
                    )]
                    cat_trial_rows.append(row)
                    _persist_rows(cat_trial_rows, CATBOOST_OUTPUTS[0])
                    tell_trial(study, opt_trial, val_mcc) if is_new_trial else None
                    if score > best_score:
                        best_score = score
                        best_info = {
                            "threshold": float(threshold), "metrics": val_metrics,
                            "val_mcc": float(val_mcc), "params": dict(params),
                            "trial": row["Trial"], "fit_seconds": fit_seconds,
                            "best_iteration": int(model.get_best_iteration()),
                        }
                        model.save_model(str(branch_dir / "catboost_best.cbm"))
                    del model
                    print(f"    [{trial_index:02d}/{CATBOOST_N_TRIALS}] Val_MCC={val_mcc:.4f} | Val_AUPRC={val_metrics['AUPRC']:.4f} | {params}")

                if best_info is None:
                    raise RuntimeError(f"No CatBoost Optuna result for {config_name}/{branch}")
                cat_validation_rows.append({
                    "Target_Config": config_name, "Ablation": ablation,
                    "Model": f"CatBoost_{branch}_GridBest",
                    "Decision_Threshold": best_info["threshold"],
                    **best_info["metrics"],
                    "Val_MCC_at_Threshold": best_info["val_mcc"],
                    "Fit_Seconds": round(best_info["fit_seconds"], 3),
                    "Best_Iteration": best_info["best_iteration"],
                    "Selected_Trial": best_info["trial"],
                    "Parameters": json.dumps({**CATBOOST_COMMON, **best_info["params"]}, sort_keys=True),
                })
                best_model = CatBoostClassifier()
                best_model.load_model(str(branch_dir / "catboost_best.cbm"))
                for dataset_config in datasets:
                    test_name = dataset_config["name"]
                    X_test_raw, y_test, variant_ids, _, _ = collect_catboost_inputs(dataset_config["test"], target_config, is_train=False)
                    X_test = make_catboost_features(X_test_raw, dna_dim, prot_dim, pca_dna, pca_prot)
                    probabilities = best_model.predict_proba(X_test)[:, 1]
                    predictions = pipeline._threshold_predictions(probabilities, best_info["threshold"])
                    metrics = FusionEvaluatorProfiler.compute_metrics(y_test, probabilities, predictions)
                    cat_test_rows.append({
                        "Target_Config": config_name, "Ablation": ablation,
                        "Dataset": test_name, "Model": f"CatBoost_{branch}_GridBest",
                        "Selected_Trial": best_info["trial"],
                        "Decision_Threshold": best_info["threshold"], **metrics,
                        "Parameters": json.dumps({**CATBOOST_COMMON, **best_info["params"]}, sort_keys=True),
                    })
                    pred_dir = branch_dir / "tests" / test_name
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({
                        "Variant_ID": variant_ids,
                        "True_Label": y_test,
                        "Predicted_Probability": probabilities,
                        "Decision_Threshold": best_info["threshold"],
                        "Prediction_Class": predictions,
                    }).to_csv(pred_dir / "predictions.csv", index=False)
                    del X_test_raw, X_test, y_test, variant_ids
                del best_model, X_train, X_val
            del X_train_raw, X_val_raw

        cat_trials_df = pd.DataFrame(cat_trial_rows)
        cat_validation_df = pd.DataFrame(cat_validation_rows)
        cat_test_df = pd.DataFrame(cat_test_rows)
        cat_trials_df.to_csv(CATBOOST_OUTPUTS[0], index=False)
        cat_validation_df.to_csv(CATBOOST_OUTPUTS[1], index=False)
        cat_test_df.to_csv(CATBOOST_OUTPUTS[2], index=False)

        # Compare against the already completed fixed-config XGBoost and LightGBM stages.
        xgb_no_pca = pd.read_csv(ML_OUTPUT_DIR / "xgb_no_pca" / "xgb_no_pca_test_metrics.csv")
        xgb_pca = pd.read_csv(Path(pipeline.batch_run_dir) / "fixed_config_baseline.csv")
        xgb_pca = xgb_pca.loc[xgb_pca["Network"].eq("Pure_XGBoost_Concat")].copy()
        xgb_pca["Target_Config"] = xgb_pca["Ablation"].map({cfg["ablation"]: cfg["name"] for cfg in TARGET_CONFIGS})
        xgb_pca["Model"] = "Pure_XGBoost_PCA_Baseline"
        lgb_previous = pd.read_csv(ML_OUTPUT_DIR / "lightgbm" / "lightgbm_test_metrics.csv")
        compare_cols = ["Target_Config", "Ablation", "Dataset", "Model", "Decision_Threshold", "MCC", "AUROC", "AUPRC"]
        cat_comparison_df = pd.concat([
            xgb_pca[compare_cols],
            xgb_no_pca[compare_cols],
            lgb_previous[compare_cols],
            cat_test_df[compare_cols],
        ], ignore_index=True).sort_values(["Target_Config", "Dataset", "Model"])
        cat_comparison_df.to_csv(CATBOOST_OUTPUTS[3], index=False)
        print(f"\\n[COMPLETE] CatBoost results saved in {CATBOOST_DIR}")
        display(cat_validation_df)
        display(cat_comparison_df)
    global_metrics = refresh_global_metrics_from_context(context, display_fn=display, show=False)
    print(f"[GLOBAL RESULTS] Updated {global_metrics['path']}")
    return {"trials": cat_trials_df, "validation": cat_validation_df, "test": cat_test_df, "comparison": cat_comparison_df}
