"""Collect and rank completed pure-DL experiment results."""
from pathlib import Path
import json
import pandas as pd
from .global_results import refresh_global_metrics_from_context

def collect_dl_results(context, search_config, display_fn=None):
    pipeline = context["pipeline"]
    SEED_CONFIRM_DIR = Path(pipeline.batch_run_dir) / "pure_dl_seed_confirmation_v1"
    BASELINE_BATCH_RUN_DIR = context.get("BASELINE_BATCH_RUN_DIR")
    CONFIG = context["CONFIG"]
    DEVICE = context["DEVICE"]
    DNA_MODEL = context["DNA_MODEL"]
    PROTEIN_MODEL = context["PROTEIN_MODEL"]
    TARGET_CONFIGS = context["TARGET_CONFIGS"]
    datasets = context["datasets"]
    DL_SEARCH_CONFIG = search_config
    # These constants are defined locally so the combined cell does not depend on variables
    # that existed only in the former separate summary cell.
    TEST_DATASETS = [
        "Train3Val_Test", "Train3Val_ClinVarHQ",
        "Train3Val_ProteinGym", "Train3Val_UniProt",
    ]
    FINAL_DL_DIR = Path(pipeline.batch_run_dir) / "pure_dl_final_v1"
    FINAL_DL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Combine all saved search stages and repeated-seed confirmation metrics.
    # Individual seed rows stay in the full table;
    # ranking uses one mean row per 24-seed candidate to avoid favoring lucky seeds.
    DL_RESULT_SOURCES = [
        ("pure_dl_regularization_grid/pure_dl_test_metrics.csv", "regularization_optuna_v1", None),
        ("pure_dl_regularization_grid_v2/pure_dl_test_metrics.csv", "regularization_optuna_v2", None),
        ("pure_dl_architecture_search_v1/architecture_variant_test_metrics.csv", "architecture_search_v1", "pure_dl_architecture_search_v1/architecture_variant_validation_metrics.csv"),
        ("pure_dl_architecture_variants_v2/architecture_variant_test_metrics.csv", "architecture_variants_v2", "pure_dl_architecture_variants_v2/architecture_variant_validation_metrics.csv"),
        ("pure_dl_architecture_variants_v3/architecture_variant_test_metrics.csv", "architecture_variants_v3", "pure_dl_architecture_variants_v3/architecture_variant_validation_metrics.csv"),
        ("pure_dl_architecture_hparam_search_v1/architecture_variant_test_metrics.csv", "architecture_hparam_search_v1", "pure_dl_architecture_hparam_search_v1/architecture_variant_validation_metrics.csv"),
        ("pure_dl_architecture_hparam_search_v1/training_hparam_optuna/training_optuna_best_test.csv", "training_hparam_optuna_best", "pure_dl_architecture_hparam_search_v1/training_hparam_optuna/training_optuna_best_validation.csv"),
        ("pure_dl_seed_confirmation_v1/seed_confirmation_test_metrics.csv", "seed_confirmation_individual_seeds", "pure_dl_seed_confirmation_v1/seed_confirmation_trials.csv"),
    ]
    
    _RESULT_COLUMNS = [
        "Source", "Candidate_Level", "Target_Config", "Dataset", "Architecture", "Variant",
        "Profile", "Seed", "N_Seeds", "Decision_Threshold", "Val_MCC", "Val_AUROC", "Val_AUPRC",
        "Best_Epoch", "Fit_Seconds", "Resumed_From_Checkpoint", "Accuracy", "Precision", "Recall",
        "Specificity", "F1_Score", "MCC", "AUROC", "AUPRC", "MCC_SD", "AUROC_SD",
        "AUPRC_SD", "Hyperparameters",
    ]
    _HPARAM_COLUMNS = [
        "fusion_dim", "Fusion_Dim", "dropout", "Dropout", "weight_decay", "Weight_Decay",
        "learning_rate", "Learning_Rate", "head_scale", "feature_dropout", "modality_dropout",
        "gaussian_noise", "dna_feature_dropout", "protein_feature_dropout", "bio_feature_dropout",
        "geom_feature_dropout", "bio_gaussian_noise", "geom_gaussian_noise", "Loss_Type",
        "Batch_Size", "Max_Epochs", "max_epochs", "early_stop_patience", "optimizer",
        "grad_clip_norm", "label_smoothing", "pos_weight_multiplier", "focal_gamma", "focal_alpha",
        "concat_bilinear_rank", "cross_attention_dropout", "cross_attention_heads",
        "transformer_ffn_expansion", "transformer_num_heads", "transformer_depth",
        "gating_temperature", "Selected_Trial",
    ]
    
    def _dl_parameter_json(row):
        params = {}
        for col in _HPARAM_COLUMNS:
            if col in row.index and pd.notna(row[col]):
                params[col] = row[col].item() if hasattr(row[col], "item") else row[col]
        return json.dumps(params, ensure_ascii=False, sort_keys=True) if params else ""
    
    def _append_dl_test_rows(frame, source, candidate_level="single_run"):
        frame = frame.copy()
        if "Variant" not in frame.columns:
            frame["Variant"] = frame["Model"] if "Model" in frame.columns else frame["Architecture"]
        if "Profile" not in frame.columns:
            frame["Profile"] = frame["Selected_Trial"] if "Selected_Trial" in frame.columns else ""
        if "Seed" not in frame.columns:
            frame["Seed"] = ""
        rows = []
        for _, row in frame.iterrows():
            rows.append({
                "Source": source, "Candidate_Level": candidate_level,
                "Target_Config": row.get("Target_Config", ""), "Dataset": row.get("Dataset", ""),
                "Architecture": row.get("Architecture", ""), "Variant": row.get("Variant", ""),
                "Profile": row.get("Profile", ""), "Seed": row.get("Seed", ""),
                "N_Seeds": row.get("N_Seeds", 1),
                "Decision_Threshold": row.get("Decision_Threshold", float("nan")),
                "Val_MCC": row.get("Val_MCC", float("nan")),
                "Val_AUROC": row.get("Val_AUROC", float("nan")),
                "Val_AUPRC": row.get("Val_AUPRC", float("nan")),
                "Best_Epoch": row.get("Best_Epoch", float("nan")),
                "Fit_Seconds": row.get("Fit_Seconds", float("nan")),
                "Resumed_From_Checkpoint": row.get("Resumed_From_Checkpoint", float("nan")),
                "Accuracy": row.get("Accuracy", float("nan")), "Precision": row.get("Precision", float("nan")),
                "Recall": row.get("Recall", float("nan")), "Specificity": row.get("Specificity", float("nan")),
                "F1_Score": row.get("F1_Score", float("nan")),
                "MCC": row.get("MCC", float("nan")), "AUROC": row.get("AUROC", float("nan")),
                "AUPRC": row.get("AUPRC", float("nan")), "MCC_SD": float("nan"),
                "AUROC_SD": float("nan"), "AUPRC_SD": float("nan"),
                "Hyperparameters": _dl_parameter_json(row),
            })
        return rows
    
    all_dl_rows = []
    for relative_path, source_name, validation_relative_path in DL_RESULT_SOURCES:
        result_path = Path(pipeline.batch_run_dir) / relative_path
        if not result_path.exists():
            print(f"[SKIP] DL result file not found: {result_path}")
            continue
        result_frame = pd.read_csv(result_path)
        if validation_relative_path is not None:
            validation_path = Path(pipeline.batch_run_dir) / validation_relative_path
            if validation_path.exists():
                validation_frame = pd.read_csv(validation_path)
                join_cols = [col for col in ["Target_Config", "Architecture", "Variant", "Profile", "Seed"] if col in result_frame and col in validation_frame]
                evidence_cols = ["Val_MCC", "Val_AUROC", "Val_AUPRC", "Best_Epoch", "Fit_Seconds", "Resumed_From_Checkpoint"]
                extra_cols = list(dict.fromkeys(col for col in _HPARAM_COLUMNS + ["Profile"] + evidence_cols if col in validation_frame and col not in result_frame))
                if join_cols and extra_cols:
                    validation_lookup = validation_frame[join_cols + extra_cols].drop_duplicates(join_cols)
                    result_frame = result_frame.merge(validation_lookup, on=join_cols, how="left")
        level = "individual_seed" if source_name == "seed_confirmation_individual_seeds" else "single_run"
        all_dl_rows.extend(_append_dl_test_rows(result_frame, source_name, level))
    
    # Keep every validation-side trial, including configurations that were not carried to test.
    DL_TRIAL_SOURCES = [
        ("pure_dl_regularization_grid/pure_dl_grid_trials.csv", "regularization_optuna_v1_trials"),
        ("pure_dl_regularization_grid_v2/pure_dl_grid_trials.csv", "regularization_optuna_v2_trials"),
        ("pure_dl_architecture_search_v1/architecture_variant_trials.csv", "architecture_search_v1_trials"),
        ("pure_dl_architecture_variants_v2/architecture_variant_trials.csv", "architecture_variants_v2_trials"),
        ("pure_dl_architecture_variants_v3/architecture_variant_trials.csv", "architecture_variants_v3_trials"),
        ("pure_dl_architecture_hparam_search_v1/architecture_variant_trials.csv", "architecture_hparam_search_v1_trials"),
        ("pure_dl_architecture_hparam_search_v1/training_hparam_optuna/training_optuna_trials.csv", "training_hparam_optuna_trials"),
        ("pure_dl_seed_confirmation_v1/seed_confirmation_trials.csv", "seed_confirmation_trials"),
    ]
    all_dl_trial_frames = []
    baseline_validation_frames = []
    for relative_path, source_name in DL_TRIAL_SOURCES:
        trial_path = Path(pipeline.batch_run_dir) / relative_path
        if trial_path.exists():
            trial_frame = pd.read_csv(trial_path)
            trial_frame.insert(0, "Source", source_name)
            all_dl_trial_frames.append(trial_frame)
        else:
            print(f"[SKIP] DL trial file not found: {trial_path}")
    
    # Add fixed-config DL baselines (current run and the original batch baseline), filtered to A/B.
    BASELINE_ABLATIONS = {
        "bio_core_dna_prot_spliceai": "A_no_geometry",
        "bio_core_dna_prot_spliceai_geom": "B_with_geometry",
    }
    def _add_fixed_baseline(path, source_name):
        path = Path(path)
        if not path.exists():
            print(f"[SKIP] DL baseline file not found: {path}")
            return
        baseline = pd.read_csv(path)
        required = {"Dataset", "Ablation", "Network", "MCC", "AUROC", "AUPRC"}
        if not required.issubset(baseline.columns):
            print(f"[SKIP] Baseline schema is not recognized: {path}")
            return
        baseline = baseline[
            baseline["Ablation"].isin(BASELINE_ABLATIONS)
            & baseline["Network"].astype(str).str.startswith("PyTorch_")
            & baseline["Dataset"].isin(TEST_DATASETS)
        ].copy()
        if "Pooling" in baseline:
            baseline = baseline[baseline["Pooling"] == "center"]
        if "DNA_Model" in baseline:
            baseline = baseline[baseline["DNA_Model"] == "nt_v1_500m_dna_1001"]
        if "Prot_Model" in baseline:
            baseline = baseline[baseline["Prot_Model"] == "esm2_650m_prot_501"]
        baseline["Target_Config"] = baseline["Ablation"].map(BASELINE_ABLATIONS)
        baseline["Architecture"] = baseline["Network"]
        baseline["Variant"] = "baseline_control"
        baseline["Profile"] = "baseline_control"
        baseline["Seed"] = ""
        baseline["Candidate_Level"] = "single_run"
        validation_columns = [col for col in ["Dataset", "Ablation", "Network", "Decision_Threshold", "Val_MCC", "Val_AUROC", "Val_AUPRC"] if col in baseline]
        if validation_columns:
            baseline_validation = baseline[validation_columns].copy()
            baseline_validation["Source"] = source_name
            baseline_validation["Target_Config"] = baseline_validation["Ablation"].map(BASELINE_ABLATIONS)
            baseline_validation["Architecture"] = baseline_validation["Network"]
            baseline_validation["Variant"] = "baseline_control"
            baseline_validation["Profile"] = "baseline_control"
            baseline_validation_frames.append(baseline_validation)
        for _, row in baseline.iterrows():
            all_dl_rows.append({
                "Source": source_name, "Candidate_Level": "single_run",
                "Target_Config": row["Target_Config"], "Dataset": row["Dataset"],
                "Architecture": row["Architecture"], "Variant": row["Variant"],
                "Profile": row["Profile"], "Seed": "", "N_Seeds": 1,
                "Decision_Threshold": row.get("Decision_Threshold", float("nan")),
                "Accuracy": row.get("Accuracy", float("nan")), "Precision": row.get("Precision", float("nan")),
                "Recall": row.get("Recall", float("nan")), "Specificity": row.get("Specificity", float("nan")),
                "F1_Score": row.get("F1_Score", float("nan")),
                "MCC": row["MCC"], "AUROC": row["AUROC"], "AUPRC": row["AUPRC"],
                "MCC_SD": float("nan"), "AUROC_SD": float("nan"), "AUPRC_SD": float("nan"),
                "Hyperparameters": json.dumps({
                    key: row[key] for key in ["Pooling", "DNA_Model", "Prot_Model"]
                    if key in row.index and pd.notna(row[key])
                }, ensure_ascii=False, sort_keys=True, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
            })
    
    _add_fixed_baseline(Path(pipeline.batch_run_dir) / "fixed_config_baseline.csv", "fixed_config_baseline_current_run")
    if BASELINE_BATCH_RUN_DIR is not None:
        _add_fixed_baseline(Path(BASELINE_BATCH_RUN_DIR) / "global_leaderboard_metrics.csv", "baseline_batchrun_20261002")
    
    # Summarize all seed-confirmation metrics (including accuracy/precision/recall) by profile and dataset.
    seed_test_frame = pd.read_csv(SEED_CONFIRM_DIR / "seed_confirmation_test_metrics.csv")
    seed_summary_frame = pd.read_csv(SEED_CONFIRM_DIR / "seed_confirmation_summary.csv")
    seed_group_keys = ["Target_Config", "Architecture", "Variant", "Profile", "Dataset"]
    seed_metric_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "MCC", "AUROC", "AUPRC"]
    seed_grouped = seed_test_frame.groupby(seed_group_keys, dropna=False)
    for group_key, group in seed_grouped:
        target_config, architecture, variant, profile, dataset = group_key
        summary_match = seed_summary_frame[
            (seed_summary_frame["Target_Config"] == target_config)
            & (seed_summary_frame["Architecture"] == architecture)
            & (seed_summary_frame["Variant"] == variant)
            & (seed_summary_frame["Profile"] == profile)
        ]
        summary_row = summary_match.iloc[0] if not summary_match.empty else pd.Series(dtype=object)
        all_dl_rows.append({
            "Source": "seed_confirmation_24_seed_mean", "Candidate_Level": "24_seed_mean",
            "Target_Config": target_config, "Dataset": dataset,
            "Architecture": architecture, "Variant": variant, "Profile": profile,
            "Seed": "24-seed mean", "N_Seeds": group["Seed"].nunique(),
            "Decision_Threshold": group["Decision_Threshold"].mean(),
            "Val_MCC": summary_row.get("Val_MCC_Mean", float("nan")),
            "Val_AUROC": summary_row.get("Val_AUROC_Mean", float("nan")),
            "Val_AUPRC": summary_row.get("Val_AUPRC_Mean", float("nan")),
            "Best_Epoch": summary_row.get("Best_Epoch_Mean", float("nan")),
            "Fit_Seconds": float("nan"), "Resumed_From_Checkpoint": float("nan"),
            **{metric: group[metric].mean() for metric in seed_metric_names},
            "MCC_SD": group["MCC"].std(ddof=1), "AUROC_SD": group["AUROC"].std(ddof=1),
            "AUPRC_SD": group["AUPRC"].std(ddof=1),
            "Hyperparameters": "24-seed mean; see seed_confirmation_trials.csv for settings",
        })
    
    all_dl_results = pd.DataFrame(all_dl_rows, columns=_RESULT_COLUMNS)
    if all_dl_results.empty:
        raise FileNotFoundError("No completed DL test-result CSVs were found.")
    all_dl_results = all_dl_results[all_dl_results["Dataset"].isin(TEST_DATASETS)].copy()
    # Keep individual seed records in the full inventory, but rank seed-confirmed profiles by their mean.
    ranking_pool = all_dl_results[all_dl_results["Candidate_Level"] != "individual_seed"].copy()
    # Collapse repeated evaluations of the same candidate across notebook stages; retain their provenance and mean metrics.
    candidate_key = ["Candidate_Level", "Target_Config", "Dataset", "Architecture", "Variant", "Profile", "Hyperparameters"]
    ranking_pool = (ranking_pool.groupby(candidate_key, dropna=False, as_index=False)
                    .agg(Source=("Source", lambda values: "; ".join(sorted(set(values)))),
                         Result_Runs=("MCC", "size"), Seed=("Seed", "first"), N_Seeds=("N_Seeds", "max"),
                         Decision_Threshold=("Decision_Threshold", "mean"),
                         Val_MCC=("Val_MCC", "mean"), Val_AUROC=("Val_AUROC", "mean"), Val_AUPRC=("Val_AUPRC", "mean"),
                         Best_Epoch=("Best_Epoch", "mean"), Fit_Seconds=("Fit_Seconds", "mean"),
                         Accuracy=("Accuracy", "mean"), Precision=("Precision", "mean"), Recall=("Recall", "mean"),
                         Specificity=("Specificity", "mean"), F1_Score=("F1_Score", "mean"),
                         MCC=("MCC", "mean"), AUROC=("AUROC", "mean"), AUPRC=("AUPRC", "mean"),
                         MCC_SD=("MCC_SD", "mean"), AUROC_SD=("AUROC_SD", "mean"), AUPRC_SD=("AUPRC_SD", "mean")))
    ranking_pool = ranking_pool.sort_values(["Dataset", "MCC", "AUROC", "AUPRC"], ascending=[True, False, False, False])
    ranking_pool["Rank_MCC_within_Test"] = ranking_pool.groupby("Dataset")["MCC"].rank(method="first", ascending=False).astype(int)
    top3_by_test = ranking_pool[ranking_pool["Rank_MCC_within_Test"] <= 3].copy()
    if set(top3_by_test["Dataset"]) != set(TEST_DATASETS):
        raise ValueError(f"Could not build top-3 rankings for every test dataset: {top3_by_test['Dataset'].unique()}")
    
    all_dl_trial_frames.extend(baseline_validation_frames)
    if all_dl_trial_frames:
        all_dl_validation_trials = pd.concat(all_dl_trial_frames, ignore_index=True, sort=False)
        all_dl_validation_trials.to_csv(FINAL_DL_DIR / "all_dl_validation_trials.csv", index=False)
    else:
        all_dl_validation_trials = pd.DataFrame()
    all_dl_results.to_csv(FINAL_DL_DIR / "all_dl_test_results.csv", index=False)
    top3_by_test.to_csv(FINAL_DL_DIR / "top3_dl_by_test_mcc.csv", index=False)
    print(f"Consolidated {len(all_dl_validation_trials)} DL validation/trial rows and {len(all_dl_results)} test rows from completed search stages.")
    print("Top 3 distinct evaluated candidates per test dataset, ranked by MCC. Seed-confirmed profiles use their 24-seed mean.")
    global_metrics = refresh_global_metrics_from_context(context, display_fn=display_fn, show=False)
    print(f"[GLOBAL RESULTS] Updated {global_metrics['path']}")
    return {
        "all_dl_test_results": all_dl_results,
        "all_dl_validation_trials": all_dl_validation_trials,
        "top3_by_test": top3_by_test,
    }
