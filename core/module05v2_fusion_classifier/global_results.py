"""Merge completed fixed-configuration results into one batch leaderboard."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


GLOBAL_COLUMNS = [
    "Run_ID", "Branch", "Evaluation_Split", "Target_Config", "Dataset",
    "Pooling", "Ablation", "DNA_Model", "Prot_Model", "Network",
    "Architecture", "Variant", "Classifier", "Feature_Set", "PCA",
    "Source", "Candidate_Level", "Profile", "Seed", "N_Seeds",
    "Selected_Trial", "Decision_Threshold", "Accuracy", "Precision",
    "Recall", "Specificity", "F1_Score", "MCC", "AUROC", "AUPRC",
    "Val_Accuracy", "Val_Precision", "Val_Recall", "Val_Specificity",
    "Val_F1_Score", "Val_MCC", "Val_AUROC", "Val_AUPRC", "MCC_SD",
    "AUROC_SD", "AUPRC_SD", "MCC_Seed_STD", "AUROC_Seed_STD",
    "AUPRC_Seed_STD", "Best_Epoch", "Fit_Seconds", "Parameters",
    "Hyperparameters", "Fusion_Hyperparameters", "Geom_Features",
    "Source_File",
]

METRICS = (
    "Accuracy", "Precision", "Recall", "Specificity", "F1_Score",
    "MCC", "AUROC", "AUPRC",
)
GLOBAL_RESULTS_SORT_BY = "MCC"


def refresh_global_metrics(
    pipeline,
    target_configs,
    datasets,
    dna_model,
    protein_model,
    pooling="center",
    display_fn=None,
    sort_by=None,
    show=True,
):
    """Rebuild the global test-results CSV from completed stage artifacts.

    Missing stage files and rows without a test MCC are skipped. The old global
    CSV is deliberately not read as an input, so rerunning this function cannot
    append duplicate rows from an earlier refresh.
    """
    display = display_fn or print
    root = Path(pipeline.batch_run_dir).resolve()
    config_by_name = {str(config["name"]): config for config in target_configs}
    config_by_ablation = {
        str(config["ablation"]): str(config["name"])
        for config in target_configs
    }
    expected_datasets = {str(item["name"]) for item in datasets}
    geometry_label = {
        name: "+".join(config.get("geom_feature_names") or []) or pd.NA
        for name, config in config_by_name.items()
    }

    sources = [
        ("Baseline", "fixed_config_baseline.csv", None, None),
        ("DL", "pure_dl_final_v1/all_dl_test_results.csv", None, None),
        ("ML", "xgb_no_pca/xgb_no_pca_test_metrics.csv",
         "xgb_no_pca/xgb_no_pca_validation_metrics.csv", "xgb_no_pca"),
        ("ML", "lightgbm/lightgbm_test_metrics.csv",
         "lightgbm/lightgbm_validation_metrics.csv", "lightgbm"),
        ("ML", "catboost/catboost_test_metrics.csv",
         "catboost/catboost_validation_metrics.csv", "catboost"),
        ("Hybrid", "hybrid_optuna_v2/hybrid_test_metrics.csv", None, None),
    ]

    frames = []
    status_rows = []

    def _read_csv(relative_path):
        path = root / relative_path
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _set_config_columns(frame):
        if "Target_Config" not in frame:
            frame["Target_Config"] = pd.NA
        if "Ablation" not in frame:
            frame["Ablation"] = pd.NA
        if "Target_Config" in frame:
            missing_config = frame["Target_Config"].isna()
            frame.loc[missing_config, "Target_Config"] = (
                frame.loc[missing_config, "Ablation"].astype(str)
                .map(config_by_ablation)
            )
        missing_ablation = frame["Ablation"].isna()
        frame.loc[missing_ablation, "Ablation"] = (
            frame.loc[missing_ablation, "Target_Config"].astype(str)
            .map({name: cfg["ablation"] for name, cfg in config_by_name.items()})
        )
        frame["Geom_Features"] = frame.get("Geom_Features", pd.Series(pd.NA, index=frame.index))
        missing_geom = frame["Geom_Features"].isna()
        frame.loc[missing_geom, "Geom_Features"] = (
            frame.loc[missing_geom, "Target_Config"].astype(str).map(geometry_label)
        )
        return frame

    def _backfill_dl_validation(frame):
        """Fill validation fields omitted from selected DL test summaries.

        The consolidated validation-trial file retains the validation metrics
        for the selected baseline/Optuna candidates. Match within each stage so
        equal architecture names from different searches cannot collide.
        """
        validation = _read_csv("pure_dl_final_v1/all_dl_validation_trials.csv")
        if validation is None or validation.empty or "Source" not in validation:
            return frame

        val_metric_columns = [
            "Val_Accuracy", "Val_Precision", "Val_Recall", "Val_Specificity",
            "Val_F1_Score", "Val_MCC", "Val_AUROC", "Val_AUPRC",
            "Decision_Threshold", "Best_Epoch", "Fit_Seconds",
        ]
        stage_match = {
            "fixed_config_baseline_current_run": (
                "fixed_config_baseline_current_run", ["Target_Config", "Architecture", "Variant"]
            ),
            "architecture_search_v1": (
                "architecture_search_v1_trials", ["Target_Config", "Architecture", "Variant"]
            ),
            "architecture_hparam_search_v1": (
                "architecture_hparam_search_v1_trials", ["Target_Config", "Architecture", "Variant"]
            ),
            "architecture_variants_v2": (
                "architecture_variants_v2_trials", ["Target_Config", "Architecture", "Variant"]
            ),
            "architecture_variants_v3": (
                "architecture_variants_v3_trials", ["Target_Config", "Architecture", "Variant"]
            ),
            "regularization_optuna_v1": (
                "regularization_optuna_v1_trials", ["Target_Config", "Architecture", "Trial"]
            ),
            "regularization_optuna_v2": (
                "regularization_optuna_v2_trials", ["Target_Config", "Architecture", "Trial"]
            ),
            "training_hparam_optuna_best": (
                "training_hparam_optuna_trials", ["Target_Config", "Architecture", "Variant", "Profile"]
            ),
            "seed_confirmation_24_seed_mean": (
                "seed_confirmation_trials", ["Target_Config", "Architecture", "Variant", "Profile"]
            ),
            "seed_confirmation_individual_seeds": (
                "seed_confirmation_trials", ["Target_Config", "Architecture", "Variant", "Profile", "Seed"]
            ),
        }
        if "Source" not in frame:
            return frame

        frame = frame.copy()
        for test_source, (validation_source, keys) in stage_match.items():
            mask = frame["Source"].astype(str).eq(test_source)
            if not mask.any():
                continue
            left = frame.loc[mask].copy()
            right = validation.loc[
                validation["Source"].astype(str).eq(validation_source)
            ].copy()
            if right.empty:
                continue

            # Selected regularization trial is stored as Profile in the test
            # summary but as Trial in the validation-trial ledger.
            if "Trial" in keys and "Profile" in keys:
                keys = ["Profile" if key == "Trial" else key for key in keys]
            elif "Trial" in keys:
                left["_Trial_Key"] = left.get("Profile", pd.Series(pd.NA, index=left.index)).astype("string")
                right["_Trial_Key"] = right["Trial"].astype("string")
                keys = ["Target_Config", "Architecture", "_Trial_Key"]

            usable_keys = [key for key in keys if key in left.columns and key in right.columns]
            metrics = [column for column in val_metric_columns if column in right.columns]
            if not usable_keys or not metrics:
                continue
            # CSV inference can load the same seed as "42" on the test side
            # and 42.0 on the validation side. Build normalized temporary keys
            # so dtype differences do not make a valid match fail.
            match_keys = []
            for index, key in enumerate(usable_keys):
                match_key = f"_dl_validation_join_{index}"
                if key == "Seed":
                    left[match_key] = pd.to_numeric(left[key], errors="coerce")
                    right[match_key] = pd.to_numeric(right[key], errors="coerce")
                else:
                    left[match_key] = left[key].astype("string").fillna("")
                    right[match_key] = right[key].astype("string").fillna("")
                match_keys.append(match_key)
            lookup = right[match_keys + metrics].drop_duplicates(match_keys)
            merged = left.merge(
                lookup, on=match_keys, how="left", suffixes=("", "_from_validation"),
                validate="many_to_one",
            )
            for column in metrics:
                source_column = f"{column}_from_validation" if column in left.columns else column
                if source_column not in merged.columns:
                    continue
                values = merged[source_column]
                if column in left.columns:
                    values = merged[column].where(merged[column].notna(), values)
                if column not in frame.columns:
                    frame[column] = pd.NA
                frame.loc[mask, column] = values.to_numpy()
        return frame

    def _backfill_hybrid_metrics(frame):
        """Restore aggregate metrics available in per-seed Hybrid results."""
        seed_metrics = _read_csv("hybrid_optuna_v2/hybrid_seed_confirmation_metrics.csv")
        if seed_metrics is None or seed_metrics.empty:
            return frame
        keys = [
            "Target_Config", "Dataset", "Architecture", "Variant",
            "Classifier", "Feature_Set",
        ]
        if not all(key in frame.columns and key in seed_metrics.columns for key in keys):
            return frame
        metric_columns = [
            column for column in ("Accuracy", "Precision", "Recall", "Specificity", "F1_Score")
            if column in seed_metrics.columns
        ]
        if not metric_columns:
            return frame
        grouped = seed_metrics.groupby(keys, dropna=False)[metric_columns].mean().reset_index()
        merged = frame.merge(
            grouped, on=keys, how="left", suffixes=("", "_from_seeds"),
            validate="many_to_one",
        )
        for column in metric_columns:
            source_column = f"{column}_from_seeds" if column in frame.columns else column
            if column not in merged.columns:
                merged[column] = pd.NA
            merged[column] = merged[column].where(merged[column].notna(), merged[source_column])
            if source_column != column:
                merged.drop(columns=[source_column], inplace=True)
        return merged

    for branch, test_rel, validation_rel, model_family in sources:
        test_frame = _read_csv(test_rel)
        if test_frame is None:
            status_rows.append({"Branch": branch, "Source_File": test_rel, "Rows_Imported": 0, "Status": "missing; skipped"})
            continue
        if test_frame.empty:
            status_rows.append({"Branch": branch, "Source_File": test_rel, "Rows_Imported": 0, "Status": "empty; skipped"})
            continue

        frame = test_frame.copy()
        if branch == "DL":
            frame["Branch"] = "DL"
            source_col = frame.get("Source", pd.Series("", index=frame.index)).astype(str)
            obsolete_or_baseline = source_col.str.contains(
                "baseline_batchrun|fixed_config_baseline_current_run",
                case=False, regex=True, na=False,
            )
            frame = frame.loc[~obsolete_or_baseline].copy()
            frame = _backfill_dl_validation(frame)
        elif branch == "Baseline":
            frame["Target_Config"] = frame["Ablation"].astype(str).map(config_by_ablation)
            frame["Network"] = frame.get("Network", pd.Series("", index=frame.index)).astype(str)
            frame["Branch"] = frame["Network"].map(
                lambda value: "ML" if value.startswith("Pure_XGBoost") else "DL"
            )
            frame["PCA"] = frame["Network"].eq("Pure_XGBoost_Concat")
        elif branch == "ML":
            validation = _read_csv(validation_rel) if validation_rel else None
            if validation is not None and not validation.empty:
                val = validation.copy()
                val = _set_config_columns(val)
                val_metric_names = {
                    "MCC": "Val_MCC", "AUROC": "Val_AUROC", "AUPRC": "Val_AUPRC",
                    "Accuracy": "Val_Accuracy", "Precision": "Val_Precision",
                    "Recall": "Val_Recall", "Specificity": "Val_Specificity",
                    "F1_Score": "Val_F1_Score",
                }
                val = val.rename(columns=val_metric_names)
                if "Model" not in frame and "Model" in val:
                    frame["Model"] = frame.get("Network")
                if "Model" not in val:
                    val["Model"] = val.get("Network")
                join_keys = [key for key in ("Target_Config", "Model")
                             if key in val.columns and key in frame.columns]
                val_columns = join_keys + [column for column in val_metric_names.values() if column in val.columns]
                if join_keys:
                    frame = frame.merge(
                        val[val_columns].drop_duplicates(join_keys),
                        on=join_keys, how="left", validate="many_to_one",
                    )
            frame["Branch"] = "ML"
            model_col = frame.get("Model", frame.get("Network", pd.Series("", index=frame.index))).astype(str)
            frame["Network"] = model_col
            frame["Classifier"] = model_col.map(
                lambda value: "LightGBM" if "LightGBM" in value
                else "CatBoost" if "CatBoost" in value else "XGBoost"
            )
            frame["PCA"] = model_col.str.contains("PCA", case=False) & ~model_col.str.contains("NoPCA", case=False)
        elif branch == "Hybrid":
            frame["Branch"] = "Hybrid"
            frame["Network"] = frame.get("Network", frame.get("Model", pd.Series("", index=frame.index)))
            if "N_Seeds" not in frame and "Confirmation_Seeds" in frame:
                frame["N_Seeds"] = frame["Confirmation_Seeds"]
            elif "Confirmation_Seeds" in frame:
                frame["N_Seeds"] = frame["N_Seeds"].where(
                    frame["N_Seeds"].notna(), frame["Confirmation_Seeds"]
                )
            frame = _backfill_hybrid_metrics(frame)

        frame = _set_config_columns(frame)
        if "Dataset" not in frame:
            frame["Dataset"] = pd.NA
        frame = frame[
            frame["Target_Config"].astype(str).isin(config_by_name)
            & frame["Dataset"].astype(str).isin(expected_datasets)
        ].copy()
        if "MCC" not in frame:
            frame["MCC"] = pd.NA
        frame["MCC"] = pd.to_numeric(frame["MCC"], errors="coerce")
        frame = frame.loc[frame["MCC"].notna()].copy()
        if frame.empty:
            status_rows.append({"Branch": branch, "Source_File": test_rel, "Rows_Imported": 0, "Status": "no completed test rows; skipped"})
            continue

        if "Model" in frame and "Network" not in frame:
            frame["Network"] = frame["Model"]
        elif "Model" in frame and branch == "Hybrid":
            frame["Network"] = frame["Model"]
        elif branch == "DL" and "Network" not in frame:
            architecture = frame.get("Architecture", pd.Series("DL", index=frame.index)).astype(str)
            variant = frame.get("Variant", pd.Series("default", index=frame.index)).fillna("default").astype(str)
            network = architecture + "__" + variant
            if "Profile" in frame:
                profile = frame["Profile"].fillna("").astype(str)
                network = network + profile.map(lambda value: f"__{value}" if value else "")
            frame["Network"] = network
        if "Architecture" not in frame:
            frame["Architecture"] = frame.get("Network", pd.Series(pd.NA, index=frame.index))
        if "Candidate_Level" not in frame:
            frame["Candidate_Level"] = "baseline" if branch == "Baseline" else "test_result"
        if "Source" not in frame:
            frame["Source"] = test_rel
        if branch in ("Baseline", "ML"):
            frame["Source"] = test_rel
        frame["Run_ID"] = root.name
        frame["Evaluation_Split"] = "Test"
        frame["Pooling"] = frame.get("Pooling", pooling)
        frame["DNA_Model"] = frame.get("DNA_Model", dna_model)
        frame["Prot_Model"] = frame.get("Prot_Model", protein_model)
        frame["Source_File"] = test_rel
        if model_family:
            frame["Model_Family"] = model_family
        for column in GLOBAL_COLUMNS:
            if column not in frame:
                frame[column] = pd.NA
        frames.append(frame[GLOBAL_COLUMNS])
        status_rows.append({"Branch": branch, "Source_File": test_rel, "Rows_Imported": len(frame), "Status": "imported"})

    if not frames:
        raise FileNotFoundError("No completed test-result files were found; global leaderboard was left unchanged.")

    result = pd.concat(frames, ignore_index=True, sort=False)
    sort_by = sort_by or GLOBAL_RESULTS_SORT_BY
    sortable_metrics = set(METRICS) | {"Val_MCC", "Val_AUROC", "Val_AUPRC"}
    if sort_by not in sortable_metrics:
        raise ValueError(f"sort_by must be one of {sorted(sortable_metrics)}; received {sort_by!r}")
    tie_breakers = [metric for metric in ("MCC", "AUROC", "AUPRC") if metric != sort_by]
    sort_columns = [sort_by] + tie_breakers + [
        "Branch", "Target_Config", "Dataset", "Network", "Source", "Profile", "Seed"
    ]
    ascending = [False] * (1 + len(tie_breakers)) + [True] * 7
    result = result.drop_duplicates().sort_values(
        sort_columns, ascending=ascending, kind="stable", na_position="last",
    ).reset_index(drop=True)
    output = root / "global_leaderboard_metrics.csv"
    temporary = output.with_name(output.name + ".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(output)

    status = pd.DataFrame(status_rows)
    summary = result.groupby(["Branch", "Target_Config", "Dataset"], dropna=False).size().rename("Completed_Rows").reset_index()
    print(f"[GLOBAL RESULTS] Wrote {len(result)} completed test-result rows to {output.name}; sorted by {sort_by} descending.")
    if show:
        display(status)
        display(summary)
    return {"results": result, "status": status, "summary": summary, "path": output}


def refresh_global_metrics_from_context(context, display_fn=None, show=True):
    """Refresh the canonical global table using a branch runner context."""
    return refresh_global_metrics(
        pipeline=context["pipeline"],
        target_configs=context["TARGET_CONFIGS"],
        datasets=context["datasets"],
        dna_model=context["DNA_MODEL"]["name"],
        protein_model=context["PROTEIN_MODEL"]["name"],
        pooling="center",
        display_fn=display_fn,
        sort_by=context.get("GLOBAL_RESULTS_SORT_BY", GLOBAL_RESULTS_SORT_BY),
        show=show,
    )
