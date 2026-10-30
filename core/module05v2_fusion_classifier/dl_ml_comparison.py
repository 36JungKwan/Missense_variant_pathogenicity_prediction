"""Validation-selected comparison and test-set reporting for DL, ML, and Hybrid."""


def run_three_branch_comparison(pipeline, display_fn=None):
    """Select one candidate per branch/config using validation MCC and compare tests."""
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    display = display_fn or print
    root = Path(pipeline.batch_run_dir)
    out = root / "three_branch_comparison"
    out.mkdir(parents=True, exist_ok=True)
    datasets = [
        "Train3Val_Test", "Train3Val_ClinVarHQ",
        "Train3Val_ProteinGym", "Train3Val_UniProt",
    ]
    configs = ["A_no_geometry", "B_with_geometry"]

    # DL: use completed candidate-level rows; seed-level rows are summarized elsewhere.
    dl_path = root / "pure_dl_final_v1" / "all_dl_test_results.csv"
    dl_all = pd.read_csv(dl_path)
    dl = dl_all[
        dl_all["Architecture"].astype(str).str.startswith("PyTorch_")
        & (dl_all["Candidate_Level"] != "individual_seed")
        & (dl_all["Source"] != "baseline_batchrun_20261002")
        & dl_all["Target_Config"].isin(configs)
        & dl_all["Dataset"].isin(datasets)
    ].copy()
    dl_key = ["Target_Config", "Architecture", "Variant", "Profile", "Candidate_Level", "Hyperparameters"]
    dl = dl.groupby(dl_key + ["Dataset"], dropna=False, as_index=False).agg(
        Source=("Source", lambda v: "; ".join(sorted(set(v.dropna().astype(str))))),
        N_Seeds=("N_Seeds", "max"), Val_MCC=("Val_MCC", "mean"),
        MCC=("MCC", "mean"), AUROC=("AUROC", "mean"), AUPRC=("AUPRC", "mean"),
    )
    dl_coverage = dl.groupby(dl_key, dropna=False)["Dataset"].nunique()
    dl_complete_keys = dl_coverage[dl_coverage == len(datasets)].reset_index()[dl_key]
    dl = dl.merge(dl_complete_keys, on=dl_key, how="inner")
    if dl.empty:
        raise ValueError(f"No complete validation-selected DL candidates in {dl_path}")
    dl_all_candidates = dl.copy()
    dl_candidates = dl.groupby(dl_key, dropna=False, as_index=False).agg(
        Val_MCC=("Val_MCC", "mean"), N_Seeds=("N_Seeds", "max")
    ).sort_values(["Target_Config", "Val_MCC"], ascending=[True, False]).drop_duplicates("Target_Config")
    dl = dl.merge(dl_candidates[dl_key], on=dl_key, how="inner")
    dl["Branch"] = "DL"
    dl["Model"] = dl["Architecture"].str.replace("PyTorch_", "", regex=False) + ": " + dl["Variant"].astype(str)
    profile_mask = dl["Profile"].notna() & dl["Profile"].astype(str).ne("")
    dl.loc[profile_mask, "Model"] += " [" + dl.loc[profile_mask, "Profile"].astype(str) + "]"
    dl = dl[["Target_Config", "Dataset", "Branch", "Model", "Candidate_Level", "N_Seeds", "Val_MCC", "MCC", "AUROC", "AUPRC", "Source"]]

    # ML: join validation-selected grid winners to their held-out test metrics.
    def load_ml(test_rel, val_rel, source):
        test = pd.read_csv(root / test_rel)
        val = pd.read_csv(root / val_rel)
        if "Model" in test and test["Model"].astype(str).str.contains("GridBest").any():
            test = test[test["Model"].astype(str).str.contains("GridBest")].copy()
        v_mcc = "Val_MCC_at_Threshold" if "Val_MCC_at_Threshold" in val else "MCC"
        v_auprc = "Val_AUPRC" if "Val_AUPRC" in val else "AUPRC"
        keep = val[["Target_Config", "Model", "Selected_Trial", v_mcc, v_auprc]].drop_duplicates(
            ["Target_Config", "Model", "Selected_Trial"]
        ).rename(columns={v_mcc: "Val_MCC", v_auprc: "Val_AUPRC"})
        result = test.merge(keep, on=["Target_Config", "Model", "Selected_Trial"], how="left", validate="many_to_one")
        result["Branch"] = "ML"
        result["Model"] = result["Model"].astype(str)
        result["Candidate_Level"] = "grid_best"
        result["N_Seeds"] = 1
        result["Source"] = source
        return result[["Target_Config", "Dataset", "Branch", "Model", "Candidate_Level", "N_Seeds", "Val_MCC", "Val_AUPRC", "MCC", "AUROC", "AUPRC", "Source"]]

    ml = pd.concat([
        load_ml("xgb_no_pca/xgb_no_pca_test_metrics.csv", "xgb_no_pca/xgb_no_pca_validation_metrics.csv", "XGBoost no PCA"),
        load_ml("lightgbm/lightgbm_test_metrics.csv", "lightgbm/lightgbm_validation_metrics.csv", "LightGBM"),
        load_ml("catboost/catboost_test_metrics.csv", "catboost/catboost_validation_metrics.csv", "CatBoost"),
    ], ignore_index=True)
    baseline = pd.read_csv(root / "fixed_config_baseline.csv")
    baseline = baseline[baseline["Network"].eq("Pure_XGBoost_Concat")].copy()
    baseline["Target_Config"] = np.where(baseline["Ablation"].astype(str).str.endswith("_geom"), "B_with_geometry", "A_no_geometry")
    if "Val_AUPRC" not in baseline:
        baseline["Val_AUPRC"] = np.nan
    baseline["Branch"], baseline["Model"], baseline["Candidate_Level"], baseline["N_Seeds"], baseline["Source"] = "ML", "XGBoost_PCA_Baseline", "baseline", 1, "XGBoost PCA baseline"
    ml = pd.concat([ml, baseline[["Target_Config", "Dataset", "Branch", "Model", "Candidate_Level", "N_Seeds", "Val_MCC", "Val_AUPRC", "MCC", "AUROC", "AUPRC", "Source"]]], ignore_index=True)
    ml = ml[ml["Target_Config"].isin(configs) & ml["Dataset"].isin(datasets)].dropna(subset=["Val_MCC", "MCC"])
    ml_all_candidates = ml.copy()
    ml_rank = ml.groupby(["Target_Config", "Model"], as_index=False).agg(Val_MCC=("Val_MCC", "mean"), Val_AUPRC=("Val_AUPRC", "mean"))
    ml_choice = ml_rank.sort_values(["Target_Config", "Val_MCC", "Val_AUPRC"], ascending=[True, False, False]).drop_duplicates("Target_Config")[["Target_Config", "Model"]]
    ml = ml.merge(ml_choice, on=["Target_Config", "Model"], how="inner")

    # Hybrid uses the completed Optuna search and seed-confirmation results.
    # The earlier exploratory table stays archived and is excluded from reports.
    hybrid_path = root / "hybrid_optuna_v2" / "hybrid_test_metrics.csv"
    if not hybrid_path.exists():
        raise FileNotFoundError(f"Completed Hybrid results were not found: {hybrid_path}")
    hybrid = pd.read_csv(hybrid_path)
    hybrid["Target_Config"] = np.where(hybrid["Ablation"].astype(str).str.endswith("_geom"), "B_with_geometry", "A_no_geometry")
    hybrid = hybrid[hybrid["Network"].astype(str).str.startswith("Hybrid_") & hybrid["Dataset"].isin(datasets)].copy()
    hybrid["Model"] = "Hybrid: " + hybrid["Network"].astype(str)
    hybrid["Branch"], hybrid["Candidate_Level"], hybrid["N_Seeds"], hybrid["Source"] = "Hybrid", "optuna_candidate", hybrid.get("Confirmation_Seeds", 1), "Hybrid"
    if "Feature_Set" not in hybrid:
        hybrid["Feature_Set"] = "latent_only"
    hybrid_all_candidates = hybrid.copy()
    hy_rank = hybrid.groupby(["Target_Config", "Model", "Source"], as_index=False).agg(Val_MCC=("Val_MCC", "mean"), Val_AUPRC=("Val_AUPRC", "mean"))
    hy_choice = hy_rank.sort_values(["Target_Config", "Val_MCC", "Val_AUPRC"], ascending=[True, False, False]).drop_duplicates("Target_Config")[["Target_Config", "Model", "Source"]]
    hybrid = hybrid.merge(hy_choice, on=["Target_Config", "Model", "Source"], how="inner")
    hybrid = hybrid[["Target_Config", "Dataset", "Branch", "Model", "Candidate_Level", "N_Seeds", "Val_MCC", "Val_AUPRC", "MCC", "AUROC", "AUPRC", "Source"]]

    long = pd.concat([dl, ml, hybrid], ignore_index=True, sort=False)
    long = long[long["Target_Config"].isin(configs) & long["Dataset"].isin(datasets)]
    if long.duplicated(["Target_Config", "Dataset", "Branch"]).any():
        raise ValueError("Expected one validation-selected model per branch/configuration/test set")
    expected = len(configs) * len(datasets) * 3
    if len(long) != expected:
        raise ValueError(f"Expected {expected} selected branch/test rows; found {len(long)}")
    long["Test_Set"] = long["Dataset"].map({"Train3Val_Test": "Test", "Train3Val_ClinVarHQ": "ClinVarHQ", "Train3Val_ProteinGym": "ProteinGym", "Train3Val_UniProt": "UniProt"})

    metrics = ["Model", "Val_MCC", "MCC", "AUROC", "AUPRC"]
    wide = long.pivot(index=["Target_Config", "Dataset", "Test_Set"], columns="Branch", values=metrics).reset_index()
    wide.columns = ["_".join(str(part) for part in col if part) if isinstance(col, tuple) else col for col in wide.columns]
    wide = wide.sort_values(["Target_Config", "Dataset"]).reset_index(drop=True)
    mcc_cols = [f"MCC_{branch}" for branch in ("DL", "ML", "Hybrid")]
    wide["Best_Branch_by_MCC"] = wide[mcc_cols].idxmax(axis=1).str.replace("MCC_", "", regex=False)

    summary = long.groupby(["Target_Config", "Branch"], as_index=False).agg(
        Macro_MCC_4_tests=("MCC", "mean"), Macro_AUROC_4_tests=("AUROC", "mean"), Macro_AUPRC_4_tests=("AUPRC", "mean")
    )
    wins = long.loc[long.groupby(["Target_Config", "Dataset"])["MCC"].idxmax()].groupby(["Target_Config", "Branch"]).size().rename("Test_sets_won_by_MCC").reset_index()
    summary = summary.merge(wins, on=["Target_Config", "Branch"], how="left").fillna({"Test_sets_won_by_MCC": 0})
    # This is a descriptive, test-ranked report over all completed candidates. It
    # does not participate in validation-based model selection above.
    dl_top = dl_all_candidates.copy()
    dl_top["Model_Label"] = (
        dl_top["Architecture"].astype(str).str.replace("PyTorch_", "", regex=False)
        + ": " + dl_top["Variant"].astype(str)
    )
    dl_top["Model_Label"] += dl_top["Profile"].fillna("").astype(str).map(lambda p: f" [{p}]" if p else "")
    ml_top = ml_all_candidates.rename(columns={"Model": "Model_Label"})
    hybrid_top = hybrid_all_candidates.rename(columns={"Model": "Model_Label"})
    top3_pool = pd.concat([dl_top, ml_top, hybrid_top], ignore_index=True, sort=False)
    top3_pool = top3_pool.dropna(subset=["MCC"])
    top3 = top3_pool.sort_values(["Dataset", "MCC", "Val_MCC"], ascending=[True, False, False]).groupby(
        "Dataset", as_index=False, group_keys=False
    ).head(3).copy()
    top3["Rank_MCC_within_Test"] = top3.groupby("Dataset")["MCC"].rank(method="first", ascending=False).astype(int)

    long.to_csv(out / "three_branch_validation_selected_results_long.csv", index=False)
    wide.to_csv(out / "three_branch_test_comparison.csv", index=False)
    summary.to_csv(out / "three_branch_macro_summary.csv", index=False)
    top3.to_csv(out / "three_branch_top3_by_test.csv", index=False)
    print("Candidate selection uses validation MCC; held-out test metrics are reported after selection. Hybrid results come from the completed search and seed confirmation.")
    print("Selected models and test metrics by configuration:")
    display(wide[["Target_Config", "Test_Set", "Model_DL", "MCC_DL", "Model_ML", "MCC_ML", "Model_Hybrid", "MCC_Hybrid", "Best_Branch_by_MCC"]].round(4))
    print("Branch macro averages over the four test sets:")
    display(summary.round(4))
    print("Top 3 among all completed candidates per test dataset (test-ranked for reporting only; not used to select models):")
    display(top3[["Dataset", "Rank_MCC_within_Test", "Target_Config", "Branch", "Model_Label", "Val_MCC", "MCC", "AUROC", "AUPRC", "Source"]].round(4))

    short = {"Train3Val_Test": "Test", "Train3Val_ClinVarHQ": "ClinVarHQ", "Train3Val_ProteinGym": "ProteinGym", "Train3Val_UniProt": "UniProt"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    colors = {"DL": "#4472C4", "ML": "#ED7D31", "Hybrid": "#70AD47"}
    for ax, cfg in zip(axes, configs):
        part = long[long["Target_Config"].eq(cfg)]
        labels = [short[d] for d in datasets]
        x = np.arange(len(datasets)); width = .24
        for j, branch in enumerate(("DL", "ML", "Hybrid")):
            values = part[part["Branch"].eq(branch)].set_index("Dataset").reindex(datasets)["MCC"]
            ax.bar(x + (j - 1) * width, values, width, label=branch, color=colors[branch])
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_title("A: no geometry" if cfg == configs[0] else "B: geometry")
        ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.25); ax.set_ylabel("MCC")
    axes[0].legend(frameon=False)
    fig.suptitle("MCC on each held-out test set for the validation-selected model in each branch")
    fig.tight_layout()
    fig.savefig(out / "three_branch_test_mcc_comparison.png", dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Comparison artifacts saved in: {out}")
    return {"comparison_long": long, "test_comparison": wide, "macro_summary": summary, "top3_by_test": top3, "output_dir": out}


# Backward-compatible entry point for notebooks that still use the old function name.
