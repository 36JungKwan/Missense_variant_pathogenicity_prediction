"""Stage 2: evaluate the validation-selected LID geometry across architectures."""

import os
import re

import pandas as pd

from core.module05_fusion_classifier.dataset import GEOM_V2_FEATURE_COLUMNS
from core.module05_fusion_classifier.pipeline import FusionBatchPipeline


BASE_DIR = "D:/variant_data"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BATCH_RUN_DIR = os.path.join(REPO_ROOT, "experiments", "batch_run_20260930_lid_ablation")
METRICS_PATH = os.path.join(BATCH_RUN_DIR, "global_leaderboard_metrics.csv")

MODELS_SPACE = [
    {"name": "nt_v2_500m", "seq_type": "dna"},
    {"name": "esm1b_650m", "seq_type": "protein"},
]
EXPERIMENTS = [
    {"name": "PyTorch_Concat", "type": "pytorch", "fusion": "concat"},
    {"name": "PyTorch_CrossAttn", "type": "pytorch", "fusion": "cross_attention"},
    {"name": "PyTorch_Transformer", "type": "pytorch", "fusion": "transformer"},
    {"name": "PyTorch_Gating", "type": "pytorch", "fusion": "gating"},
    {"name": "Pure_XGBoost_Concat", "type": "xgboost_pure", "fusion": "concat"},
    {"name": "Hybrid_Concat_XGBoost", "type": "hybrid", "fusion": "concat"},
    {"name": "Hybrid_CrossAttn_XGBoost", "type": "hybrid", "fusion": "cross_attention"},
    {"name": "Hybrid_Transformer_XGBoost", "type": "hybrid", "fusion": "transformer"},
    {"name": "Hybrid_Gating_XGBoost", "type": "hybrid", "fusion": "gating"},
]
DATASETS = [
    {"name": "Train3Val_Test", "train": "train3", "val": "val", "test": "test"},
    {"name": "Train3Val_ClinVarHQ", "train": "train3", "val": "val", "test": "clinvarhq"},
    {"name": "Train3Val_UniProt", "train": "train3", "val": "val", "test": "uniprot"},
    {"name": "Train3Val_ProteinGym", "train": "train3", "val": "val", "test": "proteingym"},
]


def choose_geometry():
    metrics = pd.read_csv(METRICS_PATH)
    stage1 = metrics[
        metrics["Ablation"].astype(str).str.startswith("bio_core_dna_prot_spliceai_geom_lid_")
        & ~metrics["Ablation"].astype(str).str.endswith("_architecture")
    ].copy()
    if stage1.empty:
        raise RuntimeError("Chua co ket qua Stage 1. Hay chay controllers/05_lid_k_ablation.py truoc.")

    grouped = (
        stage1.groupby(["Ablation", "Geom_Features"], dropna=False)
        .agg(Val_MCC=("Val_MCC", "first"), Val_AUROC=("Val_AUROC", "first"))
        .sort_values(["Val_MCC", "Val_AUROC"], ascending=False)
    )
    best_label, best_geom = grouped.index[0]
    match = re.search(r"_k(\d+)$", str(best_label))
    k = int(match.group(1)) if match else 32
    geometry_root = "geometry_v2" if k == 32 else f"geometry_v2_k{k}"
    features = [name for name in str(best_geom).split("+") if name]
    return best_label, k, geometry_root, features, grouped.iloc[0].to_dict()


def main():
    best_label, k, geometry_root, features, score = choose_geometry()
    print(
        f"[*] Stage 2 geometry={best_label} | k={k} | "
        f"Val_MCC={score['Val_MCC']:.4f} | features={'+'.join(features)}"
    )

    pipeline = FusionBatchPipeline(
        base_dir=BASE_DIR,
        config={
            "batch_size": 256,
            "epochs": 30,
            "lr": 1e-4,
            "weight_decay": 1e-3,
            "lr_factor": 0.5,
            "lr_patience": 3,
            "early_stop_patience": 6,
            "num_workers": 0,
        },
        datasets=DATASETS,
        models_space=MODELS_SPACE,
        pooling_strategies=["mean"],
        experiments=EXPERIMENTS,
        explainability={"enable_shap": False, "enable_lime": False},
        fm_profile_json=f"{BASE_DIR}/profiling/fm_profiling.json",
        geom_profile_json=f"{BASE_DIR}/profiling/geom_profiling.json",
        geometry_dir=f"{BASE_DIR}/{geometry_root}/{{context}}",
        geom_feature_names=GEOM_V2_FEATURE_COLUMNS,
        batch_run_dir=BATCH_RUN_DIR,
    )

    pipeline.run_fixed_geometry_ablation(
        geometry_configs=[
            {
                "label": f"{best_label}_architecture",
                "features": features,
            }
        ],
        experiment_names=[experiment["name"] for experiment in EXPERIMENTS],
        resume=True,
        run_explainability=False,
    )
    print(f"[DONE] Stage 2 metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
