"""Stage 1: isolate LID and sweep neighborhood size k."""

import os

from core.module05_fusion_classifier.dataset import GEOM_V2_FEATURE_COLUMNS
from core.module05_fusion_classifier.pipeline import FusionBatchPipeline


BASE_DIR = "D:/variant_data"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BATCH_RUN_DIR = os.path.join(REPO_ROOT, "experiments", "batch_run_20260930_lid_ablation")
K_VALUES = [8, 16, 32, 64]

BASE_GEOM = [
    "dna_LVD_L2",
    "dna_LVD_Relative",
    "prot_LLR",
    "prot_LVD_Cosine",
    "prot_LVD_Relative",
]

MODELS_SPACE = [
    {"name": "nt_v2_500m", "seq_type": "dna"},
    {"name": "esm1b_650m", "seq_type": "protein"},
]
EXPERIMENTS = [
    {"name": "Pure_XGBoost_Concat", "type": "xgboost_pure", "fusion": "concat"},
]
DATASETS = [
    {"name": "Train3Val_Test", "train": "train3", "val": "val", "test": "test"},
    {"name": "Train3Val_ClinVarHQ", "train": "train3", "val": "val", "test": "clinvarhq"},
    {"name": "Train3Val_UniProt", "train": "train3", "val": "val", "test": "uniprot"},
    {"name": "Train3Val_ProteinGym", "train": "train3", "val": "val", "test": "proteingym"},
]


def build_pipeline(geometry_root):
    return FusionBatchPipeline(
        base_dir=BASE_DIR,
        config={"batch_size": 256, "num_workers": 0},
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


def main():
    # Control: no LID, evaluated once using the existing k=32 geometry root.
    control_pipeline = build_pipeline("geometry_v2")
    control_pipeline.run_fixed_geometry_ablation(
        geometry_configs=[
            {
                "label": "bio_core_dna_prot_spliceai_geom_lid_control",
                "features": BASE_GEOM,
            }
        ],
        experiment_names=["Pure_XGBoost_Concat"],
        resume=True,
        run_explainability=False,
    )

    for k in K_VALUES:
        geometry_root = "geometry_v2" if k == 32 else f"geometry_v2_k{k}"
        pipeline = build_pipeline(geometry_root)
        configs = [
            {
                "label": f"bio_core_dna_prot_spliceai_geom_lid_dna_k{k}",
                "features": [*BASE_GEOM, "dna_LID"],
            },
            {
                "label": f"bio_core_dna_prot_spliceai_geom_lid_prot_k{k}",
                "features": [*BASE_GEOM, "prot_LID"],
            },
            {
                "label": f"bio_core_dna_prot_spliceai_geom_lid_both_k{k}",
                "features": [*BASE_GEOM, "dna_LID", "prot_LID"],
            },
        ]
        pipeline.run_fixed_geometry_ablation(
            geometry_configs=configs,
            experiment_names=["Pure_XGBoost_Concat"],
            resume=True,
            run_explainability=False,
        )

    print(f"[DONE] Stage 1 metrics: {BATCH_RUN_DIR}/global_leaderboard_metrics.csv")


if __name__ == "__main__":
    main()
