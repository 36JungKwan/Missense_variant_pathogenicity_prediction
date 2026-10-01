"""Backfill validation metrics into an existing Module 5 leaderboard.

This is intended for completed pure-XGBoost geometry ablations whose test
metrics already exist but whose validation metrics were not written yet.
"""

import gc
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(".."))

from core.module05_fusion_classifier.dataset import GEOM_V2_FEATURE_COLUMNS
from core.module05_fusion_classifier.evaluator_profiler import FusionEvaluatorProfiler
from core.module05_fusion_classifier.pipeline import (
    FusionBatchPipeline,
    _extract_features_for_ml,
)
from core.module05_fusion_classifier.xgboost_model import XGBoostFusionManager


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BATCH_RUN_DIR = os.path.join(REPO_ROOT, "experiments", "batch_run_20260929_1740")
BASE_DIR = "D:/variant_data"
GEOMETRY_DIR = f"{BASE_DIR}/geometry_v2/{{context}}"
METRICS_PATH = os.path.join(BATCH_RUN_DIR, "global_leaderboard_metrics.csv")

DNA_MODEL = "nt_v2_500m"
PROT_MODEL = "esm1b_650m"
POOLING = "mean"
ABLATION = "bio_core_dna_prot_spliceai_geom"
NETWORK = "Pure_XGBoost_Concat"
ACTIVE_MODALITIES = ["dna", "prot", "bio_core", "spliceai", "geom"]


def main():
    metrics = pd.read_csv(METRICS_PATH)
    mask = (
        (metrics["Ablation"] == ABLATION)
        & (metrics["DNA_Model"] == DNA_MODEL)
        & (metrics["Prot_Model"] == PROT_MODEL)
        & (metrics["Pooling"] == POOLING)
        & (metrics["Network"] == NETWORK)
    )
    target = metrics.loc[mask].copy()
    if target.empty:
        raise RuntimeError("Khong tim thay dong geometry ablation phu hop trong leaderboard")

    config = {"batch_size": 256, "num_workers": 0}
    pipeline = FusionBatchPipeline(
        base_dir=BASE_DIR,
        config=config,
        datasets=[{"name": "Train3Val_Test", "train": "train3", "val": "val", "test": "test"}],
        models_space=[
            {"name": DNA_MODEL, "seq_type": "dna"},
            {"name": PROT_MODEL, "seq_type": "protein"},
        ],
        pooling_strategies=[POOLING],
        experiments=[{"name": NETWORK, "type": "xgboost_pure", "fusion": "concat"}],
        explainability={"enable_shap": False, "enable_lime": False},
        fm_profile_json=f"{BASE_DIR}/profiling/fm_profiling.json",
        geom_profile_json=f"{BASE_DIR}/profiling/geom_profiling.json",
        geometry_dir=GEOMETRY_DIR,
        geom_feature_names=GEOM_V2_FEATURE_COLUMNS,
        batch_run_dir=BATCH_RUN_DIR,
    )

    unique_geom = target["Geom_Features"].dropna().drop_duplicates().tolist()
    print(f"[*] Backfill validation: {len(unique_geom)} geometry subsets")

    # Load the validation backbone once. Every subset uses the same DNA,
    # protein, bio and full geometry rows; only the selected geometry columns
    # change between checkpoints.
    full_val_loader = pipeline._get_dataloader(
        "val",
        POOLING,
        DNA_MODEL,
        PROT_MODEL,
        ACTIVE_MODALITIES,
        False,
        GEOM_V2_FEATURE_COLUMNS,
        "train3",
    )
    _, dna_val, prot_val, bg_val, y_val, _ = _extract_features_for_ml(
        torch.nn.Identity(), full_val_loader, pipeline.device, False
    )
    bio_dim = len(full_val_loader.dataset.bio_cols)
    base_bio = bg_val[:, :bio_dim]
    base_geom = bg_val[:, bio_dim:]
    geom_index = {name: index for index, name in enumerate(GEOM_V2_FEATURE_COLUMNS)}

    cache_dataset = {"train": "train3", "val": "val"}
    for index, geom_key in enumerate(unique_geom, start=1):
        geom_features = [name for name in str(geom_key).split("+") if name]
        selected_indices = [geom_index[name] for name in geom_features]
        bg_subset = pd.concat(
            [
                pd.DataFrame(base_bio),
                pd.DataFrame(base_geom[:, selected_indices]),
            ],
            axis=1,
        ).to_numpy()

        cache_dir = pipeline._get_train_cache_ckpt_dir(
            cache_dataset,
            POOLING,
            ABLATION,
            DNA_MODEL,
            PROT_MODEL,
            NETWORK,
            geom_key,
        )
        model_path = os.path.join(cache_dir, "best_model_xgboost.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Thieu XGBoost checkpoint: {model_path}")

        xgb_manager = XGBoostFusionManager()
        xgb_manager.load_model(cache_dir, prefix="best_model")
        y_probs, y_preds, _ = xgb_manager.predict_pure(dna_val, prot_val, bg_subset)
        val_metrics = FusionEvaluatorProfiler.compute_metrics(y_val, y_probs, y_preds)

        row_mask = mask & (metrics["Geom_Features"] == geom_key)
        for name, value in val_metrics.items():
            metrics.loc[row_mask, f"Val_{name}"] = value

        if index % 25 == 0 or index == len(unique_geom):
            metrics.to_csv(METRICS_PATH, index=False)
            print(f"[{index:04d}/{len(unique_geom)}] {geom_key} | Val_MCC={val_metrics['MCC']:.4f}")

        del bg_subset, y_probs, y_preds, xgb_manager
        gc.collect()

    metrics.to_csv(METRICS_PATH, index=False)
    print(f"[DONE] Updated: {METRICS_PATH} | rows={len(metrics)}")
    print(f"[DONE] Added columns: {', '.join(f'Val_{name}' for name in val_metrics)}")


if __name__ == "__main__":
    main()
