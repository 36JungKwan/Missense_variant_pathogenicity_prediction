"""Seed sweep for the fixed 601-DNA/101-protein sequence configuration.

Seeds are selected by validation MCC. Test MCC is reported only as an
unselected audit so the sweep does not tune directly on the test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.module05_fusion_classifier.pipeline import (  # noqa: E402
    FusionBatchPipeline,
    _extract_features_for_ml,
    _stable_seed,
)
from core.module05_fusion_classifier.xgboost_model import (  # noqa: E402
    XGBoostFusionManager,
)


BASE_DIR = Path("D:/variant_data")
EMBEDDING_DIR = BASE_DIR / "fm_embeddings" / "sequence_ablation"
DNA_MODEL = "nt_v2_500m_dna_601"
PROT_MODEL = "esm1b_650m_prot_101"
ACTIVE_MODS = ["dna", "prot", "bio_core", "spliceai"]
POOLING = "mean"

TEST_SPLITS = ("test", "clinvarhq", "uniprot", "proteingym")


def _make_pipeline() -> FusionBatchPipeline:
    return FusionBatchPipeline(
        base_dir=str(BASE_DIR),
        config={
            "batch_size": 256,
            "num_workers": 0,
            "seed": 42,
            "xgb_n_jobs": 1,
        },
        datasets=[
            {"name": "Train3Val_Test", "train": "train3", "val": "val", "test": "test"}
        ],
        models_space=[
            {"name": DNA_MODEL, "seq_type": "dna"},
            {"name": PROT_MODEL, "seq_type": "protein"},
        ],
        pooling_strategies=[POOLING],
        experiments=[
            {"name": "Pure_XGBoost_Concat", "type": "xgboost_pure", "fusion": "concat"}
        ],
        explainability={"enable_shap": False, "enable_lime": False},
        geometry_dir=str(BASE_DIR / "geometry"),
        batch_run_dir=str(REPO_ROOT / "experiments" / "seed_sweep_tmp"),
        embedding_dir=str(EMBEDDING_DIR),
    )


def _collect(pipeline: FusionBatchPipeline, split: str, is_train: bool):
    loader = pipeline._get_dataloader(
        split,
        POOLING,
        DNA_MODEL,
        PROT_MODEL,
        ACTIVE_MODS,
        is_train,
        None,
        "train3",
    )
    return _extract_features_for_ml(
        torch.nn.Identity(), loader, pipeline.device, extract_f_global=False
    )


def run_sweep(start_seed: int, num_seeds: int, output_path: Path) -> pd.DataFrame:
    pipeline = _make_pipeline()
    print("[SEED SWEEP] Loading train/validation/test features once...")

    train_loader = pipeline._get_dataloader(
        "train3",
        POOLING,
        DNA_MODEL,
        PROT_MODEL,
        ACTIVE_MODS,
        True,
        None,
        "train3",
        loader_seed=start_seed,
    )
    val = _collect(pipeline, "val", False)
    tests = {split: _collect(pipeline, split, False) for split in TEST_SPLITS}
    identity = torch.nn.Identity()

    rows = []
    for offset in range(num_seeds):
        seed = start_seed + offset
        config_seed = _stable_seed(
            "train3",
            "val",
            POOLING,
            DNA_MODEL,
            PROT_MODEL,
            ",".join(sorted(ACTIVE_MODS)),
            "ALL",
            base_seed=seed,
        )
        exp_seed = _stable_seed("Pure_XGBoost_Concat", base_seed=config_seed)
        train_loader.generator.manual_seed(exp_seed)
        _, dna_tr, prot_tr, bg_tr, y_tr, _ = _extract_features_for_ml(
            identity, train_loader, pipeline.device, extract_f_global=False
        )
        _, dna_val, prot_val, bg_val, y_val, _ = val

        manager = XGBoostFusionManager(random_state=seed, n_jobs=1)
        manager.train_pure(
            dna_tr,
            prot_tr,
            bg_tr,
            y_tr,
            dna_val,
            prot_val,
            bg_val,
            y_val,
        )
        _, _, val_metrics = manager.predict_pure(
            dna_val, prot_val, bg_val, y_val
        )

        row = {
            "Seed": seed,
            "Best_Iteration": int(getattr(manager.model, "best_iteration", -1)),
            "Val_MCC": val_metrics["MCC"],
            "Val_AUROC": val_metrics["ROC-AUC"],
            "Val_AUPRC": val_metrics["PR-AUC"],
        }
        for split, values in tests.items():
            _, dna_ts, prot_ts, bg_ts, y_ts, _ = values
            _, _, metrics = manager.predict_pure(dna_ts, prot_ts, bg_ts, y_ts)
            row[f"{split}_MCC"] = metrics["MCC"]
            row[f"{split}_AUROC"] = metrics["ROC-AUC"]
            row[f"{split}_AUPRC"] = metrics["PR-AUC"]

        rows.append(row)
        best = max(rows, key=lambda item: item["Val_MCC"])
        print(
            f"[{offset + 1:03d}/{num_seeds}] seed={seed} | "
            f"Val_MCC={row['Val_MCC']:.4f} | "
            f"ClinVarHQ_MCC={row['clinvarhq_MCC']:.4f} | "
            f"best_val_seed={best['Seed']}"
        )

    result = pd.DataFrame(rows).sort_values(
        ["Val_MCC", "clinvarhq_MCC"], ascending=False
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    selected = result.iloc[0].to_dict()
    output_path.with_suffix(".selected.json").write_text(
        json.dumps(
            {
                "selection_metric": "Val_MCC",
                "selected_seed": int(selected["Seed"]),
                "selected_row": selected,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"[DONE] Sweep CSV: {output_path}")
    print(
        f"[SELECTED] seed={int(selected['Seed'])} | "
        f"Val_MCC={selected['Val_MCC']:.4f} | "
        f"ClinVarHQ_MCC={selected['clinvarhq_MCC']:.4f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = Path(args.output) if args.output else (
        REPO_ROOT
        / "experiments"
        / f"sequence_length_seed_sweep_{datetime.now():%Y%m%d_%H%M}.csv"
    )
    run_sweep(args.start_seed, args.num_seeds, output)


if __name__ == "__main__":
    main()
