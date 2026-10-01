"""Run the 5 x 5 DNA/protein sequence-length ablation.

The script extracts ten reusable embedding groups (five DNA and five protein
lengths), then trains exactly one fixed Module 5 configuration for each of the
25 length pairs. Existing legacy embeddings are never reused because this run
has its own embedding namespace.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.module02_fm_embedding_extraction.extractor import FeatureExtractor
from core.module02_fm_embedding_extraction.models import BioModelManager
from core.module05_fusion_classifier.pipeline import FusionBatchPipeline


BASE_DIR = Path("D:/variant_data")
EMBEDDING_DIR = BASE_DIR / "fm_embeddings" / "sequence_ablation"

DNA_LENGTHS = (301, 601, 1001, 2001, 10001)
PROTEIN_LENGTHS = (31, 101, 201, 501, 1001)

INPUT_FILES = {
    "train1": BASE_DIR / "train1_full_seq_final.parquet",
    "train2": BASE_DIR / "train2_full_seq_final.parquet",
    "train3": BASE_DIR / "train3_full_seq_final.parquet",
    "val": BASE_DIR / "val_full_seq_final.parquet",
    "test": BASE_DIR / "test_full_seq_after_vep_final.parquet",
    "clinvarhq": BASE_DIR / "clinvarhq_full_seq_after_vep_final.parquet",
    "uniprot": BASE_DIR / "uniprot_full_seq_after_vep_final.parquet",
    "proteingym": BASE_DIR / "proteingym_full_seq_after_vep_final.parquet",
}

TEST_SPLITS = ("test", "clinvarhq", "uniprot", "proteingym")

DNA_MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
PROTEIN_MODEL_ID = "facebook/esm1b_t33_650M_UR50S"


def _sequence_columns(seq_type: str, length: int) -> tuple[str, str]:
    if seq_type == "dna":
        return f"ref_seq_{length}", f"alt_seq_{length}"
    if seq_type == "protein":
        return f"prot_ref_seq_{length}", f"prot_alt_seq_{length}"
    raise ValueError(f"Unknown sequence type: {seq_type}")


def _model_name(seq_type: str, length: int) -> str:
    if seq_type == "dna":
        return f"nt_v2_500m_dna_{length}"
    return f"esm1b_650m_prot_{length}"


def _validate_inputs() -> None:
    print("=" * 80)
    print("[PRECHECK] Sequence-length ablation inputs")
    print(f"[PRECHECK] DNA lengths: {DNA_LENGTHS}")
    print(f"[PRECHECK] Protein lengths: {PROTEIN_LENGTHS}")

    for split, path in INPUT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing input parquet for {split}: {path}")

        required = ["Variant_ID"]
        for length in DNA_LENGTHS:
            required.extend(_sequence_columns("dna", length))
        for length in PROTEIN_LENGTHS:
            required.extend(_sequence_columns("protein", length))

        df = pd.read_parquet(path, columns=required)
        for seq_type, lengths in (("dna", DNA_LENGTHS), ("protein", PROTEIN_LENGTHS)):
            for length in lengths:
                ref_col, alt_col = _sequence_columns(seq_type, length)
                ref = df[ref_col].astype("string")
                alt = df[alt_col].astype("string")
                center = length // 2
                bad = (
                    ref.isna()
                    | alt.isna()
                    | (ref.str.len() != length)
                    | (alt.str.len() != length)
                    | (ref.str.get(center) == alt.str.get(center))
                )
                n_bad = int(bad.sum())
                print(f"[Precheck:{split}:{seq_type}:{length}] rows={len(df)} | errors={n_bad}")
                if n_bad:
                    sample = df.loc[bad, ["Variant_ID", ref_col, alt_col]].head(3)
                    print(sample.to_string(index=False))
                    raise RuntimeError(
                        f"Sequence precheck failed for {split}, {seq_type}, length={length}"
                    )

    print("[PRECHECK] PASS: all sequence columns have the expected center mutation.\n")


def _finite_max_length(tokenizer) -> int | None:
    max_length = getattr(tokenizer, "model_max_length", None)
    if max_length is None or max_length > 10**8:
        return None
    return int(max_length)


def _validate_token_capacity(tokenizer, input_path: Path, ref_col: str) -> None:
    sample = pd.read_parquet(input_path, columns=[ref_col]).iloc[0][ref_col]
    encoded = tokenizer(str(sample), add_special_tokens=True, truncation=False)
    token_count = len(encoded["input_ids"])
    max_length = _finite_max_length(tokenizer)
    print(
        f"[Token precheck] {ref_col}: tokens={token_count} | "
        f"model_max_length={max_length or 'unbounded'}"
    )
    if max_length is not None and token_count > max_length:
        raise RuntimeError(
            f"Tokenizer would truncate {ref_col}: {token_count} tokens > {max_length}."
        )


def _manifest_path(output_prefix: Path) -> Path:
    return Path(f"{output_prefix}.sequence_manifest.json")


def _expected_outputs(output_prefix: Path) -> list[Path]:
    return [Path(f"{output_prefix}_{pooling}.pt") for pooling in ("cls", "center", "mean")]


def _is_current(output_prefix: Path, manifest: dict) -> bool:
    if not all(path.exists() for path in _expected_outputs(output_prefix)):
        return False
    manifest_path = _manifest_path(output_prefix)
    if not manifest_path.exists():
        return False
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    except (OSError, json.JSONDecodeError):
        return False


def _write_manifest(output_prefix: Path, manifest: dict) -> None:
    _manifest_path(output_prefix).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _batch_size(seq_type: str, length: int) -> int:
    if seq_type == "dna":
        return {301: 32, 601: 32, 1001: 16, 2001: 8, 10001: 1}[length]
    return {31: 32, 101: 32, 201: 16, 501: 8, 1001: 4}[length]


def _extract_family(seq_type: str, lengths: tuple[int, ...], device: str) -> None:
    if seq_type == "dna":
        model_name = "nt_v2_500m"
        model_id = DNA_MODEL_ID
    else:
        model_name = "esm1b_650m"
        model_id = PROTEIN_MODEL_ID

    model_type = "mlm"
    model_manager = None
    extractor = None
    try:
        print("=" * 80)
        print(f"[EXTRACTION] {seq_type.upper()} family: {model_id}")
        model_manager = BioModelManager(model_id=model_id, model_type=model_type, device=device)
        extractor = FeatureExtractor(model_manager=model_manager)

        first_split = "train1"
        for length in lengths:
            ref_col, alt_col = _sequence_columns(seq_type, length)
            _validate_token_capacity(
                model_manager.tokenizer,
                INPUT_FILES[first_split],
                ref_col,
            )

            run_model_name = _model_name(seq_type, length)
            manifest = {
                "model_name": run_model_name,
                "model_id": model_id,
                "seq_type": seq_type,
                "length": length,
                "ref_col": ref_col,
                "alt_col": alt_col,
                "poolings": ["cls", "center", "mean"],
            }

            for split, input_path in INPUT_FILES.items():
                split_dir = EMBEDDING_DIR / split
                split_dir.mkdir(parents=True, exist_ok=True)
                output_prefix = split_dir / run_model_name
                if _is_current(output_prefix, manifest):
                    print(f"[RESUME] Skip {split}/{run_model_name}: outputs are current")
                    continue

                print(
                    f"[EXTRACT] split={split} | model={run_model_name} | "
                    f"columns={ref_col}/{alt_col} | batch={_batch_size(seq_type, length)}"
                )
                extractor.run_extraction(
                    parquet_path=str(input_path),
                    seq_type=seq_type,
                    batch_size=_batch_size(seq_type, length),
                    output_prefix=str(output_prefix),
                    ref_col=ref_col,
                    alt_col=alt_col,
                )
                _write_manifest(output_prefix, manifest)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
    finally:
        del extractor
        del model_manager
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _dataset_configs() -> list[dict]:
    configs = []
    for train_split in ("train1", "train2", "train3"):
        for test_split in TEST_SPLITS:
            configs.append({
                "name": f"{train_split.title()}Val_{test_split.title()}",
                "train": train_split,
                "val": "val",
                "test": test_split,
            })
    return configs


def _run_training(batch_run_dir: Path, resume: bool) -> dict:
    dna_models = [
        {"name": _model_name("dna", length), "seq_type": "dna"}
        for length in DNA_LENGTHS
    ]
    protein_models = [
        {"name": _model_name("protein", length), "seq_type": "protein"}
        for length in PROTEIN_LENGTHS
    ]
    models_space = [*dna_models, *protein_models]
    sequence_pairs = [
        (dna["name"], protein["name"])
        for dna in dna_models
        for protein in protein_models
    ]

    pipeline = FusionBatchPipeline(
        base_dir=str(BASE_DIR),
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
        datasets=_dataset_configs(),
        models_space=models_space,
        pooling_strategies=["mean"],
        experiments=[{
            "name": "Pure_XGBoost_Concat",
            "type": "xgboost_pure",
            "fusion": "concat",
        }],
        explainability={"enable_shap": False, "enable_lime": False},
        geometry_dir=str(BASE_DIR / "geometry"),
        geom_feature_names=None,
        batch_run_dir=str(batch_run_dir),
        embedding_dir=str(EMBEDDING_DIR),
    )
    return pipeline.run_sequence_ablation(
        sequence_pairs=sequence_pairs,
        resume=resume,
        experiment_name="Pure_XGBoost_Concat",
        pooling="mean",
        active_mods=["dna", "prot", "bio_core", "spliceai"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--batch-run-dir", default=None)
    args = parser.parse_args()

    resume = not args.no_resume
    if not args.skip_extraction:
        _validate_inputs()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _extract_family("dna", DNA_LENGTHS, device)
        _extract_family("protein", PROTEIN_LENGTHS, device)

    if not args.skip_training:
        batch_run_dir = Path(args.batch_run_dir) if args.batch_run_dir else (
            REPO_ROOT / "experiments" / f"sequence_ablation_{datetime.now():%Y%m%d_%H%M}"
        )
        result = _run_training(batch_run_dir, resume=resume)
        print("=" * 80)
        print("[DONE] Sequence-length ablation completed")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
