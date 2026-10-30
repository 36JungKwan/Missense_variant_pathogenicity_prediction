"""Build missing embeddings and run the fixed UniProt-augmented experiment.

This is intentionally separate from notebooks 00 and 01. It reuses compatible
center embeddings, computes only missing variant embeddings, fits the existing
Module 4 bio normalizer on the three augmented training splits, then runs the
same fixed Module 5 XGBoost configuration for each training split.
"""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.module02_fm_embedding_extraction.extractor import BioSequenceDataset, FeatureExtractor
from core.module02_fm_embedding_extraction.models import BioModelManager
from core.module04_geo_bio_context_normalization.normalizer import AdvancedFeatureNormalizer
from core.module05_fusion_classifier.pipeline import FusionBatchPipeline
from core.module05_fusion_classifier.sota_benchmark import SOTA_REQUIRED_COLUMNS


BASE = Path("D:/variant_data")
OUT = BASE / "uniprot_augmented_pipeline"
INPUTS = {
    "train1": BASE / "train1_full_seq_final_spliceai_with_uniprot.parquet",
    "train2": BASE / "train2_full_seq_final_spliceai_with_uniprot.parquet",
    "train3": BASE / "train3_full_seq_final_spliceai_with_uniprot.parquet",
    "val": BASE / "val_full_seq_final_spliceai_with_uniprot.parquet",
    # This is the post-SOTA-filter file. The pre-mapping backup is excluded.
    "test": BASE / "test_full_seq_after_vep_final_spliceai_with_uniprot.parquet",
    "clinvarhq": BASE / "clinvarhq_full_seq_after_vep_final_spliceai.parquet",
    "proteingym": BASE / "proteingym_full_seq_after_vep_final_spliceai.parquet",
    "uniprot": BASE / "uniprot_full_seq_after_vep_final_spliceai.parquet",
}
BASE_INPUTS = {
    "train1": ("train1", BASE / "train1_full_seq_final.parquet"),
    "train2": ("train2", BASE / "train2_full_seq_final.parquet"),
    "train3": ("train3", BASE / "train3_full_seq_final.parquet"),
    "val": ("val", BASE / "val_full_seq_final.parquet"),
    "test": ("test", BASE / "test_full_seq_after_vep_final.parquet"),
}
UNIPROT_SOURCE = ("uniprot", INPUTS["uniprot"])

MODELS = {
    "nt_v1_500m_dna_1001": {
        "seq_type": "dna",
        "model_id": "InstaDeepAI/nucleotide-transformer-500m-human-ref",
        "ref": "ref_seq_1001",
        "alt": "alt_seq_1001",
        "batch_size": 16,
    },
    "esm2_650m_prot_501": {
        "seq_type": "protein",
        "model_id": "facebook/esm2_t33_650M_UR50D",
        "ref": "prot_ref_seq_501",
        "alt": "prot_alt_seq_501",
        # Center-only inference uses two forward passes and retries at half
        # batch size on CUDA OOM.
        "batch_size": 64,
    },
}
SOURCE_EMBED_DIR = BASE / "fm_embeddings" / "sequence_ablation"
EMBED_DIR = OUT / "fm_embeddings"
BIO_DIR = OUT / "processed_parquet"
NORM_ARTIFACTS = OUT / "normalization_artifacts"
TRAIN_SPLITS = ("train1", "train2", "train3")
EVAL_SPLITS = ("val", "test", "clinvarhq", "proteingym", "uniprot")


def _load_pt(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding artifact: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _read_sequence_index(path: Path, ref_col: str, alt_col: str) -> dict[str, tuple[str, str]]:
    df = pd.read_parquet(path, columns=["Variant_ID", ref_col, alt_col])
    return {
        str(row.Variant_ID): (str(getattr(row, ref_col)), str(getattr(row, alt_col)))
        for row in df.itertuples(index=False)
        if pd.notna(getattr(row, ref_col)) and pd.notna(getattr(row, alt_col))
    }


def _extract_missing(model_name: str, input_path: Path, ids: list[str], temp_dir: Path) -> dict:
    cfg = MODELS[model_name]
    temp_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(input_path)
    subset = source.loc[source["Variant_ID"].astype(str).isin(ids),
                        ["Variant_ID", cfg["ref"], cfg["alt"]]].copy()
    subset_path = temp_dir / f"{model_name}_missing.parquet"
    subset.to_parquet(subset_path, index=False)
    manager = BioModelManager(model_id=cfg["model_id"], model_type="mlm",
                              device="cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor(manager)
    try:
        dataset = BioSequenceDataset(str(subset_path), cfg["seq_type"], cfg["ref"], cfg["alt"])
        all_ref, all_alt = [], []

        def infer_indices(indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
            batch = [dataset[index] for index in indices]
            names = [row["name"] for row in batch]
            refs = [row["ref_seq"] for row in batch]
            alts = [row["alt_seq"] for row in batch]
            char_positions = [row["variant_char_pos"] for row in batch]
            try:
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.float16,
                    enabled=manager.device.startswith("cuda"),
                ):
                    inputs_ref = extractor._tokenize_batch(refs)
                    inputs_alt = extractor._tokenize_batch(alts)
                    positions = manager.get_variant_token_index(
                        inputs_ref, cfg["seq_type"], target_char_positions=char_positions,
                        tokenized_alt=inputs_alt,
                    )
                    outputs_ref = manager.model(
                        input_ids=inputs_ref["input_ids"],
                        attention_mask=inputs_ref.get("attention_mask"),
                        output_hidden_states=True,
                    )
                    hidden_ref = extractor._get_hidden_states_safe(outputs_ref)
                    vectors_ref = torch.stack([
                        hidden_ref[i, positions[i]] for i in range(len(names))
                    ]).to(torch.float16).cpu()
                    del outputs_ref, hidden_ref
                    outputs_alt = manager.model(
                        input_ids=inputs_alt["input_ids"],
                        attention_mask=inputs_alt.get("attention_mask"),
                        output_hidden_states=True,
                    )
                    hidden_alt = extractor._get_hidden_states_safe(outputs_alt)
                    vectors_alt = torch.stack([
                        hidden_alt[i, positions[i]] for i in range(len(names))
                    ]).to(torch.float16).cpu()
                    del outputs_alt, hidden_alt
                return vectors_ref, vectors_alt
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or len(indices) <= 1:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                midpoint = len(indices) // 2
                ref_left, alt_left = infer_indices(indices[:midpoint])
                ref_right, alt_right = infer_indices(indices[midpoint:])
                return torch.cat((ref_left, ref_right)), torch.cat((alt_left, alt_right))

        for start in range(0, len(dataset), cfg["batch_size"]):
            indices = list(range(start, min(start + cfg["batch_size"], len(dataset))))
            vectors_ref, vectors_alt = infer_indices(indices)
            all_ref.append(vectors_ref)
            all_alt.append(vectors_alt)
            if (start // cfg["batch_size"] + 1) % 100 == 0:
                print(f"[CENTER] {model_name}: {min(start + len(indices), len(dataset))}/{len(dataset)}")
    finally:
        del extractor
        del manager
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return {
        "metadata": subset["Variant_ID"].astype(str).tolist(),
        "E_ref": torch.cat(all_ref),
        "E_alt": torch.cat(all_alt),
    }


def _build_embedding(split: str, input_path: Path, model_name: str, temp_dir: Path) -> None:
    cfg = MODELS[model_name]
    target_dir = EMBED_DIR / split
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{model_name}_center.pt"
    df = pd.read_parquet(input_path, columns=["Variant_ID", cfg["ref"], cfg["alt"]])
    df["Variant_ID"] = df["Variant_ID"].astype(str)
    if df["Variant_ID"].duplicated().any():
        raise ValueError(f"Duplicate Variant_ID in {input_path}")
    expected = dict(zip(df["Variant_ID"], zip(df[cfg["ref"]].astype(str), df[cfg["alt"]].astype(str))))
    if target_path.exists():
        existing = _load_pt(target_path)
        if list(map(str, existing.get("metadata", []))) == list(expected):
            print(f"[RESUME] {split}/{model_name}: complete center embedding already exists")
            return
        raise ValueError(f"Existing embedding artifact has a different Variant_ID order: {target_path}")

    # Candidate artifacts are the original split and the prior UniProt set.
    candidates: list[tuple[Path, dict[str, tuple[str, str]]]] = []
    if split in BASE_INPUTS:
        base_name, base_path = BASE_INPUTS[split]
        if base_path.exists():
            candidates.append((SOURCE_EMBED_DIR / base_name / f"{model_name}_center.pt",
                               _read_sequence_index(base_path, cfg["ref"], cfg["alt"])))
    candidates.append((SOURCE_EMBED_DIR / "uniprot" / f"{model_name}_center.pt",
                       _read_sequence_index(UNIPROT_SOURCE[1], cfg["ref"], cfg["alt"])))

    id_to_record: dict[str, tuple[dict, int]] = {}
    for pt_path, seq_index in candidates:
        if not pt_path.exists():
            continue
        artifact = _load_pt(pt_path)
        for index, vid in enumerate(artifact["metadata"]):
            vid = str(vid)
            if vid in expected and expected[vid] == seq_index.get(vid) and vid not in id_to_record:
                id_to_record[vid] = (artifact, index)

    missing = [vid for vid in expected if vid not in id_to_record]
    extracted = None
    if missing:
        print(f"[EMBED] {split}/{model_name}: extracting {len(missing)} missing variants")
        extracted = _extract_missing(model_name, input_path, missing, temp_dir)
        for index, vid in enumerate(extracted["metadata"]):
            id_to_record[str(vid)] = (extracted, index)

    if len(id_to_record) != len(expected):
        unresolved = sorted(set(expected).difference(id_to_record))
        raise RuntimeError(f"Missing embeddings for {split}/{model_name}: {unresolved[:8]}")

    metadata = list(expected)
    result = {"metadata": metadata}
    for key in ("E_ref", "E_alt"):
        result[key] = torch.stack([artifact[key][index] for vid in metadata
                                   for artifact, index in [id_to_record[vid]]])
    torch.save(result, target_path)
    print(f"[EMBED] wrote {target_path} ({len(metadata)} variants)")


def _prepare_bio() -> None:
    BIO_DIR.mkdir(parents=True, exist_ok=True)
    NORM_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    normalizer = AdvancedFeatureNormalizer(epsilon=1e-8)
    train_frames = [pd.read_parquet(INPUTS[split]) for split in TRAIN_SPLITS]
    merged = pd.concat(train_frames, ignore_index=True)
    # The existing Module 4 workflow concatenates the three training splits
    # directly; an ID may legitimately occur in more than one split and should
    # retain the same weighting during normalizer fitting.
    train_merged_norm = normalizer.fit_transform_bio(merged, str(NORM_ARTIFACTS))
    train_merged_norm.to_parquet(BIO_DIR / "train_aug_merged_normalized.parquet", index=False)

    for split, frame in zip(TRAIN_SPLITS, train_frames):
        normalizer.transform_bio(frame, str(NORM_ARTIFACTS)).to_parquet(
            BIO_DIR / f"{split}_normalized.parquet", index=False)
    for split in (*EVAL_SPLITS,):
        frame = pd.read_parquet(INPUTS[split])
        normalizer.transform_bio(frame, str(NORM_ARTIFACTS)).to_parquet(
            BIO_DIR / f"{split}_normalized.parquet", index=False)


def _copy_external_embeddings() -> None:
    for split in ("clinvarhq", "proteingym", "uniprot"):
        target = EMBED_DIR / split
        target.mkdir(parents=True, exist_ok=True)
        for model_name in MODELS:
            src = SOURCE_EMBED_DIR / split / f"{model_name}_center.pt"
            dst = target / src.name
            if not src.exists():
                raise FileNotFoundError(f"Missing reusable evaluation embedding: {src}")
            shutil.copy2(src, dst)


def _validate_files() -> None:
    for split, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing input {split}: {path}")
    test_cols = pd.read_parquet(INPUTS["test"], columns=list(SOTA_REQUIRED_COLUMNS))
    incomplete = test_cols[list(SOTA_REQUIRED_COLUMNS)].isna().any(axis=1)
    if incomplete.any():
        raise ValueError(
            "Current augmented test is not fully SOTA-complete: "
            f"{int(incomplete.sum())} rows still have a missing required SOTA value"
        )
    for split, path in INPUTS.items():
        cols = ["Variant_ID"]
        if split in (*TRAIN_SPLITS, "val", "test"):
            cols.extend(c for cfg in MODELS.values() for c in (cfg["ref"], cfg["alt"]))
        frame = pd.read_parquet(path, columns=list(dict.fromkeys(cols)))
        if frame["Variant_ID"].isna().any() or frame["Variant_ID"].astype(str).duplicated().any():
            raise ValueError(f"Invalid Variant_IDs in {split}")
        if split in (*TRAIN_SPLITS, "val", "test"):
            for cfg in MODELS.values():
                for col in (cfg["ref"], cfg["alt"]):
                    lengths = frame[col].astype("string").str.len()
                    expected_len = 1001 if cfg["seq_type"] == "dna" else 501
                    if lengths.isna().any() or not lengths.eq(expected_len).all():
                        raise ValueError(f"Bad sequence length in {split}/{col}; expected {expected_len}")


def _run_training(batch_dir: Path) -> dict:
    models_space = [
        {"name": "nt_v1_500m_dna_1001", "seq_type": "dna"},
        {"name": "esm2_650m_prot_501", "seq_type": "protein"},
    ]
    datasets = []
    for train_split in TRAIN_SPLITS:
        for test_split in EVAL_SPLITS:
            datasets.append({
                "name": f"{train_split.title()}Val_{test_split.title()}",
                "train": train_split,
                "val": "val",
                "test": test_split,
            })
    pipeline = FusionBatchPipeline(
        base_dir=str(OUT),
        config={
            "batch_size": 256, "epochs": 30, "lr": 1e-4,
            "weight_decay": 1e-3, "lr_factor": 0.5,
            "lr_patience": 3, "early_stop_patience": 6,
            "num_workers": 0, "seed": 42, "xgb_n_jobs": 1,
        },
        datasets=datasets,
        models_space=models_space,
        pooling_strategies=["center"],
        experiments=[{"name": "Pure_XGBoost_Concat", "type": "xgboost_pure", "fusion": "concat"}],
        explainability={"enable_shap": False, "enable_lime": False, "random_state": 42},
        batch_run_dir=str(batch_dir),
        embedding_dir=str(EMBED_DIR),
    )
    result = pipeline.run_sequence_ablation(
        sequence_pairs=[("nt_v1_500m_dna_1001", "esm2_650m_prot_501")],
        resume=True,
        experiment_name="Pure_XGBoost_Concat",
        pooling="center",
        active_mods=["dna", "prot", "bio_core", "spliceai"],
    )
    _write_baseline_comparison(Path(result["metrics_path"]), batch_dir)
    return result


def _write_baseline_comparison(metrics_path: Path, batch_dir: Path) -> None:
    baseline_path = ROOT / "experiments" / "batch_run_20261002_2044" / "global_leaderboard_metrics.csv"
    if not baseline_path.exists() or not metrics_path.exists():
        print(f"[COMPARE] Baseline or current metrics missing; skipped comparison: {baseline_path}")
        return
    current = pd.read_csv(metrics_path)
    baseline = pd.read_csv(baseline_path)
    config = {
        "Pooling": "center",
        "Ablation": "bio_core_dna_prot_spliceai",
        "DNA_Model": "nt_v1_500m_dna_1001",
        "Prot_Model": "esm2_650m_prot_501",
        "Network": "Pure_XGBoost_Concat",
    }
    for col, value in config.items():
        current = current.loc[current[col].astype(str).eq(value)]
        baseline = baseline.loc[baseline[col].astype(str).eq(value)]
    baseline_dataset = {
        "Test": "Train3Val_Test",
        "Clinvarhq": "Train3Val_ClinVarHQ",
        "Proteingym": "Train3Val_ProteinGym",
        "Uniprot": "Train3Val_UniProt",
    }
    baseline = baseline.loc[baseline["Dataset"].isin(baseline_dataset.values())]
    base_cols = ["Dataset", "MCC", "AUROC", "AUPRC", "Val_MCC", "Val_AUPRC"]
    baseline = baseline[base_cols].rename(columns={
        "Dataset": "Baseline_Dataset",
        "MCC": "Baseline_MCC",
        "AUROC": "Baseline_AUROC",
        "AUPRC": "Baseline_AUPRC",
        "Val_MCC": "Baseline_Val_MCC",
        "Val_AUPRC": "Baseline_Val_AUPRC",
    })
    current = current.loc[current["Dataset"].astype(str).str.startswith("Train3Val_")].copy()
    current["Eval_split"] = current["Dataset"].astype(str).str.removeprefix("Train3Val_")
    current["Baseline_Dataset"] = current["Eval_split"].map(baseline_dataset)
    out = current.merge(baseline, on="Baseline_Dataset", how="left")
    for metric, base_metric in (("MCC", "Baseline_MCC"), ("AUROC", "Baseline_AUROC"),
                                ("AUPRC", "Baseline_AUPRC")):
        out[f"Delta_{metric}"] = pd.to_numeric(out[metric], errors="coerce") - pd.to_numeric(
            out[base_metric], errors="coerce")
    out["Comparison_note"] = (
        "Baseline is the existing Train3 run. Validation and augmented test cohorts differ; "
        "interpret deltas with cohort composition in mind."
    )
    batch_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(batch_dir / "train3_aug_vs_existing_baseline.csv", index=False)
    print(f"[COMPARE] Wrote {batch_dir / 'train3_aug_vs_existing_baseline.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-normalization", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--batch-dir", type=Path, default=OUT / "experiments" / "batch_run_augmented")
    args = parser.parse_args()

    _validate_files()
    if not args.skip_embedding:
        temp_dir = OUT / "_embedding_temp"
        for split in (*TRAIN_SPLITS, "val", "test"):
            for model_name in MODELS:
                _build_embedding(split, INPUTS[split], model_name, temp_dir)
        _copy_external_embeddings()
    if not args.skip_normalization:
        _prepare_bio()
    if not args.skip_training:
        result = _run_training(args.batch_dir)
        print("[DONE] augmented pipeline result:")
        print(result)


if __name__ == "__main__":
    main()
