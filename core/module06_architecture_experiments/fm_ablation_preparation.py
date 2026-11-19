"""Preflight and resumable token-cache preparation for FM ablations.

This module only prepares token-level representations. Model evaluation and
pooling-ablation extraction are deliberately separate notebook stages.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


BASE_MODELS = (
    {"name": "nt_v1_500m", "model_id": "InstaDeepAI/nucleotide-transformer-500m-human-ref", "seq_type": "dna", "model_type": "mlm", "batch_size": 32},
    {"name": "nt_v3_650m", "model_id": "InstaDeepAI/NTv3_650M_pre", "seq_type": "dna", "model_type": "mlm", "batch_size": 16},
    {"name": "nt_v2_500m", "model_id": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", "seq_type": "dna", "model_type": "mlm", "batch_size": 16},
    {"name": "esm1b_650m", "model_id": "facebook/esm1b_t33_650M_UR50S", "seq_type": "protein", "model_type": "mlm", "batch_size": 16},
    {"name": "esm1v_650m", "model_id": "facebook/esm1v_t33_650M_UR90S_1", "seq_type": "protein", "model_type": "mlm", "batch_size": 16},
    {"name": "esm2_650m", "model_id": "facebook/esm2_t33_650M_UR50D", "seq_type": "protein", "model_type": "mlm", "batch_size": 16},
    {"name": "esmc_600m", "model_id": "biohub/ESMC-600M-hf", "seq_type": "protein", "model_type": "mlm", "batch_size": 16},
)

SPLIT_FILES = {
    "train3": "train3_full_seq_final.parquet",
    "val": "val_full_seq_final.parquet",
    "test": "test_full_seq_after_vep_final.parquet",
    "clinvarhq": "clinvarhq_full_seq_after_vep_final.parquet",
    "proteingym": "proteingym_full_seq_after_vep_final.parquet",
    "uniprot": "uniprot_full_seq_after_vep_final.parquet",
}


def build_model_configs(dna_length: int = 1001, protein_length: int = 501):
    """Return the seven FM configs for the requested paired sequence lengths."""
    configs = []
    for base in BASE_MODELS:
        length = int(dna_length if base["seq_type"] == "dna" else protein_length)
        seq_prefix = "dna" if base["seq_type"] == "dna" else "prot"
        ref_col = f"ref_seq_{length}" if seq_prefix == "dna" else f"prot_ref_seq_{length}"
        alt_col = f"alt_seq_{length}" if seq_prefix == "dna" else f"prot_alt_seq_{length}"
        configs.append({
            **base,
            "name": f"{base['name']}_{seq_prefix}_{length}",
            "sequence_length": length,
            "ref_col": ref_col,
            "alt_col": alt_col,
        })
    return configs


def build_split_configs(base_data_dir):
    """Use the same full-sequence source parquets as Notebook 13's token cache."""
    root = Path(base_data_dir)
    missing = [str(root / filename) for filename in SPLIT_FILES.values() if not (root / filename).is_file()]
    if missing:
        raise FileNotFoundError("FM ablation parquet files are missing:\n" + "\n".join(missing))
    return [{"name": split, "input_path": str(root / filename)} for split, filename in SPLIT_FILES.items()]


def _input_matches(manifest, input_path):
    try:
        source = Path(input_path).resolve()
        source_stat = source.stat()
        cached_path = Path(manifest["input_path"])
        metadata_matches = (
            int(manifest["input_size_bytes"]) == source_stat.st_size
            and int(manifest["input_mtime_ns"]) == source_stat.st_mtime_ns
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False
    same_path = cached_path.resolve() == source
    relocated_path = cached_path.name.casefold() == source.name.casefold()
    return metadata_matches and (same_path or relocated_path)


def inspect_token_cache(cache_root, model_configs, split_configs):
    """Report complete/partial/missing token caches without loading model weights."""
    root = Path(cache_root)
    rows = []
    for model in model_configs:
        for split in split_configs:
            folder = root / model["name"] / split["name"]
            manifest_path = folder / "manifest.json"
            manifest = {}
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    manifest = {}
            rows.append({
                "Model": model["name"],
                "Split": split["name"],
                "Status": manifest.get("status", "missing"),
                "Shards": len(list(folder.glob("shard_*.pt"))) if folder.is_dir() else 0,
                "Path": str(folder),
                "Model_ID_Matches": manifest.get("model_id") == model["model_id"] if manifest else False,
                "Sequence_Length_Matches": manifest.get("sequence_length") == model["sequence_length"] if manifest else False,
                "Input_Matches": _input_matches(manifest, split["input_path"]) if manifest else False,
            })
    return rows


def extract_missing_token_caches(
    *, model_configs, split_configs, cache_root, batch_run_dir, device="cuda",
    strict_context=False,
):
    """Resume missing shards through Module 6's shared token-cache extractor."""
    from core.module06_architecture_experiments.token_embedding_cache import (
        _atomic_json,
        run_token_embedding_extraction,
    )

    root = Path(cache_root)
    missing_models = []
    for model in model_configs:
        missing_split = False
        for split in split_configs:
            folder = root / model["name"] / split["name"]
            manifest_path = folder / "manifest.json"
            if not manifest_path.is_file():
                missing_split = True
                break
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                missing_split = True
                break
            expected_shards = int(manifest.get("n_shards", 0) or 0)
            shards_complete = expected_shards > 0 and all(
                (folder / f"shard_{index:06d}.pt").is_file()
                for index in range(expected_shards)
            )
            if (
                manifest.get("model_id") != model["model_id"]
                or int(manifest.get("sequence_length", -1)) != int(model["sequence_length"])
                or not _input_matches(manifest, split["input_path"])
                or manifest.get("status") != "complete"
                or not shards_complete
            ):
                missing_split = True
                break
        if missing_split:
            missing_models.append(model)

    result = {"index": None}
    if missing_models:
        result = run_token_embedding_extraction(
            model_configs=missing_models,
            split_configs=split_configs,
            cache_root=cache_root,
            batch_run_dir=batch_run_dir,
            device=device,
            batch_sizes={model["name"]: int(model["batch_size"]) for model in missing_models},
            strict_context=strict_context,
        )

    # Produce one complete index for the requested model set, including caches
    # that predated this batch. Module 6's loader needs both model entries.
    models_index = {}
    for model in model_configs:
        split_index = {}
        for split in split_configs:
            folder = root / model["name"] / split["name"]
            manifest_path = folder / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not _input_matches(manifest, split["input_path"]):
                raise ValueError(
                    f"Token cache input does not match the current parquet for "
                    f"{model['name']}/{split['name']}: {manifest.get('input_path')} "
                    f"vs {Path(split['input_path']).resolve()}"
                )
            input_path = str(Path(split["input_path"]).resolve())
            if manifest.get("input_path") != input_path:
                manifest["input_path"] = input_path
                _atomic_json(manifest, manifest_path)
            split_index[split["name"]] = {
                "status": manifest["status"],
                "path": str(folder.resolve()),
                "manifest": str(manifest_path.resolve()),
                "n_samples": manifest.get("n_samples"),
                "n_shards": manifest.get("n_shards"),
                "hidden_size": manifest.get("hidden_size"),
                "storage_bytes": manifest.get("storage_bytes"),
            }
        models_index[model["name"]] = split_index
    index_path = Path(batch_run_dir) / "module06_architecture_experiments" / "fm_ablation_token_cache_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "cache_root": str(root.resolve()), "models": models_index,
               "index_path": str(index_path.resolve())}
    fd, temporary = tempfile.mkstemp(prefix="fm_token_index_", suffix=".tmp", dir=index_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        os.replace(temporary, index_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"status": "complete", "extracted_models": [m["name"] for m in missing_models],
            "index": payload, "index_path": str(index_path.resolve())}
