"""Resumable token-level FM extraction shared by Module 6 architectures."""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.module02_fm_embedding_extraction.extractor import BioSequenceDataset
from core.module02_fm_embedding_extraction.models import BioModelManager


TOKEN_CACHE_SCHEMA_VERSION = 1


def _atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f"{path.stem}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    except Exception as save_error:
        if os.path.exists(temporary_name):
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                raise RuntimeError(
                    f"Failed to save token-cache shard to {path}: {save_error}. "
                    f"Could not remove incomplete temporary file {temporary_name}: {cleanup_error}"
                ) from save_error
        raise


def _tokenize(tokenizer, sequences, device, model_id, *, strict_context=False, model=None):
    options = {
        "return_tensors": "pt", "padding": True,
        "truncation": not strict_context,
    }
    if getattr(tokenizer, "is_fast", False):
        options["return_offsets_mapping"] = True
    try:
        encoded = tokenizer(sequences, **options)
    except (NotImplementedError, ValueError, TypeError):
        options.pop("return_offsets_mapping", None)
        encoded = tokenizer(sequences, **options)
    encoded = encoded.to(device)

    if strict_context:
        config = getattr(model, "config", None)
        capacity = next((
            int(getattr(config, key)) for key in (
                "max_position_embeddings", "n_positions", "max_seq_len",
                "max_sequence_length",
            ) if getattr(config, key, None) is not None
            and 0 < int(getattr(config, key)) < 10_000_000
        ), None) if config is not None else None
        token_count = int(encoded["attention_mask"].sum(dim=1).max().item()) if "attention_mask" in encoded else int(encoded["input_ids"].shape[1])
        if capacity is not None and token_count > capacity:
            raise ValueError(
                f"Requested full sequence needs {token_count} FM tokens, exceeding "
                f"{model_id} context capacity {capacity}. Refusing to truncate a "
                "sequence-length ablation; use a supported sequence length/model."
            )

    # Keep the same NTv3 padding convention as Module 2 so hidden token axes
    # remain compatible with its convolutional downsampling implementation.
    if "ntv3" in str(model_id).lower():
        input_ids = encoded["input_ids"]
        downsamples = int(getattr(getattr(encoded, "config", None), "num_downsamples", 7))
        multiple = 2 ** downsamples
        target_len = int(math.ceil(input_ids.shape[1] / multiple) * multiple)
        pad_len = target_len - input_ids.shape[1]
        if pad_len:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            encoded["input_ids"] = torch.nn.functional.pad(input_ids, (0, pad_len), value=pad_id)
            old_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
            encoded["attention_mask"] = torch.nn.functional.pad(old_mask, (0, pad_len), value=0)
            if "offset_mapping" in encoded:
                encoded["offset_mapping"] = torch.nn.functional.pad(
                    encoded["offset_mapping"], (0, 0, 0, pad_len), value=0
                )
    if "attention_mask" not in encoded:
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
    return encoded


def _hidden_states(model, encoded, use_cuda_autocast=False):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda_autocast):
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
        )
    hidden = getattr(outputs, "hidden_states", None)
    if hidden is not None:
        return hidden[-1]
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is not None:
        return last_hidden
    if isinstance(outputs, dict):
        if outputs.get("hidden_states") is not None:
            return outputs["hidden_states"][-1]
        if outputs.get("last_hidden_state") is not None:
            return outputs["last_hidden_state"]
    if isinstance(outputs, (tuple, list)) and outputs and torch.is_tensor(outputs[0]):
        return outputs[0]
    raise RuntimeError("Foundation model output did not contain token hidden states.")


def _offsets_or_empty(encoded, batch_size):
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return torch.empty((batch_size, 0, 2), dtype=torch.int32)
    return offsets.detach().to(device="cpu", dtype=torch.int32)


def _variant_indices(manager, encoded_ref, encoded_alt, seq_type, char_positions):
    ref_indices = manager.get_variant_token_index(
        encoded_ref,
        seq_type=seq_type,
        target_char_positions=char_positions,
        tokenized_alt=encoded_alt,
    )
    # Map the alternate allele independently: tokenization can shift or split
    # the mutation differently even though the character position is shared.
    alt_indices = manager.get_variant_token_index(
        encoded_alt,
        seq_type=seq_type,
        target_char_positions=char_positions,
        tokenized_alt=encoded_ref,
    )
    return ref_indices.detach().cpu().to(torch.int32), alt_indices.detach().cpu().to(torch.int32)


def _prepare_model_sequences(sequences, char_positions, model_config):
    """Apply an optional symmetric crop while keeping the variant at the center."""
    target_length = model_config.get("center_crop_length")
    if target_length is None:
        return [str(value) for value in sequences], [int(value) for value in char_positions]

    target_length = int(target_length)
    expected_source_length = model_config.get("source_sequence_length")
    if target_length <= 0 or target_length % 2 != 1:
        raise ValueError(f"Center crop length must be a positive odd value, got {target_length}.")
    output_sequences, output_positions = [], []
    for row, (sequence, position) in enumerate(zip(sequences, char_positions)):
        sequence = str(sequence)
        position = int(position)
        if expected_source_length is not None and len(sequence) != int(expected_source_length):
            raise ValueError(
                f"Expected source sequence length {expected_source_length}, got {len(sequence)} "
                f"for model cache {model_config['name']}."
            )
        if len(sequence) < target_length:
            raise ValueError(
                f"Cannot center-crop row {row}: source length {len(sequence)} < target {target_length}."
            )
        if position != len(sequence) // 2:
            raise ValueError(
                f"Expected the variant at the source-sequence center, got position {position} "
                f"for length {len(sequence)}. Refusing a shifted DNA window."
            )
        start = position - target_length // 2
        stop = start + target_length
        cropped = sequence[start:stop]
        cropped_position = position - start
        if len(cropped) != target_length or cropped_position != target_length // 2:
            raise RuntimeError(
                f"Center crop failed to preserve the variant at index {target_length // 2}."
            )
        output_sequences.append(cropped)
        output_positions.append(cropped_position)
    if len(output_sequences) != len(sequences):
        raise ValueError("Sequence and variant-position batch sizes differ during center crop.")
    return output_sequences, output_positions


def _expected_manifest(model_config, split_config, batch_size):
    source = Path(split_config["input_path"])
    stat = source.stat()
    expected = {
        "schema_version": TOKEN_CACHE_SCHEMA_VERSION,
        "model_name": model_config["name"],
        "model_id": model_config["model_id"],
        "model_type": model_config["model_type"],
        "seq_type": model_config["seq_type"],
        "sequence_length": int(model_config["sequence_length"]),
        "ref_col": model_config["ref_col"],
        "alt_col": model_config["alt_col"],
        "input_path": str(source.resolve()),
        "input_size_bytes": int(stat.st_size),
        "input_mtime_ns": int(stat.st_mtime_ns),
        "batch_size": int(batch_size),
        "hidden_layer": "last",
        "storage_dtype": "float16",
        "row_order": "source_parquet_order",
    }
    if model_config.get("center_crop_length") is not None:
        expected.update({
            "source_sequence_length": int(model_config["source_sequence_length"]),
            "center_crop_length": int(model_config["center_crop_length"]),
            "crop_anchor": "variant_center",
        })
    return expected


def _estimate_split_storage(dataset, manager, model_config, *, strict_context=False):
    hidden_size = getattr(manager.model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(manager.model.config, "d_model", None)
    if not hidden_size or len(dataset) == 0:
        return None
    sample_count = min(32, len(dataset))
    refs = dataset.df[dataset.ref_col].astype(str).head(sample_count).tolist()
    alts = dataset.df[dataset.alt_col].astype(str).head(sample_count).tolist()
    source_positions = dataset.variant_char_positions[:sample_count].tolist()
    refs, positions = _prepare_model_sequences(refs, source_positions, model_config)
    alts, alt_positions = _prepare_model_sequences(alts, source_positions, model_config)
    if alt_positions != positions:
        raise RuntimeError("Ref/alt variant centers diverged during model sequence preparation.")
    ref_tokens = _tokenize(
        manager.tokenizer, refs, manager.device, manager.model_id,
        strict_context=strict_context, model=manager.model,
    )
    alt_tokens = _tokenize(
        manager.tokenizer, alts, manager.device, manager.model_id,
        strict_context=strict_context, model=manager.model,
    )
    max_sample_tokens = int(
        ref_tokens["attention_mask"].sum(dim=1).max().item()
        + alt_tokens["attention_mask"].sum(dim=1).max().item()
    )
    # Two alleles × final hidden state × float16, plus a modest headroom for
    # serialized metadata/format overhead. This is a sample-based estimate.
    estimate = int(len(dataset) * max_sample_tokens * int(hidden_size) * 2 * 1.08)
    del ref_tokens, alt_tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return estimate


def _check_manifest(folder, expected):
    manifest_path = Path(folder) / "manifest.json"
    shard_files = list(Path(folder).glob("shard_*.pt"))
    if not manifest_path.exists():
        if shard_files:
            raise RuntimeError(
                f"Token cache contains shard files but no manifest: {folder}. "
                "Move the incomplete cache aside before restarting."
            )
        manifest = {**expected, "status": "in_progress", "completed_shards": []}
        _atomic_json(manifest, manifest_path)
        return manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    relocated_input = False
    if "input_path" in mismatches:
        cached_path = Path(manifest.get("input_path", ""))
        expected_path = Path(expected["input_path"])
        input_metadata_matches = all(
            manifest.get(key) == expected.get(key)
            for key in ("input_size_bytes", "input_mtime_ns")
        )
        if (
            cached_path.name.casefold() == expected_path.name.casefold()
            and input_metadata_matches
        ):
            mismatches.remove("input_path")
            relocated_input = True
    if mismatches:
        raise ValueError(
            f"Token cache manifest does not match the requested extraction at {folder}; "
            f"changed fields: {mismatches}. Use a new cache directory for a different setup."
        )
    if relocated_input:
        manifest["input_path"] = expected["input_path"]
        _atomic_json(manifest, manifest_path)
    return manifest


def _extract_split(manager, model_config, split_config, output_folder, batch_size, *, strict_context=False):
    dataset = BioSequenceDataset(
        parquet_path=split_config["input_path"],
        seq_type=model_config["seq_type"],
        ref_col=model_config["ref_col"],
        alt_col=model_config["alt_col"],
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    expected = _expected_manifest(model_config, split_config, batch_size)
    expected["resolved_model_commit"] = getattr(manager.model.config, "_commit_hash", None)
    expected["tokenizer_class"] = type(manager.tokenizer).__name__
    expected["tokenizer_vocab_size"] = int(getattr(manager.tokenizer, "vocab_size", 0) or 0)
    expected["special_token_ids"] = {
        name: getattr(manager.tokenizer, name, None)
        for name in ("bos_token_id", "eos_token_id", "cls_token_id", "sep_token_id", "pad_token_id")
    }
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    estimated_storage_bytes = _estimate_split_storage(
        dataset, manager, model_config, strict_context=strict_context
    )
    if estimated_storage_bytes is not None:
        free_bytes = shutil.disk_usage(folder).free
        print(
            f"[Token cache estimate] {model_config['name']}/{split_config['name']}: "
            f"~{estimated_storage_bytes / (1024 ** 3):.2f} GiB; "
            f"free space at cache root: {free_bytes / (1024 ** 3):.2f} GiB"
        )
    manifest = _check_manifest(folder, expected)
    completed = set(manifest.get("completed_shards", []))
    expected_shards = math.ceil(len(dataset) / int(batch_size))
    completed_indices = {
        index for index in range(expected_shards)
        if (
            f"shard_{index:06d}.pt" in completed
            and (folder / f"shard_{index:06d}.pt").is_file()
        )
    }
    if estimated_storage_bytes is not None and len(dataset):
        completed_samples = sum(
            min(int(batch_size), len(dataset) - index * int(batch_size))
            for index in completed_indices
        )
        remaining_samples = len(dataset) - completed_samples
        estimated_remaining_bytes = int(
            estimated_storage_bytes * remaining_samples / len(dataset)
        )
        free_bytes = shutil.disk_usage(folder).free
        if estimated_remaining_bytes > free_bytes:
            raise OSError(
                f"Insufficient free disk space to resume token-cache extraction for "
                f"{model_config['name']}/{split_config['name']}: approximately "
                f"{estimated_remaining_bytes / (1024 ** 3):.2f} GiB is still required, "
                f"but only {free_bytes / (1024 ** 3):.2f} GiB is available at {folder}. "
                "Free disk space and rerun this cell; completed shards will be reused."
            )

    extraction_start = time.perf_counter()
    elapsed_before_run = float(manifest.get("elapsed_seconds", 0.0))
    for shard_index, batch in enumerate(tqdm(loader, desc=f"Token embeddings: {folder.name}")):
        shard_name = f"shard_{shard_index:06d}.pt"
        shard_path = folder / shard_name
        if shard_name in completed and shard_path.exists():
            continue

        source_positions = [int(value) for value in batch["variant_char_pos"]]
        ref_sequences, char_positions = _prepare_model_sequences(
            batch["ref_seq"], source_positions, model_config
        )
        alt_sequences, alt_char_positions = _prepare_model_sequences(
            batch["alt_seq"], source_positions, model_config
        )
        if alt_char_positions != char_positions:
            raise RuntimeError("Ref/alt variant centers diverged during model sequence preparation.")
        device = manager.device
        encoded_ref = _tokenize(
            manager.tokenizer, ref_sequences, device, manager.model_id,
            strict_context=strict_context, model=manager.model,
        )
        encoded_alt = _tokenize(
            manager.tokenizer, alt_sequences, device, manager.model_id,
            strict_context=strict_context, model=manager.model,
        )
        use_cuda_autocast = (
            device.startswith("cuda")
            and "nucleotide-transformer-v2" in manager.model_id.lower()
        )
        indices_ref, indices_alt = _variant_indices(
            manager, encoded_ref, encoded_alt, model_config["seq_type"], char_positions
        )
        # Copy the tokenization metadata to CPU, then process alleles one at a
        # time so both full hidden-state tensors are never resident on GPU.
        input_ids_ref = encoded_ref["input_ids"].detach().cpu().to(torch.int32)
        mask_ref = encoded_ref["attention_mask"].detach().cpu().to(torch.bool)
        offsets_ref = _offsets_or_empty(encoded_ref, len(char_positions))
        with torch.inference_mode():
            hidden_ref = _hidden_states(
                manager.model, encoded_ref, use_cuda_autocast=use_cuda_autocast
            )
        hidden_ref = hidden_ref.detach().to(device="cpu", dtype=torch.float16)
        hidden_size = int(hidden_ref.shape[-1])
        del encoded_ref

        input_ids_alt = encoded_alt["input_ids"].detach().cpu().to(torch.int32)
        mask_alt = encoded_alt["attention_mask"].detach().cpu().to(torch.bool)
        offsets_alt = _offsets_or_empty(encoded_alt, len(char_positions))
        with torch.inference_mode():
            hidden_alt = _hidden_states(
                manager.model, encoded_alt, use_cuda_autocast=use_cuda_autocast
            )
        hidden_alt = hidden_alt.detach().to(device="cpu", dtype=torch.float16)
        del encoded_alt
        shard = {
            "schema_version": TOKEN_CACHE_SCHEMA_VERSION,
            "row_start": int(shard_index * int(batch_size)),
            "row_stop": int(shard_index * int(batch_size) + len(char_positions)),
            "variant_ids": [str(value) for value in batch["name"]],
            "variant_char_pos": torch.tensor(char_positions, dtype=torch.int32),
            "token_hidden_ref": hidden_ref,
            "token_hidden_alt": hidden_alt,
            "attention_mask_ref": mask_ref,
            "attention_mask_alt": mask_alt,
            "variant_token_index_ref": indices_ref,
            "variant_token_index_alt": indices_alt,
            "input_ids_ref": input_ids_ref,
            "input_ids_alt": input_ids_alt,
            "offset_mapping_ref": offsets_ref,
            "offset_mapping_alt": offsets_alt,
        }
        _atomic_torch_save(shard, shard_path)
        completed.add(shard_name)
        manifest.update({
            "status": "in_progress",
            "completed_shards": sorted(completed),
            "n_samples": len(dataset),
            "estimated_storage_bytes": estimated_storage_bytes,
            "hidden_size": hidden_size,
            "max_ref_tokens": int(hidden_ref.shape[1]),
            "max_alt_tokens": int(hidden_alt.shape[1]),
            "storage_bytes": sum(path.stat().st_size for path in folder.glob("shard_*.pt")),
            "elapsed_seconds": elapsed_before_run + (time.perf_counter() - extraction_start),
        })
        _atomic_json(manifest, folder / "manifest.json")
        del shard, hidden_ref, hidden_alt

    missing = [
        f"shard_{index:06d}.pt" for index in range(expected_shards)
        if not (folder / f"shard_{index:06d}.pt").exists()
    ]
    if missing:
        raise RuntimeError(f"Token extraction incomplete for {folder}; missing shards: {missing[:5]}")
    manifest.update({
        "status": "complete",
        "n_samples": len(dataset),
        "estimated_storage_bytes": estimated_storage_bytes,
        "n_shards": expected_shards,
        "storage_bytes": sum(path.stat().st_size for path in folder.glob("shard_*.pt")),
        "elapsed_seconds": elapsed_before_run + (time.perf_counter() - extraction_start),
        "completed_shards": [f"shard_{index:06d}.pt" for index in range(expected_shards)],
    })
    _atomic_json(manifest, folder / "manifest.json")
    return manifest


def run_token_embedding_extraction(
    model_configs, split_configs, cache_root, batch_run_dir, device="cuda",
    batch_sizes=None, strict_context=False,
):
    """Extract/resume token caches and persist their locations in the active batch.

    Args use the same model and split naming convention as Module 2. Artifacts live
    in a shared cache; the active Module 6 batch stores an index pointing to them.
    """
    cache_root = Path(cache_root)
    batch_dir = Path(batch_run_dir)
    index_path = batch_dir / "module06_architecture_experiments" / "token_embedding_cache_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            index = {}
    else:
        index = {}
    existing_cache_root = index.get("cache_root")
    requested_cache_root = str(cache_root.resolve())
    if existing_cache_root and str(Path(existing_cache_root).resolve()) != requested_cache_root:
        raise ValueError(
            "This batch token-cache index already points to a different cache root: "
            f"{existing_cache_root}. Use a new batch run to change TOKEN_EMBEDDING_CACHE_ROOT."
        )
    index.update({"schema_version": TOKEN_CACHE_SCHEMA_VERSION, "cache_root": requested_cache_root, "models": index.get("models", {})})
    _atomic_json(index, index_path)

    batch_sizes = batch_sizes or {}
    for model_config in model_configs:
        model_name = str(model_config["name"])
        model_batch_size = int(batch_sizes.get(model_name, 2))
        model_index = index["models"].setdefault(model_name, {})
        manager = BioModelManager(
            model_id=model_config["model_id"],
            model_type=model_config["model_type"],
            device=str(device),
        )
        if str(manager.device).startswith("cpu"):
            manager.model.float()
        manager.model.eval()
        try:
            for split_config in split_configs:
                split_name = str(split_config["name"])
                output_folder = cache_root / model_name / split_name
                model_index[split_name] = {
                    "status": "in_progress",
                    "path": str(output_folder.resolve()),
                    "manifest": str((output_folder / "manifest.json").resolve()),
                }
                index["updated_at"] = pd.Timestamp.utcnow().isoformat()
                _atomic_json(index, index_path)
                manifest = _extract_split(
                    manager, model_config, split_config, output_folder, model_batch_size,
                    strict_context=strict_context,
                )
                model_index[split_name] = {
                    "status": manifest["status"],
                    "path": str(output_folder.resolve()),
                    "manifest": str((output_folder / "manifest.json").resolve()),
                    "n_samples": manifest["n_samples"],
                    "n_shards": manifest["n_shards"],
                    "hidden_size": manifest.get("hidden_size"),
                    "storage_bytes": manifest.get("storage_bytes"),
                    "estimated_storage_bytes": manifest.get("estimated_storage_bytes"),
                    "elapsed_seconds": manifest.get("elapsed_seconds"),
                }
                index["updated_at"] = pd.Timestamp.utcnow().isoformat()
                _atomic_json(index, index_path)
                print(
                    f"[Token cache] {model_name}/{split_name}: "
                    f"{manifest['n_samples']} samples, {manifest['n_shards']} shards -> {output_folder}"
                )
        finally:
            del manager
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    index["index_path"] = str(index_path.resolve())
    _atomic_json(index, index_path)
    return {"index": index, "index_path": str(index_path.resolve())}


def load_token_embedding_cache_index(batch_run_dir):
    path = Path(batch_run_dir) / "module06_architecture_experiments" / "token_embedding_cache_index.json"
    if not path.exists():
        raise FileNotFoundError(f"Module 6 token cache index not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_token_embedding_cache(index, model_name, split_name):
    """Return a completed model/split cache folder from the persisted batch index."""
    try:
        entry = index["models"][model_name][split_name]
    except KeyError as error:
        raise KeyError(f"No token cache is indexed for {model_name}/{split_name}") from error
    if entry.get("status") != "complete":
        raise RuntimeError(
            f"Token cache is not complete for {model_name}/{split_name}: {entry.get('status')}"
        )
    folder = Path(entry["path"])
    manifest = Path(entry["manifest"])
    if not folder.is_dir() or not manifest.is_file():
        raise FileNotFoundError(
            f"Indexed token cache files are missing for {model_name}/{split_name}: {folder}"
        )
    return folder
