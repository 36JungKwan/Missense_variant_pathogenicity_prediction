from __future__ import annotations

import json
import gc
import hashlib
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn

from core.module05v2_fusion_classifier.dataset import BIO_CORE_COLUMNS, SPLICEAI_COLUMNS
from core.module05v2_fusion_classifier.resume_utils import (
    ensure_stage_signature,
    save_progress_csv,
    upsert_record,
)
from core.module06_architecture_experiments.architecture_experiment_runner import (
    _architecture_loader,
    run_architecture_experiment,
)
from core.module06_architecture_experiments.cross_attention_3modal_experiment import (
    CrossAttentionThreeModalities,
    _CrossAttentionBlock,
)
from core.module06_architecture_experiments.fusion_classifier_benchmark import (
    _fit_fixed_tree,
    _load_tree_model,
    _predict_proba,
    _save_tree_model,
    _tree_node_count,
    _compute_metrics,
)
from core.module06_architecture_experiments.token_topk_experiment import _VariantCenteredWindow
from core.module06_architecture_experiments.window_token_cache import add_window_cache_to_loader
from core.module06_architecture_experiments.token_embedding_cache import (
    resolve_token_embedding_cache,
)
from core.module06_architecture_experiments.transformer_3modal_experiment import (
    _ScalarFeatureTokenizer,
)


GEOM_FEATURES = [
    f"{modality}_{feature}"
    for modality in ("dna", "prot")
    for feature in ("LVD_L2", "LVD_Cosine", "LID", "LVD_Relative")
]
PCA_COMPONENTS = 256
CLASSIFIERS = ("XGBoost", "CatBoost", "LightGBM")
PURE_ML_TOKEN_WINDOW_RADIUS = 16


def _resource_snapshot(label):
    """Report host RSS/available RAM and CUDA free memory when observable."""
    parts = [f"[Resources] {label}"]
    try:
        import psutil
        process = psutil.Process()
        info = process.memory_info()
        parts.append(f"process_RSS={info.rss / (1024 ** 3):.2f} GiB")
        peak = getattr(info, "peak_wset", None)
        if peak:
            parts.append(f"process_peak={peak / (1024 ** 3):.2f} GiB")
        parts.append(f"system_available={psutil.virtual_memory().available / (1024 ** 3):.2f} GiB")
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            parts.append(f"CUDA_free={free / (1024 ** 3):.2f}/{total / (1024 ** 3):.2f} GiB")
    except Exception:
        pass
    print(" | ".join(parts))


class _ResourceSampler:
    """Sample process/system/GPU memory during a tree-model fit."""

    def __init__(self, interval_seconds=0.25):
        import threading
        self._threading = threading
        self._stop = threading.Event()
        self._interval = float(interval_seconds)
        self.peak_rss = 0
        self.minimum_available = None
        self.peak_gpu_used = None

    def _sample_loop(self):
        try:
            import psutil
            process = psutil.Process()
        except Exception:
            process = None
        while not self._stop.is_set():
            if process is not None:
                try:
                    self.peak_rss = max(self.peak_rss, process.memory_info().rss)
                    available = psutil.virtual_memory().available
                    self.minimum_available = (
                        available if self.minimum_available is None
                        else min(self.minimum_available, available)
                    )
                except Exception:
                    pass
            try:
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    used = total - free
                    self.peak_gpu_used = max(self.peak_gpu_used or 0, used)
            except Exception:
                pass
            self._stop.wait(self._interval)

    def start(self):
        self._thread = self._threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        stats = {
            "Fit_Peak_RSS_GiB": round(self.peak_rss / (1024 ** 3), 3) if self.peak_rss else None,
            "Fit_Min_Available_RAM_GiB": (
                round(self.minimum_available / (1024 ** 3), 3)
                if self.minimum_available is not None else None
            ),
            "Fit_Peak_GPU_Device_Used_GiB": (
                round(self.peak_gpu_used / (1024 ** 3), 3)
                if self.peak_gpu_used is not None else None
            ),
        }
        return stats


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def _load_rows(path):
    path = Path(path)
    return pd.read_csv(path).to_dict("records") if path.is_file() else []


def _collect_feature_matrix(context, target, split_name, seed, *,
                            embedding_representation, is_train=False,
                            matrix_path=None):
    if embedding_representation != "window_radius16_ref_alt_concat_flatten":
        raise ValueError(
            "Pure ML feature collection requires the Cross-Attention-compatible "
            "radius-16 ref/alt token-window representation."
        )
    # The tree models consume the same local token inputs as Cross-Attention,
    # flattened in relative-position order. They do not use the pooled FM vectors.
    loader_context = dict(context)
    loader_context["token_only"] = True
    loader = _architecture_loader(
        loader_context["pipeline"], loader_context, target, split_name, "center",
        loader_context["DNA_MODEL"], loader_context["PROTEIN_MODEL"], is_train,
        "train3", seed, "ref_alt_concat",
    )
    loader = add_window_cache_to_loader(
        loader=loader,
        token_index=loader_context["token_embedding_cache_index"],
        split_name=split_name,
        dna_model_name=loader_context["DNA_MODEL"]["name"],
        protein_model_name=loader_context["PROTEIN_MODEL"]["name"],
        is_train=is_train,
        batch_size=loader_context.get("token_batch_size"),
        radius=PURE_ML_TOKEN_WINDOW_RADIUS,
    )
    dataset = loader.dataset
    bio_dim = int(dataset.bio_tensor.shape[1]) if getattr(dataset, "has_bio", False) else 0
    geom_dim = int(dataset.geom_tensor.shape[1]) if getattr(dataset, "has_geom", False) else 0
    width = 2 * PURE_ML_TOKEN_WINDOW_RADIUS + 1
    dna_hidden = int(dataset.token_hidden_sizes["dna"])
    protein_hidden = int(dataset.token_hidden_sizes["protein"])
    dna_dim = width * 2 * dna_hidden + width  # paired token features plus validity mask
    protein_dim = width * 2 * protein_hidden + width
    dims = {"dna": dna_dim, "protein": protein_dim, "bio": bio_dim, "geometry": geom_dim}
    offsets = {}
    cursor_dim = 0
    for name, dim in dims.items():
        offsets[name] = (cursor_dim, cursor_dim + dim)
        cursor_dim += dim
    if matrix_path is None:
        matrix = np.empty((len(dataset), cursor_dim), dtype=np.float32)
    else:
        matrix_path = Path(matrix_path)
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.lib.format.open_memmap(
            matrix_path, mode="w+", dtype=np.float32,
            shape=(len(dataset), cursor_dim),
        )
    labels = np.empty(len(dataset), dtype=np.int32)
    ids = []
    cursor = 0
    for batch in loader:
        count = int(batch["label"].shape[0])
        stop = cursor + count
        for modality, hidden_size in (("dna", dna_hidden), ("protein", protein_hidden)):
            tokens = batch[f"token_{modality}"].cpu().numpy()
            mask = batch[f"token_{modality}_mask"].cpu().numpy().astype(bool, copy=False)
            ref = tokens[:, 0] * mask[:, 0, :, None]
            alt = tokens[:, 1] * mask[:, 1, :, None]
            paired = np.concatenate((ref, alt), axis=-1).reshape(count, width * 2 * hidden_size)
            start, end = offsets[modality]
            raw_end = start + width * 2 * hidden_size
            matrix[cursor:stop, start:raw_end] = paired
            matrix[cursor:stop, raw_end:end] = np.logical_or(mask[:, 0], mask[:, 1]).astype(np.float32)
        for name in ("bio", "geometry"):
            start, end = offsets[name]
            if end > start:
                key = "bio_features" if name == "bio" else "geom_features"
                matrix[cursor:stop, start:end] = batch[key].cpu().numpy()
        labels[cursor:stop] = batch["label"].reshape(-1).cpu().numpy().astype(np.int32)
        ids.extend(str(value) for value in batch["variant_id"])
        cursor = stop
    if cursor != len(dataset):
        raise RuntimeError(f"Feature collection row count mismatch: {cursor} != {len(dataset)}")
    block_bytes = {
        name: (end - start) * len(dataset) * matrix.dtype.itemsize
        for name, (start, end) in offsets.items()
    }
    window_bytes = sum(
        tensor.numel() * tensor.element_size()
        for mapping_name in ("windows", "window_masks")
        for tensor in getattr(dataset, mapping_name, {}).values()
    )
    print(
        f"[Pure ML RAM] {split_name}: samples={len(dataset):,}, "
        f"features={matrix.shape[1]:,}; matrix={matrix.nbytes / (1024 ** 3):.2f} GiB "
        f"(DNA={block_bytes['dna'] / (1024 ** 3):.2f}, "
        f"protein={block_bytes['protein'] / (1024 ** 3):.2f}, "
        f"bio={block_bytes['bio'] / (1024 ** 3):.3f}, "
        f"geometry={block_bytes['geometry'] / (1024 ** 3):.3f}); "
        f"active token-window tensors={window_bytes / (1024 ** 3):.2f} GiB"
    )
    _resource_snapshot(f"after assembling {split_name}")
    if isinstance(matrix, np.memmap):
        matrix.flush()
    del loader
    return matrix, labels, ids, dims


def _feature_matrix_cache_signature(context, target, split_name):
    """Fingerprint the batch-local token inputs used to build a feature mmap."""
    token_index = context["token_embedding_cache_index"]
    source_caches = {}
    for modality, model_key in (("dna", "DNA_MODEL"), ("protein", "PROTEIN_MODEL")):
        model_name = str(context[model_key]["name"])
        folder = resolve_token_embedding_cache(token_index, model_name, split_name)
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = []
        for shard_path in sorted(folder.glob("shard_*.pt")):
            stat = shard_path.stat()
            shards.append((shard_path.name, stat.st_size, stat.st_mtime_ns))
        source_path = Path(manifest["input_path"])
        source_stat = source_path.stat()
        source_caches[modality] = {
            "model_name": model_name,
            "manifest": manifest,
            "manifest_stat": (manifest_path.stat().st_size, manifest_path.stat().st_mtime_ns),
            "source_stat": (source_stat.st_size, source_stat.st_mtime_ns),
            "shards": shards,
        }
    signature = {
        "version": 1,
        "batch_run_dir": str(context["pipeline"].batch_run_dir),
        "target": target,
        "split": str(split_name),
        "token_window_radius": PURE_ML_TOKEN_WINDOW_RADIUS,
        "representation": "radius16_ref_alt_concat_flatten_plus_bio_geometry_float32",
        "source_caches": source_caches,
    }
    encoded = json.dumps(signature, sort_keys=True, default=_json_safe).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_or_create_feature_memmap(context, target, split_name, seed, cache_dir):
    """Build once to a disk-backed .npy array, then reuse it without RAM copies."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / f"{_slug(split_name)}_features.npy"
    labels_path = cache_dir / f"{_slug(split_name)}_labels.npy"
    metadata_path = cache_dir / f"{_slug(split_name)}_manifest.json"
    signature = _feature_matrix_cache_signature(context, target, split_name)
    metadata = None
    matrix = labels = None
    cache_valid = False
    if metadata_path.is_file() and matrix_path.is_file() and labels_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("signature") == signature:
                matrix = np.load(matrix_path, mmap_mode="r")
                labels = np.load(labels_path, mmap_mode="r")
                if (
                    list(matrix.shape) == metadata.get("shape")
                    and matrix.dtype == np.float32
                    and list(labels.shape) == [matrix.shape[0]]
                ):
                    cache_valid = True
                    print(
                        f"[Pure ML mmap] Reusing {split_name}: {matrix_path} "
                        f"({matrix.nbytes / (1024 ** 3):.2f} GiB mapped, not eagerly copied)."
                    )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            metadata = None
        finally:
            if not cache_valid:
                for mapped in (matrix, labels):
                    if isinstance(mapped, np.memmap):
                        mapped._mmap.close()
                matrix = labels = None
    if cache_valid:
        return matrix, labels, metadata["dims"]

    print(f"[Pure ML mmap] Creating disk-backed feature matrix for {split_name}...")
    if metadata_path.exists():
        metadata_path.unlink()
    matrix, labels, _, dims = _collect_feature_matrix(
        context, target, split_name, seed, is_train=(split_name == "train3"),
        embedding_representation="window_radius16_ref_alt_concat_flatten",
        matrix_path=matrix_path,
    )
    labels_tmp = labels_path.with_name(labels_path.stem + ".tmp.npy")
    with labels_tmp.open("wb") as stream:
        np.save(stream, labels, allow_pickle=False)
    labels_tmp.replace(labels_path)
    metadata = {
        "signature": signature,
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "dims": dims,
    }
    metadata_tmp = metadata_path.with_suffix(".tmp.json")
    metadata_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metadata_tmp.replace(metadata_path)
    del matrix, labels
    matrix = np.load(matrix_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    print(
        f"[Pure ML mmap] Ready {split_name}: {matrix_path} "
        f"({matrix.nbytes / (1024 ** 3):.2f} GiB mapped; feature pages are OS-reclaimable)."
    )
    return matrix, labels, dims


def _apply_pca(train, other, dims, components, seed, n_threads=16):
    dna_dim, protein_dim = dims["dna"], dims["protein"]
    if not dna_dim or not protein_dim:
        raise ValueError("Pure ML PCA requires both DNA and protein embeddings.")
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=max(1, int(n_threads))):
        pca_dna = PCA(
            n_components=min(int(components), len(train), dna_dim),
            svd_solver="randomized", random_state=int(seed),
        ).fit(train[:, :dna_dim])
        pca_protein = PCA(
            n_components=min(int(components), len(train), protein_dim),
            svd_solver="randomized", random_state=int(seed),
        ).fit(train[:, dna_dim:dna_dim + protein_dim])

        def transform(matrix):
            dna = pca_dna.transform(matrix[:, :dna_dim])
            protein = pca_protein.transform(matrix[:, dna_dim:dna_dim + protein_dim])
            tabular = matrix[:, dna_dim + protein_dim:]
            return np.concatenate((dna, protein, tabular), axis=1).astype(np.float32, copy=False)

        return transform(train), [transform(matrix) for matrix in other], pca_dna, pca_protein


def _save_validation(path, row):
    rows = _load_rows(path)
    rows = upsert_record(rows, row, ["Seed", "Classifier", "PCA"])
    save_progress_csv(rows, path)
    return row


def _pure_ml_seed_complete(output_root, seed, datasets):
    for classifier in CLASSIFIERS:
        for label in ("NoPCA", "PCA"):
            if not _pure_ml_mode_seed_complete(output_root, seed, datasets, label):
                return False
    return True


def _pure_ml_mode_seed_complete(output_root, seed, datasets, label):
    """Check one PCA mode independently so a completed stage is not rebuilt."""
    for classifier in CLASSIFIERS:
        run_dir = Path(output_root) / classifier / label / f"seed_{seed}"
        model_path = run_dir / "best_model"
        model_path = model_path.with_suffix(
            ".json" if classifier == "XGBoost" else ".txt" if classifier == "LightGBM" else ".cbm"
        )
        val_rows = _load_rows(run_dir / "validation_metrics.csv")
        test_rows = _load_rows(run_dir / "test_metrics.csv")
        prediction_rows = _load_rows(run_dir / "test_predictions.csv")
        if not model_path.is_file() or not val_rows:
            return False
        for dataset in datasets:
            name = str(dataset["name"])
            metric = next((row for row in test_rows if str(row.get("Dataset")) == name), None)
            n_predictions = sum(1 for row in prediction_rows if str(row.get("Dataset")) == name)
            if (
                metric is None or pd.isna(metric.get("MCC"))
                or n_predictions < int(float(metric.get("N_Test_Samples", 0)))
            ):
                return False
    return True


def _pure_ml_fit_config(classifier, params, execution_config, use_pca):
    config = dict(execution_config)
    thread_key = "cpu_threads_pca" if use_pca else "cpu_threads_no_pca"
    config["cpu_threads"] = max(1, int(config.get(thread_key, config.get("cpu_threads", 16))))
    if not use_pca and "prefer_gpu_no_pca" in config:
        config["prefer_gpu"] = bool(config["prefer_gpu_no_pca"])
    effective_params = dict(params)
    if not use_pca and "no_pca_max_bin" in config:
        max_bin = int(config["no_pca_max_bin"])
        if max_bin < 2:
            raise ValueError("no_pca_max_bin must be at least 2.")
        bin_key = "border_count" if classifier == "CatBoost" else "max_bin"
        effective_params[bin_key] = min(int(effective_params.get(bin_key, max_bin)), max_bin)
    return effective_params, config


def run_pure_ml_pca_study(*, context, target, seeds, classifier_parameters,
                          output_root, pca_components=PCA_COMPONENTS, pca_modes=None,
                          execution_config=None, display_fn=print):
    """Evaluate tree models on flattened Cross-Attention-compatible token windows."""
    pca_modes = pca_modes or {"PCA": True, "NoPCA": False}
    if isinstance(pca_modes, dict):
        pca_modes = list(pca_modes.items())
    pca_modes = [(str(name), bool(enabled)) for name, enabled in pca_modes]
    mode_names = {name for name, _ in pca_modes}
    if mode_names != {"NoPCA", "PCA"} or len(pca_modes) != 2:
        raise ValueError(
            "Pure ML comparison requires exactly two modes: {'NoPCA': False, 'PCA': True}."
        )
    if dict(pca_modes) != {"NoPCA": False, "PCA": True}:
        raise ValueError("NoPCA must disable PCA and PCA must enable it.")
    # Enforce the requested order even if a caller constructs the mapping in a
    # different order: complete the PCA stage before starting any NoPCA fit.
    pca_modes = sorted(pca_modes, key=lambda item: 0 if item[0] == "PCA" else 1)
    execution_config = dict(execution_config or {
        "prefer_gpu": False, "fallback_to_cpu": True, "cpu_threads": 16,
        "gpu_ram_part": 0.8,
    })
    default_cpu_threads = max(1, int(execution_config.get("cpu_threads", 16)))
    output_root = Path(output_root) / "pure_ml_ref_alt_concat_window33_gpu_first_pca16_nopca8"
    output_root.mkdir(parents=True, exist_ok=True)
    signature = {
        "study": "pure_ml_ref_alt_concat_window33_pca_vs_no_pca",
        "target": target,
        "seeds": list(map(int, seeds)),
        "classifiers": list(CLASSIFIERS),
        "pca_modes": dict(pca_modes),
        "classifier_parameters": classifier_parameters,
        "execution_config": execution_config,
        "pca_components": int(pca_components),
        "pca_fit_scope": "train3_only; separately fit DNA-window and protein-window blocks; preserve tabular features",
        "input_representation": (
            "radius-16 token window; concat(E_ref_token, E_alt_token) at each relative position; "
            "flatten positions in order; append bio/SpliceAI/geometry"
        ),
        "token_window_radius": PURE_ML_TOKEN_WINDOW_RADIUS,
    }
    ensure_stage_signature(output_root, "pure_ml_ref_alt_concat_window33_ablation", signature)
    summary_path = output_root / "test_metrics_by_seed.csv"
    summary_rows = _load_rows(summary_path)
    # Mirror the fast in-memory workflow used by the earlier notebooks: hydrate
    # each feature split once, keep the matrices resident, and reuse them for
    # all classifiers, PCA modes, and seeds in this study.
    feature_cache = None
    for label, use_pca in pca_modes:
        cpu_threads = max(1, int(execution_config.get(
            "cpu_threads_pca" if use_pca else "cpu_threads_no_pca",
            default_cpu_threads,
        )))
        for seed in map(int, seeds):
            if _pure_ml_seed_complete(output_root, seed, context["datasets"]):
                print(f"[RESUME] Pure ML seed {seed}: all six classifier/PCA runs complete; skipping feature loading.")
                for classifier in CLASSIFIERS:
                    for summary_label, _use_pca in pca_modes:
                        run_test_rows = _load_rows(
                            output_root / classifier / summary_label / f"seed_{seed}" / "test_metrics.csv"
                        )
                        for row in run_test_rows:
                            summary_rows = upsert_record(
                                summary_rows, row,
                                ["Classifier", "PCA", "Seed", "Dataset"],
                            )
                save_progress_csv(summary_rows, summary_path)
                continue
            if _pure_ml_mode_seed_complete(output_root, seed, context["datasets"], label):
                print(
                    f"[RESUME] Pure ML {label}/seed_{seed}: all three classifiers and "
                    "four test splits are complete; skipping this mode without rebuilding features."
                )
                for classifier in CLASSIFIERS:
                    run_test_rows = _load_rows(
                        output_root / classifier / label / f"seed_{seed}" / "test_metrics.csv"
                    )
                    for row in run_test_rows:
                        summary_rows = upsert_record(
                            summary_rows, row,
                            ["Classifier", "PCA", "Seed", "Dataset"],
                        )
                save_progress_csv(summary_rows, summary_path)
                continue
            if feature_cache is None:
                _resource_snapshot("before Pure ML feature preload")
                print(
                    f"[Pure ML] Preloading radius-{PURE_ML_TOKEN_WINDOW_RADIUS} "
                    "train/validation feature matrices into disk-backed mmap files..."
                )
                matrix_cache_dir = output_root / "feature_matrix_cache"
                train, y_train, dims = _load_or_create_feature_memmap(
                    context, target, "train3", seed, matrix_cache_dir
                )
                validation, y_val, validation_dims = _load_or_create_feature_memmap(
                    context, target, "val", seed, matrix_cache_dir
                )
                if validation_dims != dims:
                    raise RuntimeError(
                        f"Feature dimensions differ between train and validation: "
                        f"{dims} vs {validation_dims}"
                    )
                feature_cache = {
                    "train": train, "y_train": y_train,
                    "validation": validation, "y_val": y_val,
                    "dims": dims,
                }
                total_bytes = train.nbytes + validation.nbytes
                print(
                    f"[Pure ML] Train/validation matrices mapped from disk: "
                    f"{total_bytes / (1024 ** 3):.2f} GiB logical float32 size; "
                    "reused across classifiers, modes, and seeds. OS page cache can "
                    "evict mapped pages under memory pressure. Test splits are loaded "
                    "one at a time after each model group is fitted."
                )
                _resource_snapshot("after Pure ML feature preload")
            else:
                print(f"[Pure ML] Reusing preloaded feature matrices from RAM for seed {seed}.")

            train = feature_cache["train"]
            y_train = feature_cache["y_train"]
            validation = feature_cache["validation"]
            y_val = feature_cache["y_val"]
            dims = feature_cache["dims"]
            pca_cache = None
            if use_pca:
                print(f"[Pure ML] Fitting PCA for seed {seed} before classifier training...")
                _resource_snapshot(f"before PCA fit for seed {seed}")
                pca_monitor = _ResourceSampler().start()
                try:
                    x_train_pca, transformed, pca_dna, pca_protein = _apply_pca(
                        train, [validation], dims, pca_components, seed,
                        n_threads=cpu_threads,
                    )
                finally:
                    pca_memory = pca_monitor.stop()
                print(f"[Pure ML PCA] seed_{seed}: peak_memory={pca_memory}")
                _resource_snapshot(f"after PCA fit for seed {seed}")
                pca_cache = {
                    "train": x_train_pca,
                    "validation": transformed[0],
                    "dna": pca_dna,
                    "protein": pca_protein,
                    "memory": pca_memory,
                }
            run_state = []
            for classifier in CLASSIFIERS:
                source = classifier_parameters[classifier]
                base_params, fit_execution_config = _pure_ml_fit_config(
                    classifier, source["params"], execution_config, use_pca
                )
                run_dir = output_root / classifier / label / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                model_path = run_dir / "best_model"
                if classifier == "XGBoost":
                    model_path = model_path.with_suffix(".json")
                elif classifier == "LightGBM":
                    model_path = model_path.with_suffix(".txt")
                else:
                    model_path = model_path.with_suffix(".cbm")
                val_path = run_dir / "validation_metrics.csv"
                test_path = run_dir / "test_metrics.csv"
                predictions_path = run_dir / "test_predictions.csv"
                ensure_stage_signature(run_dir, f"pure_ml_{classifier}_{label}_seed_{seed}", {
                    "target": target, "seed": seed, "classifier": classifier,
                    "parameters": base_params, "pca": use_pca,
                    "execution_config": execution_config,
                    "pca_components": int(pca_components),
                    "input": "radius16 token-level ref_alt_concat flattened by relative position + tabular",
                })
                saved_validation = _load_rows(val_path)
                val_record = next((row for row in saved_validation
                                   if int(float(row.get("Seed", -1))) == seed), None)
                saved_test = _load_rows(test_path)
                prediction_rows = _load_rows(predictions_path)
                expected_datasets = [str(item["name"]) for item in context["datasets"]]
                complete_tests = all(any(
                    str(row.get("Dataset")) == name
                    and int(float(row.get("Seed", -1))) == seed
                    and pd.notna(row.get("MCC"))
                    for row in saved_test
                ) and any(
                    str(row.get("Dataset")) == name
                    and int(float(row.get("Seed", -1))) == seed
                    for row in prediction_rows
                ) for name in expected_datasets)
                if val_record is not None and model_path.is_file() and complete_tests:
                    print(f"[RESUME] PureML {classifier}/{label}/seed_{seed}: complete; skipping fit and test.")
                    for row in saved_test:
                        summary_rows = upsert_record(
                            summary_rows, row,
                            ["Classifier", "PCA", "Seed", "Dataset"],
                        )
                    save_progress_csv(summary_rows, summary_path)
                    continue

                x_train, x_val = train, validation
                if use_pca:
                    pca_dna_path = run_dir / "pca_dna.joblib"
                    pca_protein_path = run_dir / "pca_protein.joblib"
                    x_train = pca_cache["train"]
                    x_val = pca_cache["validation"]
                    joblib.dump(pca_cache["dna"], pca_dna_path)
                    joblib.dump(pca_cache["protein"], pca_protein_path)

                if val_record is not None and model_path.is_file():
                    threshold = float(val_record["Decision_Threshold"])
                    fit_seconds = float(val_record.get("Fit_Seconds", 0.0))
                else:
                    params = dict(base_params)
                    _resource_snapshot(f"before {classifier}/{label}/seed_{seed} fit")
                    started = time.perf_counter()
                    resource_monitor = _ResourceSampler().start()
                    try:
                        model, execution_device = _fit_fixed_tree(
                            classifier, params, seed, x_train, y_train, x_val, y_val,
                            execution_config=fit_execution_config,
                            return_execution_device=True,
                        )
                    finally:
                        fit_memory = resource_monitor.stop()
                    fit_seconds = time.perf_counter() - started
                    print(
                        f"[Pure ML fit] {classifier}/{label}/seed_{seed}: "
                        f"device={execution_device}, CPU_threads={cpu_threads}, "
                        f"fit_time={fit_seconds:.1f}s, peak_memory={fit_memory}"
                    )
                    _resource_snapshot(f"after {classifier}/{label}/seed_{seed} fit")
                    probabilities = _predict_proba(classifier, model, x_val)
                    threshold, val_mcc = context["pipeline"]._select_validation_threshold(y_val, probabilities)
                    val_metrics = _compute_metrics(context["pipeline"], y_val, probabilities, threshold)
                    _save_tree_model(classifier, model, model_path)
                    val_record = {
                        "Branch": "Pure_ML", "Target_Config": target["name"],
                        "Classifier": classifier, "PCA": use_pca, "PCA_Mode": label, "Seed": seed,
                        "Input_Representation": "window_radius16_ref_alt_concat_flatten",
                        "Token_Window_Radius": PURE_ML_TOKEN_WINDOW_RADIUS,
                        "PCA_Components": int(pca_components) if use_pca else 0,
                        "Decision_Threshold": float(threshold), "Val_MCC": float(val_mcc),
                        "Fit_Seconds": round(fit_seconds, 3),
                        "Execution_Device": execution_device,
                        "CPU_Threads": cpu_threads,
                        **fit_memory,
                        **({
                            f"PCA_{key}": value
                            for key, value in pca_cache["memory"].items()
                        } if use_pca else {}),
                        "Legacy_Parameter_Source": source.get("source_csv"),
                        "Legacy_Parameter_Trial": source.get("source_trial"),
                        "Legacy_Parameter_Val_MCC": source.get("source_val_mcc"),
                        "Hyperparameters": json.dumps(params, sort_keys=True, default=_json_safe),
                        **{f"Val_{name}": value for name, value in val_metrics.items()},
                    }
                    _save_validation(val_path, val_record)
                if val_record is not None and model_path.is_file():
                    execution_device = str(val_record.get("Execution_Device", "CPU"))
                    if "CPU_Threads" not in val_record:
                        val_record["CPU_Threads"] = cpu_threads
                        _save_validation(val_path, val_record)
                run_state.append({
                    "classifier": classifier,
                    "label": label,
                    "use_pca": use_pca,
                    "run_dir": run_dir,
                    "model_path": model_path,
                    "test_path": test_path,
                    "predictions_path": predictions_path,
                    "saved_test": saved_test,
                    "prediction_rows": prediction_rows,
                    "base_params": base_params,
                    "val_record": val_record,
                    "threshold": threshold,
                    "fit_seconds": fit_seconds,
                    "execution_device": execution_device,
                })
                if "model" in locals():
                    del model
                if "probabilities" in locals():
                    del probabilities
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Hold only train/validation while fitting. At evaluation time, load
            # one test split, score every pending classifier, then release it.
            for dataset in context["datasets"]:
                dataset_name = str(dataset["name"])
                pending = []
                for state in run_state:
                    metric_exists = any(
                        str(row.get("Dataset")) == dataset_name
                        and int(float(row.get("Seed", -1))) == seed
                        and pd.notna(row.get("MCC"))
                        for row in state["saved_test"]
                    )
                    prediction_exists = any(
                        str(row.get("Dataset")) == dataset_name
                        and int(float(row.get("Seed", -1))) == seed
                        for row in state["prediction_rows"]
                    )
                    if metric_exists and prediction_exists:
                        saved_row = next(
                            row for row in state["saved_test"]
                            if str(row.get("Dataset")) == dataset_name
                            and int(float(row.get("Seed", -1))) == seed
                        )
                        summary_rows = upsert_record(
                            summary_rows, saved_row,
                            ["Classifier", "PCA", "Seed", "Dataset"],
                        )
                    else:
                        pending.append(state)

                if not pending:
                    save_progress_csv(summary_rows, summary_path)
                    continue

                print(f"[Pure ML] Loading one test split for scoring: {dataset_name}...")
                test_matrix, y_test, ids, _ = _collect_feature_matrix(
                    context, target, dataset["test"], seed,
                    embedding_representation="window_radius16_ref_alt_concat_flatten",
                )
                for state in pending:
                    classifier = state["classifier"]
                    model = _load_tree_model(classifier, state["model_path"])
                    if state["use_pca"]:
                        dna_dim, protein_dim = dims["dna"], dims["protein"]
                        dna = pca_cache["dna"].transform(test_matrix[:, :dna_dim])
                        protein = pca_cache["protein"].transform(
                            test_matrix[:, dna_dim:dna_dim + protein_dim]
                        )
                        tabular = test_matrix[:, dna_dim + protein_dim:]
                        x_test = np.concatenate((dna, protein, tabular), axis=1).astype(
                            np.float32, copy=False
                        )
                        del dna, protein, tabular
                    else:
                        x_test = test_matrix
                    threshold = state["threshold"]
                    probabilities = _predict_proba(classifier, model, x_test)
                    metrics = _compute_metrics(context["pipeline"], y_test, probabilities, threshold)
                    row = {
                        "Run_ID": Path(context["pipeline"].batch_run_dir).name,
                        "Branch": "Pure_ML", "Target_Config": target["name"],
                        "Dataset": dataset_name, "Classifier": classifier,
                        "PCA": state["use_pca"], "PCA_Mode": state["label"], "Seed": seed,
                        "Input_Representation": "window_radius16_ref_alt_concat_flatten",
                        "Token_Window_Radius": PURE_ML_TOKEN_WINDOW_RADIUS,
                        "Decision_Threshold": threshold, "N_Test_Samples": len(y_test),
                        "Fit_Seconds": round(state["fit_seconds"], 3),
                        "Execution_Device": state["execution_device"],
                        "CPU_Threads": cpu_threads,
                        **{
                            key: state["val_record"].get(key)
                            for key in (
                                "Fit_Peak_RSS_GiB", "Fit_Min_Available_RAM_GiB",
                                "Fit_Peak_GPU_Device_Used_GiB",
                            )
                        },
                        **({
                            f"PCA_{key}": value
                            for key, value in pca_cache["memory"].items()
                        } if state["use_pca"] else {}),
                        "Hyperparameters": json.dumps(
                            state["base_params"], sort_keys=True, default=_json_safe
                        ),
                        **metrics,
                    }
                    saved_test = upsert_record(
                        state["saved_test"], row,
                        ["Seed", "Classifier", "PCA", "Dataset"],
                    )
                    prediction_rows = [r for r in state["prediction_rows"] if not (
                        str(r.get("Dataset")) == dataset_name
                        and int(float(r.get("Seed", -1))) == seed
                    )]
                    prediction_rows.extend({
                        "Classifier": classifier, "PCA": state["use_pca"],
                        "PCA_Mode": state["label"], "Seed": seed,
                        "Dataset": dataset_name, "Variant_ID": str(variant_id),
                        "Label": int(label_value),
                        "Probability_Pathogenic": float(probability),
                        "Prediction": int(probability >= threshold),
                    } for variant_id, label_value, probability in zip(ids, y_test, probabilities))
                    save_progress_csv(saved_test, state["test_path"])
                    save_progress_csv(prediction_rows, state["predictions_path"])
                    state["saved_test"] = saved_test
                    state["prediction_rows"] = prediction_rows
                    summary_rows = upsert_record(
                        summary_rows, row,
                        ["Classifier", "PCA", "Seed", "Dataset"],
                    )
                    save_progress_csv(summary_rows, summary_path)
                    print(
                        f"[Pure ML] {classifier}/{state['label']}/seed_{seed}/{dataset_name}: "
                        f"MCC={metrics['MCC']:.4f}"
                    )
                    del model, x_test, probabilities
                del test_matrix, y_test, ids
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return pd.DataFrame(_load_rows(summary_path))


class CrossAttentionSubsetModel(nn.Module):
    """The original tri-modal cross-attention with only configured branches present."""

    def __init__(self, dimensions, config, active_branches):
        super().__init__()
        self.active_branches = tuple(active_branches)
        if not self.active_branches:
            raise ValueError("At least one modality branch must be active.")
        model_dim = int(config["model_dim"])
        dropout = float(config["dropout"])
        feature_dropout = float(config["feature_dropout"])
        self.k_radius = int(config["k_radius"])
        self.pooling = str(config["pooling"])
        if "dna" in self.active_branches:
            self.dna_modality = nn.Parameter(torch.zeros(model_dim))
            self.dna_window = _VariantCenteredWindow(dimensions["token_dna"], model_dim, dropout)
        if "protein" in self.active_branches:
            self.protein_modality = nn.Parameter(torch.zeros(model_dim))
            self.protein_window = _VariantCenteredWindow(dimensions["token_protein"], model_dim, dropout)
        if "bio" in self.active_branches:
            self.bio_modality = nn.Parameter(torch.zeros(model_dim))
            self.geometry_modality = nn.Parameter(torch.zeros(model_dim))
            self.bio_tokenizer = _ScalarFeatureTokenizer(dimensions["bio"], model_dim, feature_dropout)
            self.geom_tokenizer = _ScalarFeatureTokenizer(dimensions["geometry"], model_dim, feature_dropout)

        self.cross_layers = nn.ModuleList()
        n_attention_layers = int(config["num_layers"]) if len(self.active_branches) > 1 else 0
        for _ in range(n_attention_layers):
            self.cross_layers.append(nn.ModuleDict({
                branch: _CrossAttentionBlock(
                    model_dim, int(config["num_heads"]), int(config["ffn_multiplier"]),
                    dropout, float(config["attention_dropout"]),
                )
                for branch in self.active_branches
            }))
        self.pool_scores = nn.ModuleDict({
            branch: nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
            for branch in self.active_branches
        })
        self.output_norm = nn.LayerNorm(model_dim * len(self.active_branches))
        self.classifier = nn.Sequential(
            nn.Linear(model_dim * len(self.active_branches), model_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def _pool(self, tokens, mask, branch):
        if self.pooling == "mean":
            weights = mask.unsqueeze(-1).to(tokens.dtype)
            return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        scores = self.pool_scores[branch](tokens).squeeze(-1).masked_fill(~mask, -torch.inf)
        return (tokens * torch.softmax(scores, dim=1).unsqueeze(-1)).sum(dim=1)

    def encode_features(self, v_dna=None, v_prot=None, bio_features=None, geom_features=None,
                        token_dna=None, token_dna_mask=None, token_dna_variant_index=None,
                        token_protein=None, token_protein_mask=None,
                        token_protein_variant_index=None):
        states, masks = {}, {}
        if "dna" in self.active_branches:
            states["dna"], masks["dna"] = self.dna_window(
                token_dna, token_dna_mask, token_dna_variant_index,
                self.k_radius, self.dna_modality,
            )
        if "protein" in self.active_branches:
            states["protein"], masks["protein"] = self.protein_window(
                token_protein, token_protein_mask, token_protein_variant_index,
                self.k_radius, self.protein_modality,
            )
        if "bio" in self.active_branches:
            device = next(value for value in (bio_features, geom_features, v_dna, v_prot,
                token_dna, token_protein) if value is not None).device
            tokens = []
            if self.bio_tokenizer.n_features:
                tokens.append(self.bio_tokenizer(bio_features) + self.bio_modality.view(1, 1, -1))
            if self.geom_tokenizer.n_features:
                tokens.append(self.geom_tokenizer(geom_features) + self.geometry_modality.view(1, 1, -1))
            if not tokens:
                raise ValueError("Bio branch was enabled without biological or geometry inputs.")
            states["bio"] = torch.cat(tokens, dim=1)
            masks["bio"] = torch.ones(states["bio"].shape[:2], dtype=torch.bool, device=device)

        for layer in self.cross_layers:
            old_states = states
            updated = dict(old_states)
            if len(self.active_branches) > 1:
                for branch in self.active_branches:
                    context_tokens = torch.cat([
                        old_states[name] for name in self.active_branches if name != branch
                    ], dim=1)
                    context_mask = torch.cat([
                        masks[name] for name in self.active_branches if name != branch
                    ], dim=1)
                    updated[branch] = layer[branch](old_states[branch], context_tokens, context_mask)
            states = updated
        pooled = [self._pool(states[name], masks[name], name) for name in self.active_branches]
        return self.output_norm(torch.cat(pooled, dim=1))

    def forward(self, *args, **kwargs):
        return self.classifier(self.encode_features(*args, **kwargs)).squeeze(-1)


def _subset_builder(dimensions, config):
    return CrossAttentionSubsetModel(dimensions, config, config["ablation_active_branches"])


def _feature_target(base_target, bio_features, geom_features, label):
    bio_features = list(bio_features)
    geom_features = list(geom_features)
    active = ["dna", "prot"]
    if any(name in BIO_CORE_COLUMNS for name in bio_features):
        active.append("bio_core")
    if any(name in SPLICEAI_COLUMNS for name in bio_features):
        active.append("spliceai")
    if geom_features:
        active.append("geom")
    target = dict(base_target)
    target.update({
        "name": "B_with_geometry",
        "ablation": label,
        "active_modalities": active,
        "bio_feature_names": bio_features,
        "geom_feature_names": geom_features,
    })
    return target


def _modalities_target(base_target, modalities, bio_features, geom_features, label):
    target = _feature_target(base_target, bio_features, geom_features, label)
    active = list(modalities)
    target["active_modalities"] = active
    return target


def _run_dl_condition(*, context, base_target, condition, seed, config, output_root,
                      study, display_fn=print):
    target = dict(condition["target"])
    model_config = dict(config)
    model_config["seed"] = int(seed)
    active_branches = []
    modalities = {str(item).lower() for item in target["active_modalities"]}
    if "dna" in modalities:
        active_branches.append("dna")
    if "prot" in modalities:
        active_branches.append("protein")
    if target.get("bio_feature_names") or target.get("geom_feature_names"):
        active_branches.append("bio")
    model_config["ablation_active_branches"] = active_branches
    uses_original_full_model = set(active_branches) == {"dna", "protein", "bio"}
    builder = (
        (lambda dims, cfg: CrossAttentionThreeModalities(
            dims["token_dna"], dims["token_protein"], dims["bio"], dims["geometry"], cfg
        ))
        if uses_original_full_model else _subset_builder
    )
    run_key = f"{study}_{_slug(condition['label'])}_seed_{int(seed)}"
    exp_context = dict(context)
    exp_context["TARGET_CONFIGS"] = [target]
    exp_context["token_modalities"] = [
        name for branch, name in (("dna", "dna"), ("protein", "protein"))
        if branch in active_branches
    ]
    result = run_architecture_experiment(
        context=exp_context,
        experiment_config=model_config,
        model_builder=builder,
        architecture="CrossAttention_ThreeModalities",
        variant=f"{study}_{condition['label']}",
        experiment_key=f"cross_attention_ablation_{run_key}",
        source=f"module06_{study}",
        display_fn=lambda *_args, **_kwargs: None,
        manifest_architecture=f"cross_attention_active_branches_{'-'.join(active_branches)}_v1",
        embedding_representation="ref_alt_concat",
        evaluate_test=True,
    )
    validation = result["validation_metrics"].copy()
    test = result["test_metrics"].copy()
    if not validation.empty:
        validation["Study"] = study
        validation["Condition"] = condition["label"]
        validation["Seed"] = int(seed)
        save_progress_csv(validation.to_dict("records"), Path(result["output_dir"]) / "study_validation_metrics.csv")
    if not test.empty:
        test["Study"] = study
        test["Condition"] = condition["label"]
        test["Seed"] = int(seed)
        save_progress_csv(test.to_dict("records"), Path(result["output_dir"]) / "study_test_metrics.csv")
    display_fn(f"[DONE] {study}/{condition['label']}/seed_{seed}")
    return validation, test


def feature_group_conditions(base_target):
    bio = list(BIO_CORE_COLUMNS) + list(SPLICEAI_COLUMNS)
    geom = list(GEOM_FEATURES)
    groups = [
        ("Drop_AF", ["AF", "gnomADe_AF"]),
        ("Drop_Conservation", [name for name in BIO_CORE_COLUMNS if name not in {"AF", "gnomADe_AF"}]),
        ("Drop_SpliceAI", list(SPLICEAI_COLUMNS)),
        ("Drop_All_Geometry", list(GEOM_FEATURES)),
        ("Drop_DNA_Geometry", [name for name in GEOM_FEATURES if name.startswith("dna_")]),
        ("Drop_Protein_Geometry", [name for name in GEOM_FEATURES if name.startswith("prot_")]),
    ]
    return [{
        "label": label,
        "target": _feature_target(base_target, [name for name in bio if name not in removed],
                                   [name for name in geom if name not in removed], label),
    } for label, removed in groups]


def feature_28_conditions(base_target):
    bio = list(BIO_CORE_COLUMNS) + list(SPLICEAI_COLUMNS)
    geom = list(GEOM_FEATURES)
    return [{
        "label": f"Drop_{name}",
        "target": _feature_target(
            base_target, [col for col in bio if col != name],
            [col for col in geom if col != name], f"drop_feature_{name}",
        ),
    } for name in (*bio, *geom)]


def modality_conditions(base_target):
    core = list(BIO_CORE_COLUMNS)
    splice = list(SPLICEAI_COLUMNS)
    geom = list(GEOM_FEATURES)
    specs = [
        ("Bio_Core_Only", ["bio_core"], core, []),
        ("DNA_LM_Only", ["dna"], [], []),
        ("Protein_LM_Only", ["prot"], [], []),
        ("SpliceAI_Only", ["spliceai"], splice, []),
        ("Geometry_Only", ["geom"], [], geom),
        ("DNA_Protein_Only", ["dna", "prot"], [], []),
        ("Bio_SpliceAI_Geometry_Only", ["bio_core", "spliceai", "geom"], [*core, *splice], geom),
        ("Protein_Bio_SpliceAI_Geometry", ["prot", "bio_core", "spliceai", "geom"], [*core, *splice], geom),
        ("DNA_Bio_SpliceAI_Geometry", ["dna", "bio_core", "spliceai", "geom"], [*core, *splice], geom),
        ("DNA_Protein_Bio_Core", ["dna", "prot", "bio_core"], core, []),
    ]
    return [{
        "label": label,
        "target": _modalities_target(base_target, active, bio, geometry, label),
    } for label, active, bio, geometry in specs]


def run_cross_attention_ablation_study(*, context, base_target, conditions, seeds,
                                       model_config, output_root, study, baseline_test,
                                       display_fn=print):
    output_root = Path(output_root) / study
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_stage_signature(output_root, f"cross_attention_{study}", {
        "study": study, "conditions": [condition["target"] for condition in conditions],
        "seeds": list(map(int, seeds)), "model_config": model_config,
        "baseline_source": context.get("baseline_source"),
        "target": base_target,
    }, mutable_paths=(("model_config", "batch_size"),)
    if context.get("allow_batch_size_resume") else ())
    validation_rows, test_rows = [], []
    for condition in conditions:
        for seed in map(int, seeds):
            print(f"[START] {study}/{condition['label']}/seed_{seed}")
            validation, test = _run_dl_condition(
                context=context, base_target=base_target, condition=condition,
                seed=seed, config=model_config, output_root=output_root,
                study=study, display_fn=display_fn,
            )
            validation_rows.extend(validation.to_dict("records"))
            test_rows.extend(test.to_dict("records"))
    validation_frame = pd.DataFrame(validation_rows)
    test_frame = pd.DataFrame(test_rows)
    save_progress_csv(validation_frame.to_dict("records"), output_root / "validation_metrics_by_seed.csv")
    save_progress_csv(test_frame.to_dict("records"), output_root / "test_metrics_by_seed.csv")
    comparison = test_frame.merge(
        baseline_test[["Seed", "Dataset", "MCC", "AUROC"]].rename(columns={
            "MCC": "Baseline_MCC", "AUROC": "Baseline_AUROC",
        }), on=["Seed", "Dataset"], how="left", validate="many_to_one",
    ) if not test_frame.empty else pd.DataFrame()
    if not comparison.empty:
        comparison["Delta_MCC"] = comparison["MCC"] - comparison["Baseline_MCC"]
        comparison["Delta_AUROC"] = comparison["AUROC"] - comparison["Baseline_AUROC"]
        save_progress_csv(comparison.to_dict("records"), output_root / "delta_vs_baseline_by_seed.csv")
        summary = comparison.groupby(["Condition", "Dataset"], dropna=False)[
            ["MCC", "AUROC", "Baseline_MCC", "Baseline_AUROC", "Delta_MCC", "Delta_AUROC"]
        ].agg(["mean", "std", "count"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        save_progress_csv(summary.to_dict("records"), output_root / "delta_mean_std_by_condition_dataset.csv")
    else:
        summary = pd.DataFrame()
    return {"validation": validation_frame, "test": test_frame,
            "comparison": comparison, "summary": summary}

