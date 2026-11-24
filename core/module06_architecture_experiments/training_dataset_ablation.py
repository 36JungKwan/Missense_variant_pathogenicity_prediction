"""Training-set size/composition ablations for the fixed 3-modal Cross-Attention model."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from core.module05v2_fusion_classifier.resume_utils import (
    ensure_stage_signature,
    replace_file_with_retry,
    save_progress_csv,
)
from core.module06_architecture_experiments.architecture_experiment_runner import (
    run_architecture_experiment,
)
from core.module06_architecture_experiments.cross_attention_3modal_experiment import (
    _builder as build_cross_attention_model,
)


REFERENCE_METRICS = (
    "Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "MCC", "AUROC", "AUPRC",
)
VALIDATION_METRICS = tuple(f"Val_{name}" for name in REFERENCE_METRICS)
PROFILE_METRICS = (
    "FM_GFLOPs/Sample", "FM_Latency_ms/Sample", "Geom_GFLOPs/Sample",
    "Geom_Latency_ms/Sample", "Classifier_GFLOPs/Sample",
    "Classifier_Latency_ms/Sample", "Classifier_Total_Params",
    "Classifier_Param_Mem_MB", "Classifier_Peak_VRAM_MB",
    "E2E_GFLOPs/Sample", "E2E_Latency_ms/Sample", "E2E_Total_Params",
    "E2E_Param_Mem_MB", "Max_Peak_VRAM_MB",
)
REFERENCE_DATASETS = {
    "test": "Train3Val_Test",
    "clinvarhq": "Train3Val_ClinVarHQ",
    "proteingym": "Train3Val_ProteinGym",
    "uniprot": "Train3Val_UniProt",
}


def _atomic_json(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        replace_file_with_retry(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _file_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _label_column(path):
    import pyarrow.parquet as pq

    names = set(pq.read_schema(path).names)
    return next((name for name in ("Pathogenicity_Label", "Label", "label", "target", "Target") if name in names), None)


def _read_ids_and_labels(path):
    label_col = _label_column(path)
    if label_col is None:
        raise KeyError(f"No recognized binary label column in {path}.")
    frame = pd.read_parquet(path, columns=["Variant_ID", label_col])
    labels = pd.to_numeric(frame[label_col], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"Missing/non-numeric labels found in {path}.")
    labels = labels.astype(int).to_numpy()
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError(f"Expected binary labels 0/1 in {path}; got {sorted(set(labels))}.")
    ids = frame["Variant_ID"].astype(str).to_numpy()
    if pd.Series(ids).duplicated().any():
        raise ValueError(f"Variant_ID values are not unique in {path}.")
    return ids, labels, label_col


def _stratified_exact_sample(ids, labels, sample_count, random_state):
    """Sample exactly N rows, allocating class quotas by largest remainder."""
    sample_count = int(sample_count)
    if sample_count <= 0 or sample_count > len(ids):
        raise ValueError(f"Invalid stratified sample size {sample_count} for N={len(ids)}.")
    classes, counts = np.unique(labels, return_counts=True)
    exact = counts.astype(np.float64) * (sample_count / len(labels))
    quotas = np.floor(exact).astype(int)
    left = sample_count - int(quotas.sum())
    remainder_order = sorted(
        range(len(classes)), key=lambda index: (-(exact[index] - quotas[index]), int(classes[index]))
    )
    for index in remainder_order[:left]:
        quotas[index] += 1
    if np.any(quotas > counts):
        raise ValueError("Requested downsample count exceeds one or more source-class counts.")

    rng = np.random.default_rng(int(random_state))
    chosen = []
    for label, quota in zip(classes, quotas):
        candidates = np.flatnonzero(labels == label)
        chosen.extend(rng.choice(candidates, size=int(quota), replace=False).tolist())
    chosen = np.asarray(sorted(chosen), dtype=np.int64)
    if len(chosen) != sample_count:
        raise RuntimeError(f"Stratified sampling returned {len(chosen)} rows, expected {sample_count}.")
    sampled_ids = ids[chosen].astype(str).tolist()
    sampled_labels = labels[chosen]
    class_counts = {str(int(label)): int((sampled_labels == label).sum()) for label in classes}
    return sampled_ids, class_counts


def prepare_training_conditions(
    *, processed_dir, output_root, downsample_random_state=20261125,
    train_splits=("train1", "train2", "train3"),
):
    """Create resumable full/equal-count cohorts and persist their exact IDs."""
    processed_dir = Path(processed_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_paths = {
        split: processed_dir / f"{split}_normalized.parquet"
        for split in train_splits
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing normalized training parquets:\n" + "\n".join(missing))

    source_data = {}
    source_signatures = {}
    for split, path in source_paths.items():
        ids, labels, label_col = _read_ids_and_labels(path)
        source_data[split] = (ids, labels, label_col)
        source_signatures[split] = _file_signature(path)

    target_count = int(len(source_data["train3"][0]))
    train3_labels = source_data["train3"][1]
    train3_counts = {str(label): int((train3_labels == int(label)).sum()) for label in (0, 1)}
    sampling_manifest_path = output_root / "sampling_manifest.json"
    sampling_manifest = {
        "schema_version": 1,
        "downsample_random_state": int(downsample_random_state),
        "equal_sample_target": target_count,
        "source_files": source_signatures,
        "train3_reference_class_counts": train3_counts,
        "sample_id_files": {},
    }
    if sampling_manifest_path.is_file():
        saved = json.loads(sampling_manifest_path.read_text(encoding="utf-8"))
        for key in (
            "schema_version", "downsample_random_state", "equal_sample_target",
            "source_files", "train3_reference_class_counts",
        ):
            if saved.get(key) != sampling_manifest[key]:
                raise ValueError(
                    f"Saved training cohort manifest differs at {key}; start a fresh batch "
                    "instead of silently changing the sampled training set."
                )
        sampling_manifest = saved

    conditions = []
    inventory_rows = []
    for split in ("train1", "train2"):
        ids, labels, label_col = source_data[split]
        original_counts = {str(label): int((labels == int(label)).sum()) for label in (0, 1)}
        full_condition = {
            "key": f"{split}_full",
            "display_name": f"{split.title()} full",
            "training_split": split,
            "sampling_mode": "full_dataset",
            "training_sample_ids": None,
            "training_sample_count": int(len(ids)),
            "training_class_counts": original_counts,
            "source_class_counts": original_counts,
            "sample_signature": hashlib.sha256(
                json.dumps({"source": source_signatures[split], "mode": "full"}, sort_keys=True).encode()
            ).hexdigest(),
        }
        conditions.append(full_condition)
        inventory_rows.append({
            "Training_Cohort": full_condition["display_name"],
            "Training_Split": split,
            "Sampling_Mode": full_condition["sampling_mode"],
            "Training_N": len(ids),
            "Benign_N": original_counts.get("0", 0),
            "Pathogenic_N": original_counts.get("1", 0),
            "Benign_Pct": original_counts.get("0", 0) / len(ids),
            "Pathogenic_Pct": original_counts.get("1", 0) / len(ids),
            "Source_N": len(ids),
            "Status": "train",
        })

        if len(ids) < target_count:
            raise ValueError(
                f"Cannot downsample {split} (N={len(ids)}) to Train3 N={target_count}."
            )
        split_seed = int(downsample_random_state) + zlib.crc32(split.encode("utf-8"))
        sample_ids, sampled_counts = _stratified_exact_sample(
            ids, labels, target_count, split_seed
        )
        sample_file = output_root / f"sample_ids_{split}_equal_train3.json"
        sample_signature = hashlib.sha256(
            json.dumps(sample_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        sample_payload = {
            "schema_version": 1,
            "source_split": split,
            "source_file": source_signatures[split],
            "label_column": label_col,
            "downsample_random_state": int(split_seed),
            "target_count": target_count,
            "class_counts": sampled_counts,
            "sample_signature": sample_signature,
            "variant_ids": sample_ids,
        }
        if sample_file.is_file():
            saved_sample = json.loads(sample_file.read_text(encoding="utf-8"))
            if saved_sample != sample_payload:
                raise ValueError(
                    f"Saved sample IDs differ from the current {split} inputs/settings: {sample_file}. "
                    "Start a fresh batch to use a different subset."
                )
        else:
            _atomic_json(sample_payload, sample_file)
        sampling_manifest["sample_id_files"][split] = str(sample_file.resolve())
        equal_condition = {
            "key": f"{split}_equal_train3",
            "display_name": f"{split.title()} stratified to Train3 N",
            "training_split": split,
            "sampling_mode": "stratified_downsample_to_train3_count",
            "training_sample_ids": sample_ids,
            "training_sample_count": int(target_count),
            "training_class_counts": sampled_counts,
            "source_class_counts": original_counts,
            "sample_signature": sample_signature,
        }
        conditions.append(equal_condition)
        inventory_rows.append({
            "Training_Cohort": equal_condition["display_name"],
            "Training_Split": split,
            "Sampling_Mode": equal_condition["sampling_mode"],
            "Training_N": target_count,
            "Benign_N": sampled_counts.get("0", 0),
            "Pathogenic_N": sampled_counts.get("1", 0),
            "Benign_Pct": sampled_counts.get("0", 0) / target_count,
            "Pathogenic_Pct": sampled_counts.get("1", 0) / target_count,
            "Source_N": len(ids),
            "Status": "train",
        })

    inventory_rows.append({
        "Training_Cohort": "Train3 full (Notebook 17 reference)",
        "Training_Split": "train3",
        "Sampling_Mode": "existing_reference_no_rerun",
        "Training_N": target_count,
        "Benign_N": train3_counts.get("0", 0),
        "Pathogenic_N": train3_counts.get("1", 0),
        "Benign_Pct": train3_counts.get("0", 0) / target_count,
        "Pathogenic_Pct": train3_counts.get("1", 0) / target_count,
        "Source_N": target_count,
        "Status": "reuse_notebook17",
    })
    _atomic_json(sampling_manifest, sampling_manifest_path)
    inventory = pd.DataFrame(inventory_rows)
    save_progress_csv(inventory.to_dict("records"), output_root / "training_cohort_inventory.csv")
    return conditions, inventory, sampling_manifest


def _upsert_frame(existing, incoming, keys):
    if incoming is None or incoming.empty:
        return existing
    if existing is None or existing.empty:
        combined = incoming.copy()
    else:
        combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    return combined.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def load_train3_reference_metrics(
    *, reference_batch_dir, expected_seeds, train3_sample_count,
    train3_class_counts=None,
    expected_model_config=None, expected_input_data_signature=None,
):
    """Load the already completed Notebook 17 Cross-Attention/MLP baseline."""
    root = Path(reference_batch_dir) / "module06_architecture_experiments" / "fusion_classifier_benchmark"
    test_path = root / "seed_level_test_metrics.csv"
    val_path = root / "seed_level_validation_metrics.csv"
    if not test_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(
            "Notebook 17 Train3 baseline files are missing:\n"
            f"{test_path}\n{val_path}"
        )
    configuration_path = root / "run_configuration.json"
    if not configuration_path.is_file():
        raise FileNotFoundError(f"Notebook 17 run configuration is missing: {configuration_path}")
    saved_configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    saved_model_config = saved_configuration.get("fixed_model_config", {})
    mismatched_hyperparameters = {
        key: (saved_model_config.get(key), value)
        for key, value in (expected_model_config or {}).items()
        if saved_model_config.get(key) != value
    }
    if mismatched_hyperparameters:
        raise ValueError(
            "Notebook 17 Train3 reference does not use the requested fixed hyperparameters: "
            f"{mismatched_hyperparameters}"
        )
    saved_input_signature = saved_configuration.get("input_data_signature", {})
    mismatched_inputs = {
        key: (saved_input_signature.get(key), value)
        for key, value in (expected_input_data_signature or {}).items()
        if saved_input_signature.get(key) != value
    }
    if mismatched_inputs:
        raise ValueError(
            "Notebook 17 Train3 reference input/geometry signature differs from this study: "
            f"{mismatched_inputs}"
        )
    test = pd.read_csv(test_path)
    validation = pd.read_csv(val_path)

    def _filter(frame):
        required = {
            "Fusion": "CrossAttention_ThreeModalities",
            "Classifier": "MLP",
            "Target_Config": "B_with_geometry",
            "Input_Representation": "ref_alt_concat",
            "DNA_Model": "nt_v1_500m_dna_1001",
            "Prot_Model": "esm1b_650m_prot_501",
        }
        mask = pd.Series(True, index=frame.index)
        for column, value in required.items():
            if column not in frame.columns:
                raise KeyError(f"Notebook 17 results are missing required column {column}.")
            mask &= frame[column].astype(str).eq(value)
        if "Ablation" in frame.columns:
            mask &= frame["Ablation"].astype(str).eq("bio_core_dna_prot_spliceai_geom")
        return frame.loc[mask].copy()

    test = _filter(test)
    validation = _filter(validation)
    expected_seeds = {int(seed) for seed in expected_seeds}
    test["Seed"] = pd.to_numeric(test["Seed"], errors="coerce")
    validation["Seed"] = pd.to_numeric(validation["Seed"], errors="coerce")
    test = test[test["Seed"].isin(expected_seeds)].copy()
    validation = validation[validation["Seed"].isin(expected_seeds)].copy()
    required_datasets = set(REFERENCE_DATASETS.values())
    if test.duplicated(["Seed", "Dataset"]).any():
        raise ValueError("Notebook 17 Train3 baseline has duplicate seed/test-dataset rows.")
    observed = set(test["Dataset"].astype(str))
    if observed != required_datasets:
        raise ValueError(
            f"Notebook 17 Train3 baseline test sets differ: observed={sorted(observed)}, "
            f"expected={sorted(required_datasets)}."
        )
    if set(pd.to_numeric(test["Seed"], errors="coerce").dropna().astype(int)) != expected_seeds:
        raise ValueError("Notebook 17 Train3 test baseline does not contain all requested seeds.")
    if set(pd.to_numeric(validation["Seed"], errors="coerce").dropna().astype(int)) != expected_seeds:
        raise ValueError("Notebook 17 Train3 validation baseline does not contain all requested seeds.")
    if len(test) != len(expected_seeds) * len(required_datasets):
        raise ValueError("Notebook 17 Train3 baseline is incomplete for seed × test-set coverage.")
    for metric in ("MCC", "AUROC", "AUPRC"):
        if metric not in test or pd.to_numeric(test[metric], errors="coerce").isna().any():
            raise ValueError(f"Notebook 17 Train3 test baseline is missing complete {metric} values.")

    test["Training_Cohort"] = "train3_reference"
    validation["Training_Cohort"] = "train3_reference"
    test["Training_Cohort_Name"] = "Train3 full (Notebook 17 reference)"
    validation["Training_Cohort_Name"] = "Train3 full (Notebook 17 reference)"
    train3_class_counts = {
        str(key): int(value) for key, value in (train3_class_counts or {}).items()
    }
    for frame in (test, validation):
        frame["Training_Split"] = "train3"
        frame["Training_Sampling_Mode"] = "existing_reference_no_rerun"
        frame["Training_N_Samples"] = int(train3_sample_count)
        frame["Training_Benign_N"] = train3_class_counts.get("0", np.nan)
        frame["Training_Pathogenic_N"] = train3_class_counts.get("1", np.nan)
        frame["Training_Sample_Signature"] = "reused_notebook17_seed_checkpoint_results"
    return validation, test


def run_training_dataset_ablation(
    *, context, conditions, seeds, fixed_model_config, output_root,
    train3_reference_validation, train3_reference_test, display_fn=None,
):
    """Train Train1/Train2 cohorts with resumable Cross-Attention checkpoints."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    display = display_fn or print
    seeds = [int(seed) for seed in seeds]
    pipeline = context["pipeline"]
    datasets_template = tuple(context.get("test_splits", ("test", "clinvarhq", "proteingym", "uniprot")))
    cohort_signature = [
        {
            key: value for key, value in condition.items()
            if key != "training_sample_ids"
        }
        for condition in conditions
    ]
    run_signature = {
        "architecture": "CrossAttention_ThreeModalities",
        "variant": "ref_alt_concat_variant_centered_window_k16",
        "target_configs": context["TARGET_CONFIGS"],
        "dna_model": context["DNA_MODEL"],
        "protein_model": context["PROTEIN_MODEL"],
        "seeds": seeds,
        "fixed_model_config": fixed_model_config,
        "test_splits": list(datasets_template),
        "validation_split": "val",
        "geometry_root": context.get("geometry_root_name"),
        "cohorts": cohort_signature,
        "embedding_representation": "ref_alt_concat",
        "train3_reference_source": context.get("train3_reference_source"),
    }
    ensure_stage_signature(
        output_root,
        "module06_training_dataset_ablation",
        run_signature,
        mutable_paths=(("fixed_model_config", "batch_size"),),
    )

    validation_path = output_root / "seed_level_validation_metrics.csv"
    test_path = output_root / "seed_level_test_metrics.csv"
    prediction_path = output_root / "seed_level_test_predictions.csv"
    validation = pd.read_csv(validation_path) if validation_path.is_file() else pd.DataFrame()
    test = pd.read_csv(test_path) if test_path.is_file() else pd.DataFrame()
    predictions = pd.read_csv(prediction_path) if prediction_path.is_file() else pd.DataFrame()
    # The Train3 control is always refreshed from its frozen Notebook 17 source.
    if not validation.empty and "Training_Cohort" in validation:
        validation = validation[validation["Training_Cohort"].astype(str) != "train3_reference"]
    if not test.empty and "Training_Cohort" in test:
        test = test[test["Training_Cohort"].astype(str) != "train3_reference"]
    if not predictions.empty and "Training_Cohort" in predictions:
        predictions = predictions[predictions["Training_Cohort"].astype(str) != "train3_reference"]
    validation = _upsert_frame(
        validation, train3_reference_validation,
        ["Training_Cohort", "Seed", "Target_Config"],
    )
    test = _upsert_frame(
        test, train3_reference_test,
        ["Training_Cohort", "Seed", "Target_Config", "Dataset"],
    )

    for condition in conditions:
        split = str(condition["training_split"])
        test_suffixes = {
            "test": "Test",
            "clinvarhq": "ClinVarHQ",
            "proteingym": "ProteinGym",
            "uniprot": "UniProt",
        }
        datasets = [
            {
                "name": f"{split.title()}Val_{test_suffixes.get(test_split, test_split.title())}",
                "train": split,
                "val": "val",
                "test": test_split,
            }
            for test_split in datasets_template
        ]
        for seed in seeds:
            print(
                f"[Training Dataset Ablation] {condition['display_name']} | seed={seed} | "
                f"N={condition['training_sample_count']:,}"
            )
            run_context = dict(context)
            run_context.update({
                "datasets": datasets,
                "training_split": split,
                "geometry_context": split,
                "training_sample_ids": condition.get("training_sample_ids"),
                "training_subset_signature": condition["sample_signature"],
                "stage_root": str(output_root),
                "token_window_radius": int(fixed_model_config["k_radius"]),
                "token_only": True,
                "token_modalities": ("dna", "protein"),
                "allow_batch_size_resume": True,
                "input_data_signature": {
                    **dict(context.get("input_data_signature", {})),
                    "training_cohort": condition["key"],
                    "training_split": split,
                    "training_sample_count": int(condition["training_sample_count"]),
                    "training_sample_signature": condition["sample_signature"],
                    "geometry_context": split,
                    "validation_split": "val",
                    "test_splits": list(datasets_template),
                    "embedding_representation": "ref_alt_concat",
                },
            })
            experiment_key = f"{condition['key']}_seed_{seed}"
            model_config = {**fixed_model_config, "seed": seed}
            result = run_architecture_experiment(
                context=run_context,
                experiment_config=model_config,
                model_builder=build_cross_attention_model,
                architecture="CrossAttention_ThreeModalities",
                variant=f"{condition['key']}_ref_alt_concat_window_k{fixed_model_config['k_radius']}",
                experiment_key=experiment_key,
                source="module06_training_dataset_ablation",
                display_fn=lambda *_args, **_kwargs: None,
                manifest_architecture="training_dataset_ablation_cross_attention_3modal_ref_alt_concat_v1",
                embedding_representation="ref_alt_concat",
                evaluate_test=True,
            )
            metadata = {
                "Training_Cohort": condition["key"],
                "Training_Cohort_Name": condition["display_name"],
                "Training_Split": split,
                "Training_Sampling_Mode": condition["sampling_mode"],
                "Training_N_Samples": int(condition["training_sample_count"]),
                "Training_Benign_N": int(condition["training_class_counts"].get("0", 0)),
                "Training_Pathogenic_N": int(condition["training_class_counts"].get("1", 0)),
                "Training_Sample_Signature": condition["sample_signature"],
            }
            incoming_validation = result["validation_metrics"].copy()
            incoming_test = result["test_metrics"].copy()
            incoming_predictions = result["test_predictions"].copy()
            for frame in (incoming_validation, incoming_test, incoming_predictions):
                for key, value in metadata.items():
                    frame[key] = value
                frame["Seed"] = seed
            validation = _upsert_frame(
                validation, incoming_validation,
                ["Training_Cohort", "Seed", "Target_Config"],
            )
            test = _upsert_frame(
                test, incoming_test,
                ["Training_Cohort", "Seed", "Target_Config", "Dataset"],
            )
            predictions = _upsert_frame(
                predictions, incoming_predictions,
                ["Training_Cohort", "Seed", "Target_Config", "Dataset", "Variant_ID"],
            )
            save_progress_csv(validation.to_dict("records"), validation_path)
            save_progress_csv(test.to_dict("records"), test_path)
            save_progress_csv(predictions.to_dict("records"), prediction_path)
            display({
                "condition": condition["display_name"],
                "seed": seed,
                "validation": incoming_validation,
                "tests": incoming_test,
                "checkpoint_dir": result["output_dir"],
            })

    return {
        "output_root": str(output_root),
        "validation_metrics": validation,
        "test_metrics": test,
        "test_predictions": predictions,
        "validation_path": str(validation_path),
        "test_path": str(test_path),
        "prediction_path": str(prediction_path),
    }


def summarize_training_dataset_ablation(*, output_root, display_fn=None):
    """Write seed-level and per-cohort mean/std summaries, retaining full profiles."""
    output_root = Path(output_root)
    display = display_fn or print
    validation_path = output_root / "seed_level_validation_metrics.csv"
    test_path = output_root / "seed_level_test_metrics.csv"
    if not validation_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Training-dataset ablation seed-level metrics are incomplete.")
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    if validation.empty or test.empty:
        raise ValueError("No completed training-dataset ablation metrics are available.")

    metric_groups = [name for name in (*REFERENCE_METRICS, *PROFILE_METRICS) if name in test.columns]
    for name in metric_groups:
        test[name] = pd.to_numeric(test[name], errors="coerce")
    summary = test.groupby(
        ["Training_Cohort", "Training_Cohort_Name", "Training_N_Samples", "Dataset"],
        dropna=False,
    )[metric_groups].agg(["mean", "std"])
    summary = summary.reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in summary.columns.to_flat_index()
    ]
    summary_path = output_root / "test_mean_std_by_training_cohort_dataset.csv"
    save_progress_csv(summary.to_dict("records"), summary_path)

    val_metric_groups = [name for name in VALIDATION_METRICS if name in validation.columns]
    for name in val_metric_groups:
        validation[name] = pd.to_numeric(validation[name], errors="coerce")
    validation_summary = validation.groupby(
        ["Training_Cohort", "Training_Cohort_Name", "Training_N_Samples"],
        dropna=False,
    )[val_metric_groups].agg(["mean", "std"])
    validation_summary = validation_summary.reset_index()
    validation_summary.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in validation_summary.columns.to_flat_index()
    ]
    validation_summary_path = output_root / "validation_mean_std_by_training_cohort.csv"
    save_progress_csv(validation_summary.to_dict("records"), validation_summary_path)

    completion = test.groupby("Training_Cohort", dropna=False).agg(
        Completed_Seed_Dataset_Rows=("Dataset", "size"),
        Seeds=("Seed", "nunique"),
        Test_Datasets=("Dataset", "nunique"),
    ).reset_index()
    completion_path = output_root / "completion_inventory.csv"
    save_progress_csv(completion.to_dict("records"), completion_path)

    display("Test metrics: mean ± standard deviation across seeds, per cohort and test dataset")
    display(summary.sort_values(["Dataset", "MCC_mean"], ascending=[True, False]))
    display("Validation metrics: mean ± standard deviation across seeds")
    display(validation_summary.sort_values("Val_MCC_mean", ascending=False))
    display("Completion inventory")
    display(completion)
    return {
        "test_summary": summary,
        "validation_summary": validation_summary,
        "completion_inventory": completion,
        "test_summary_path": str(summary_path),
        "validation_summary_path": str(validation_summary_path),
        "completion_path": str(completion_path),
    }
