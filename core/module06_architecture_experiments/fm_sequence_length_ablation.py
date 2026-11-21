"""Shared utilities for Step 2 FM sequence-length ablations in Notebook 15."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import copy
from pathlib import Path

import pandas as pd
import torch

from core.module05v2_fusion_classifier.resume_utils import save_progress_csv

from core.module06_architecture_experiments.fm_ablation_preparation import build_model_configs


DNA_LENGTHS = (301, 601, 1001, 2001, 3501)
PROTEIN_LENGTHS = (31, 101, 201, 501, 1001)
SEEDS = (5, 6, 7, 37, 42)
DNA_MODEL_BASE = "nt_v1_500m"
PROTEIN_MODEL_BASE = "esm1b_650m"
SOURCE_TRIAL_ID = "params_27321c1de5fe2d03"
SOURCE_EXPERIMENT = "step1_optuna_nt_v1_dna1001_esm1b_prot501_5seed"


def build_selected_model_configs(dna_lengths=DNA_LENGTHS, protein_lengths=PROTEIN_LENGTHS):
    """Build only the selected NTv1/ESM1b token-cache configs for all lengths."""
    configs = []
    for length in dna_lengths:
        if int(length) == 3501:
            candidates = build_model_configs(dna_length=10001, protein_length=501)
            selected = next(
                item for item in candidates
                if item["name"].startswith(f"{DNA_MODEL_BASE}_")
            )
            selected.update({
                "name": f"{DNA_MODEL_BASE}_dna_3501",
                "sequence_length": 3501,
                "source_sequence_length": 10001,
                "center_crop_length": 3501,
            })
            configs.append(selected)
        else:
            candidates = build_model_configs(dna_length=int(length), protein_length=501)
            configs.extend(item for item in candidates if item["name"].startswith(f"{DNA_MODEL_BASE}_"))
    for length in protein_lengths:
        candidates = build_model_configs(dna_length=1001, protein_length=int(length))
        configs.extend(item for item in candidates if item["name"].startswith(f"{PROTEIN_MODEL_BASE}_"))
    names = [item["name"] for item in configs]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate model-length cache config generated.")
    return configs


def load_source_seed_configs(source_root, seeds=SEEDS):
    """Read the user-selected per-seed manifests and verify their checkpoints."""
    root = Path(source_root)
    configs = {}
    manifests = {}
    for seed in seeds:
        stage = root / "trials" / f"{SOURCE_TRIAL_ID}_seed_{int(seed)}"
        manifest_path = stage / "_resume_manifest.json"
        checkpoint_path = stage / "checkpoints" / "B_with_geometry" / "best_model.pth"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Source hyperparameter manifest is missing: {manifest_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Source best checkpoint is missing: {checkpoint_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved = payload.get("config", {})
        hyper_key = f"{SOURCE_EXPERIMENT}_config"
        config = saved.get(hyper_key)
        if not isinstance(config, dict):
            raise KeyError(f"{hyper_key} not found in {manifest_path}")
        if int(config.get("seed", -1)) != int(seed):
            raise ValueError(f"Seed mismatch in {manifest_path}: {config.get('seed')} != {seed}")
        if saved.get("dna_model", {}).get("name") != "nt_v1_500m_dna_1001":
            raise ValueError(f"Unexpected DNA model recorded in {manifest_path}")
        if saved.get("protein_model", {}).get("name") != "esm1b_650m_prot_501":
            raise ValueError(f"Unexpected protein model recorded in {manifest_path}")
        configs[int(seed)] = dict(config)
        manifests[int(seed)] = {
            "manifest": str(manifest_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "stage_dir": str(stage.resolve()),
        }

    reference = {key: value for key, value in configs[int(seeds[0])].items() if key != "seed"}
    for seed, config in configs.items():
        comparable = {key: value for key, value in config.items() if key != "seed"}
        if comparable != reference:
            raise ValueError(f"Source best hyperparameters differ for seed {seed}.")
    return configs, manifests


def verify_sequence_columns(split_configs, model_configs):
    """Check requested sequence columns exist in every source parquet schema."""
    import pyarrow.parquet as pq

    rows = []
    for split in split_configs:
        schema = pq.read_schema(split["input_path"])
        columns = set(schema.names)
        for model in model_configs:
            missing = [column for column in (model["ref_col"], model["alt_col"]) if column not in columns]
            rows.append({
                "Split": split["name"],
                "Model": model["name"],
                "Sequence_Length": int(model["sequence_length"]),
                "Source_Sequence_Length": int(model.get("source_sequence_length", model["sequence_length"])),
                "Ref_Column": model["ref_col"],
                "Alt_Column": model["alt_col"],
                "Status": "missing columns: " + ", ".join(missing) if missing else "ready",
            })
    missing_rows = [row for row in rows if row["Status"] != "ready"]
    if missing_rows:
        raise KeyError("Missing sequence columns in input parquet: " + repr(missing_rows[:10]))
    return rows


def build_center_geometry_embeddings(*, model_configs, split_configs, token_cache_index,
                                     output_root):
    """Derive exact Module 2 center vectors from token caches for geometry only.

    These artifacts are written to a separate namespace and are not the pooled
    input cache used by the Step 3 center/mean/CLS comparison.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for model in model_configs:
        model_name = str(model["name"])
        for split in split_configs:
            split_name = str(split["name"])
            try:
                cache_entry = token_cache_index["models"][model_name][split_name]
            except KeyError as exc:
                raise KeyError(f"Token cache index missing {model_name}/{split_name}") from exc
            if cache_entry.get("status") != "complete":
                raise RuntimeError(f"Token cache is incomplete: {model_name}/{split_name}")
            cache_dir = Path(cache_entry["path"])
            cache_manifest_path = Path(cache_entry["manifest"])
            cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            n_samples = int(cache_manifest["n_samples"])
            n_shards = int(cache_manifest["n_shards"])
            hidden_size = int(cache_manifest["hidden_size"])
            source_signature = hashlib.sha256(
                json.dumps(cache_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            destination = output_root / split_name / f"{model_name}_center.pt"
            signature_path = destination.with_suffix(".manifest.json")
            expected_signature = {
                "version": 1,
                "model_name": model_name,
                "sequence_length": int(model["sequence_length"]),
                "split": split_name,
                "token_cache_manifest_sha256": source_signature,
                "pooling": "hidden_state_at_variant_token_index",
                "llr": "zero_placeholder_not_used_by_selected_geometry_features",
                "n_samples": n_samples,
                "hidden_size": hidden_size,
            }
            if destination.is_file() and signature_path.is_file():
                try:
                    saved_signature = json.loads(signature_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    saved_signature = None
                if saved_signature == expected_signature:
                    payload = torch.load(destination, map_location="cpu", weights_only=False)
                    if (
                        len(payload.get("metadata", [])) == n_samples
                        and tuple(payload["E_ref"].shape) == (n_samples, hidden_size)
                        and tuple(payload["E_alt"].shape) == (n_samples, hidden_size)
                    ):
                        print(f"[GEOM EMBEDDING RESUME] {model_name}/{split_name}")
                        index_rows.append({
                            "Model": model_name, "Split": split_name,
                            "Status": "reused", "Path": str(destination),
                            "Samples": n_samples,
                        })
                        continue

            refs, alts, metadata = [], [], []
            for shard_index in range(n_shards):
                shard_path = cache_dir / f"shard_{shard_index:06d}.pt"
                if not shard_path.is_file():
                    raise FileNotFoundError(f"Token cache shard is missing: {shard_path}")
                shard = torch.load(shard_path, map_location="cpu", weights_only=False)
                ref_hidden = shard["token_hidden_ref"]
                alt_hidden = shard["token_hidden_alt"]
                ref_index = shard["variant_token_index_ref"].to(torch.long)
                alt_index = shard["variant_token_index_alt"].to(torch.long)
                ref_mask = shard["attention_mask_ref"].bool()
                alt_mask = shard["attention_mask_alt"].bool()
                row_count = len(shard["variant_ids"])
                rows = torch.arange(row_count)
                if (
                    ref_hidden.shape[0] != row_count
                    or alt_hidden.shape[0] != row_count
                    or ref_index.shape != (row_count,)
                    or alt_index.shape != (row_count,)
                ):
                    raise ValueError(f"Malformed token-cache shard: {shard_path}")
                if (
                    (ref_index < 0).any() or (ref_index >= ref_hidden.shape[1]).any()
                    or (alt_index < 0).any() or (alt_index >= alt_hidden.shape[1]).any()
                    or not ref_mask[rows, ref_index].all()
                    or not alt_mask[rows, alt_index].all()
                ):
                    raise ValueError(f"Variant center is invalid in token-cache shard: {shard_path}")
                refs.append(ref_hidden[rows, ref_index].contiguous().to(torch.float16))
                alts.append(alt_hidden[rows, alt_index].contiguous().to(torch.float16))
                metadata.extend(str(value) for value in shard["variant_ids"])
                del shard

            if len(metadata) != n_samples:
                raise ValueError(
                    f"Center-vector sample count mismatch for {model_name}/{split_name}: "
                    f"{len(metadata)} != {n_samples}"
                )
            payload = {
                "metadata": metadata,
                "E_ref": torch.cat(refs, dim=0),
                "E_alt": torch.cat(alts, dim=0),
                # Module 3 expects this field; Step 2 explicitly excludes LLR.
                "llr": torch.zeros((n_samples, 1), dtype=torch.float32),
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix=f"{destination.stem}_", suffix=".tmp", dir=destination.parent
            )
            os.close(fd)
            try:
                torch.save(payload, temporary)
                os.replace(temporary, destination)
                signature_path.write_text(
                    json.dumps(expected_signature, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            index_rows.append({
                "Model": model_name, "Split": split_name,
                "Status": "created", "Path": str(destination),
                "Samples": n_samples,
            })
            del refs, alts, metadata, payload
        
    return pd.DataFrame(index_rows)


def run_sequence_length_ablation(
    *, pipeline_factory, target_configs, datasets, model_configs, token_cache_index,
    source_configs, source_manifests, step_root, device="cuda", token_batch_size=64,
    display_fn=print, dna_lengths=None, protein_lengths=None, seeds=None,
):
    """Run/resume selected length pairs and seeds; Step 1 supplies hyperparameters only."""
    from core.module06_architecture_experiments.architecture_experiment_runner import (
        run_architecture_experiment,
    )
    from core.module06_architecture_experiments.cross_attention_3modal_experiment import (
        _builder as model_builder,
    )

    root = Path(step_root)
    run_root = root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    dna_lengths = tuple(DNA_LENGTHS if dna_lengths is None else map(int, dna_lengths))
    protein_lengths = tuple(PROTEIN_LENGTHS if protein_lengths is None else map(int, protein_lengths))
    seeds = tuple(SEEDS if seeds is None else map(int, seeds))
    by_name = {model["name"]: model for model in model_configs}
    dna_models = {
        length: by_name[f"{DNA_MODEL_BASE}_dna_{length}"] for length in dna_lengths
    }
    protein_models = {
        length: by_name[f"{PROTEIN_MODEL_BASE}_prot_{length}"] for length in protein_lengths
    }
    missing_seed_configs = sorted(set(seeds) - set(map(int, source_configs)))
    missing_seed_manifests = sorted(set(seeds) - set(map(int, source_manifests)))
    if missing_seed_configs or missing_seed_manifests:
        raise KeyError(
            "Missing source hyperparameter configs/manifests for seeds: "
            f"configs={missing_seed_configs}, manifests={missing_seed_manifests}"
        )
    progress_path = root / "step2_run_progress.csv"
    progress = pd.read_csv(progress_path).to_dict("records") if progress_path.is_file() else []
    n_runs = 0
    for dna_length in dna_lengths:
        for protein_length in protein_lengths:
            dna_model = dna_models[dna_length]
            protein_model = protein_models[protein_length]
            run_targets = copy.deepcopy(target_configs)
            run_targets[0]["geometry_dna_name"] = dna_model["name"]
            run_targets[0]["geometry_protein_name"] = protein_model["name"]
            for seed in seeds:
                experiment_key = (
                    f"step2_nt_v1_dna{dna_length}_esm1b_prot{protein_length}_seed{seed}"
                )
                config = dict(source_configs[int(seed)])
                config["seed"] = int(seed)
                pipeline = pipeline_factory(seed, dna_model, protein_model)
                context = {
                    "pipeline": pipeline,
                    "TARGET_CONFIGS": run_targets,
                    "datasets": datasets,
                    "DNA_MODEL": dna_model,
                    "PROTEIN_MODEL": protein_model,
                    "DEVICE": device,
                    "token_embedding_cache_index": token_cache_index,
                    "token_batch_size": int(token_batch_size),
                    "token_only": True,
                    "stage_root": str(run_root),
                    "input_data_signature": {
                        "stage": "step2_fm_sequence_length_ablation",
                        "dna_sequence_length": int(dna_length),
                        "protein_sequence_length": int(protein_length),
                        "dna_token_model": dna_model["model_id"],
                        "protein_token_model": protein_model["model_id"],
                        "geometry_source_dna": run_targets[0]["geometry_dna_name"],
                        "geometry_source_protein": run_targets[0]["geometry_protein_name"],
                        "geometry_feature_names": run_targets[0]["geom_feature_names"],
                        "geometry_context": "train3",
                        "geometry_root": run_targets[0].get("geometry_root"),
                        "token_window": "variant-centered ref/alt, k_radius=16, length=33",
                        "pooled_fm_inputs": False,
                        "source_hyperparameter_manifest": source_manifests[int(seed)]["manifest"],
                    },
                }
                result = run_architecture_experiment(
                    context=context,
                    experiment_config=config,
                    model_builder=model_builder,
                    architecture="CrossAttention_ThreeModalities_FM_SequenceLength",
                    variant="ref_alt_concat_window_33_sequence_length_ablation",
                    experiment_key=experiment_key,
                    source="fm_ablation_step2",
                    display_fn=display_fn,
                    manifest_architecture=(
                        "cross_attention_3modal_ref_alt_concat_window33_sequence_length_ablation"
                    ),
                    embedding_representation="ref_alt_concat",
                    evaluate_test=True,
                )
                progress = [row for row in progress if row.get("Experiment_Key") != experiment_key]
                progress.append({
                    "Experiment_Key": experiment_key,
                    "DNA_Length": int(dna_length),
                    "Protein_Length": int(protein_length),
                    "Seed": int(seed),
                    "Status": "complete",
                    "Output_Dir": result["output_dir"],
                    "Source_Hyperparameter_Manifest": source_manifests[int(seed)]["manifest"],
                    "Updated_At": pd.Timestamp.utcnow().isoformat(),
                })
                save_progress_csv(progress, progress_path)
                n_runs += 1
    return {"status": "complete", "new_or_resumed_runs": n_runs,
            "progress_path": str(progress_path.resolve()), "output_root": str(root.resolve())}


def _read_metric_files(paths, source_filter=None):
    frames = []
    for path in sorted(paths):
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if frame.empty:
            continue
        if source_filter is not None and "Source" in frame:
            frame = frame.loc[frame["Source"].astype(str).eq(source_filter)]
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def summarize_sequence_length_ablation(*, step_root, seeds=SEEDS):
    """Summarize only the newly trained, geometry-matched Step 2 runs."""
    root = Path(step_root)
    validation_paths = list((root / "runs").glob("step2_*/*_validation_metrics.csv"))
    test_paths = list((root / "runs").glob("step2_*/*_test_metrics.csv"))
    validation = _read_metric_files(validation_paths, "fm_ablation_step2")
    test = _read_metric_files(test_paths, "fm_ablation_step2")
    if validation.empty or test.empty:
        raise FileNotFoundError(f"No completed Step 2 metrics found under {root / 'runs'}")

    for frame in (validation, test):
        frame["Seed"] = pd.to_numeric(frame["Seed"], errors="coerce").astype("Int64")
        frame["DNA_Length"] = frame["DNA_Model"].astype(str).str.extract(r"_dna_(\d+)$")[0].astype(int)
        frame["Protein_Length"] = frame["Prot_Model"].astype(str).str.extract(r"_prot_(\d+)$")[0].astype(int)
        frame["Sequence_Ablation_Source"] = frame.get(
            "Sequence_Ablation_Source", pd.Series(index=frame.index, dtype=object)
        ).fillna("step2_run")
    # `seeds` defines the reporting cohort. Extra exploratory seed folders can
    # remain on disk without silently inflating the reported mean/std.
    selected_seeds = set(map(int, seeds))
    validation = validation.loc[validation["Seed"].isin(selected_seeds)].copy()
    test = test.loc[test["Seed"].isin(selected_seeds)].copy()
    validation = validation.drop_duplicates(["DNA_Length", "Protein_Length", "Seed", "Target_Config"], keep="last")
    test = test.drop_duplicates(["DNA_Length", "Protein_Length", "Seed", "Target_Config", "Dataset"], keep="last")
    expected_test_datasets = {
        "Train3Val_Test", "Train3Val_ClinVarHQ", "Train3Val_ProteinGym", "Train3Val_UniProt"
    }
    validation.to_csv(root / "step2_validation_metrics_by_combo_seed.csv", index=False)
    test.to_csv(root / "step2_test_metrics_by_combo_dataset_seed.csv", index=False)

    def mean_std(frame, keys, excluded):
        metrics = [
            column for column in frame.select_dtypes(include="number").columns
            if column not in excluded and column != "Seed"
        ]
        grouped = frame.groupby(keys, dropna=False)[metrics].agg(["mean", "std"])
        grouped.columns = [f"{name}_{stat}" for name, stat in grouped.columns]
        grouped["N_Seeds"] = frame.groupby(keys, dropna=False)["Seed"].nunique()
        return grouped.reset_index()

    validation_summary = mean_std(
        validation, ["DNA_Length", "Protein_Length"],
        {"DNA_Length", "Protein_Length", "N_Seeds"},
    ).sort_values("Val_MCC_mean", ascending=False)
    test_summary = mean_std(
        test, ["DNA_Length", "Protein_Length", "Dataset"],
        {"DNA_Length", "Protein_Length", "Dataset", "N_Seeds"},
    )
    validation_summary.to_csv(root / "step2_validation_mean_std_by_combo.csv", index=False)
    test_summary.to_csv(root / "step2_test_mean_std_by_combo_dataset.csv", index=False)
    expected_seeds = set(map(int, seeds))
    completeness = {}
    for combo, combo_rows in test.groupby(["DNA_Length", "Protein_Length"]):
        seed_complete = combo_rows.groupby("Seed")["Dataset"].agg(
            lambda values: set(values) == expected_test_datasets
        )
        available_seeds = set(map(int, seed_complete.index))
        completeness[combo] = expected_seeds.issubset(available_seeds) and bool(
            seed_complete.reindex(sorted(expected_seeds), fill_value=False).all()
        )
    inventory = validation.groupby(["DNA_Length", "Protein_Length"]).agg(
        Validation_Seeds=("Seed", "nunique"),
    ).join(test.groupby(["DNA_Length", "Protein_Length"]).agg(
        Test_Seeds=("Seed", "nunique"), Test_Rows=("Dataset", "count")
    ), how="outer").reset_index()
    inventory["All_Seeds_Have_All_Four_Test_Sets"] = [
        bool(completeness.get((row.DNA_Length, row.Protein_Length), False))
        for row in inventory.itertuples(index=False)
    ]
    inventory.to_csv(root / "step2_combo_inventory.csv", index=False)
    return {
        "validation_by_seed": validation,
        "test_by_seed": test,
        "validation_mean_std": validation_summary,
        "test_mean_std_by_dataset": test_summary,
        "inventory": inventory,
        "output_root": str(root.resolve()),
    }

