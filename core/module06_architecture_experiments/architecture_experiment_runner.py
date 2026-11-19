"""Shared train, validation, checkpoint, and resume engine for Module 6 models."""
from __future__ import annotations

import json
import hashlib
import os
import re
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from core.module05v2_fusion_classifier.evaluator_profiler import FusionEvaluatorProfiler
from core.module05v2_fusion_classifier.optuna_search import (
    optuna_trial_plan,
    tell_trial,
)
from core.module05v2_fusion_classifier.resume_utils import (
    ensure_stage_signature,
    replace_file_with_retry,
    save_progress_csv,
    upsert_record,
)

PROFILE_METRIC_COLUMNS = (
    "FM_GFLOPs/Sample",
    "FM_Latency_ms/Sample",
    "Geom_GFLOPs/Sample",
    "Geom_Latency_ms/Sample",
    "Classifier_GFLOPs/Sample",
    "Classifier_Latency_ms/Sample",
    "Classifier_Total_Params",
    "Classifier_Param_Mem_MB",
    "Classifier_Peak_VRAM_MB",
    "E2E_GFLOPs/Sample",
    "E2E_Latency_ms/Sample",
    "E2E_Total_Params",
    "E2E_Param_Mem_MB",
    "Max_Peak_VRAM_MB",
)


class _TokenOnlyPlaceholderDataset(Dataset):
    """Keep Module 5 metadata/features while supplying unused pooled placeholders."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.v_dna_tensor = torch.zeros((len(base_dataset), 1), dtype=torch.float32)
        self.v_prot_tensor = torch.zeros((len(base_dataset), 1), dtype=torch.float32)

    def __len__(self):
        return len(self.base_dataset)

    def __getattr__(self, name):
        if name == "base_dataset":
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    def __getitem__(self, index):
        sample = dict(self.base_dataset[index])
        sample["v_dna"] = self.v_dna_tensor[index]
        sample["v_prot"] = self.v_prot_tensor[index]
        return sample


def _architecture_loader(
    pipeline, context, target, split_name, pooling, dna_model, protein_model,
    is_train, geometry_context, seed, embedding_representation,
):
    """Build a pooled or token-only loader with a consistent Module 6 batch API."""
    token_only = bool(context.get("token_only", False))
    active_modalities = list(target["active_modalities"])
    if token_only:
        active_modalities = [
            modality for modality in active_modalities
            if str(modality).lower() not in {"dna", "prot"}
        ]
    loader = pipeline._get_dataloader(
        split_name, pooling, dna_model["name"], protein_model["name"],
        active_modalities, is_train, target.get("geom_feature_names"),
        geometry_context, seed, embedding_representation,
        geometry_dna_name=target.get("geometry_dna_name"),
        geometry_protein_name=target.get("geometry_protein_name"),
        load_pooled_embeddings=not token_only,
    )
    if token_only:
        dataset = _TokenOnlyPlaceholderDataset(loader.dataset)
        loader = DataLoader(
            dataset,
            batch_size=loader.batch_size,
            shuffle=is_train,
            num_workers=0,
            pin_memory=loader.pin_memory,
            generator=loader.generator,
        )
    return loader



def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state(loader):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": (
            loader.generator.get_state()
            if getattr(loader, "generator", None) is not None
            else None
        ),
    }


def _restore_rng_state(state, loader):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if (
        state.get("loader_generator") is not None
        and getattr(loader, "generator", None) is not None
    ):
        loader.generator.set_state(state["loader_generator"].cpu())


def _cpu_state_dict(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _atomic_torch_save(payload, destination):
    destination = Path(destination)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        replace_file_with_retry(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _move(batch, key, device, required=False):
    value = batch.get(key)
    if value is None and required:
        raise KeyError(f"Required architecture batch input is missing: {key}")
    return value.to(device, non_blocking=True) if value is not None else None


def _forward(model, batch, device):
    inputs = [
        _move(batch, "v_dna", device, required=True),
        _move(batch, "v_prot", device, required=True),
        _move(batch, "bio_features", device, required=True),
        _move(batch, "geom_features", device),
    ]
    # Token-level Module 6 models opt into these additional inputs. Existing
    # pooled-embedding architectures retain their original four-argument API.
    for key in (
        "token_dna", "token_dna_mask", "token_dna_variant_index",
        "token_protein", "token_protein_mask", "token_protein_variant_index",
    ):
        if key in batch:
            inputs.append(_move(batch, key, device))
    return model(*inputs)


def _attach_token_cache(loader, context, split_name, is_train=False, window_radius=None):
    """Wrap a pooled Module 5 loader with lazy token-cache lookups when requested."""
    token_index = context.get("token_embedding_cache_index")
    if token_index is None:
        return loader
    if window_radius is not None:
        from core.module06_architecture_experiments.window_token_cache import (
            add_window_cache_to_loader,
        )
        return add_window_cache_to_loader(
            loader=loader,
            token_index=token_index,
            split_name=split_name,
            dna_model_name=context["DNA_MODEL"]["name"],
            protein_model_name=context["PROTEIN_MODEL"]["name"],
            is_train=is_train,
            batch_size=context.get("token_batch_size"),
            radius=window_radius,
        )
    from core.module06_architecture_experiments.token_topk_experiment import (
        add_token_cache_to_loader,
    )
    return add_token_cache_to_loader(
        loader=loader,
        token_index=token_index,
        split_name=split_name,
        dna_model_name=context["DNA_MODEL"]["name"],
        protein_model_name=context["PROTEIN_MODEL"]["name"],
        is_train=is_train,
        batch_size=context.get("token_batch_size"),
    )


def _collect(model, loader, device, criterion=None, profiler=None, progress_desc=None):
    model.eval()
    labels, probabilities = [], []
    loss_total, n_total = 0.0, 0
    if profiler is not None:
        profiler.reset_memory_stats()
    batches = (
        tqdm(loader, desc=progress_desc, unit="batch", leave=False)
        if progress_desc else loader
    )
    with torch.no_grad():
        for batch in batches:
            if profiler is not None:
                profiler.tic_inference()
            logits = _forward(model, batch, device)
            if profiler is not None:
                profiler.toc_inference()
            y = _move(batch, "label", device, required=True).reshape_as(logits)
            if criterion is not None:
                loss_total += float(criterion(logits, y).item()) * int(y.shape[0])
                n_total += int(y.shape[0])
            labels.extend(y.reshape(-1).cpu().numpy())
            probabilities.extend(torch.sigmoid(logits).reshape(-1).cpu().numpy())
    if profiler is not None:
        profiler.finalize_fusion_inference_profiling(len(labels))
    mean_loss = loss_total / max(n_total, 1) if criterion is not None else np.nan
    return mean_loss, np.asarray(labels, dtype=np.int32), np.asarray(probabilities, dtype=np.float32)


def _metrics(pipeline, labels, probabilities):
    threshold, mcc = pipeline._select_validation_threshold(labels, probabilities)
    predictions = pipeline._threshold_predictions(probabilities, threshold)
    values = FusionEvaluatorProfiler.compute_metrics(labels, probabilities, predictions)
    return float(threshold), float(mcc), values


def _base_geometry_model_name(model_name):
    return re.sub(r"_(?:dna|prot)_\d+$", "", str(model_name))


def _get_test_profile_metrics(profiler, dna_model_name, protein_model_name, active_modalities, split_name):
    """Combine FM, geometry, and measured classifier profiling for one test split."""
    stats = profiler.get_e2e_profiling(
        dna_model_name=dna_model_name,
        prot_model_name=protein_model_name,
        active_modalities=active_modalities,
        split_name=split_name,
        pooling="center",
    )

    # Geometry profiling stores family-level keys (for example
    # ``nt_v1_500m_center``), whereas embedding files include sequence length.
    # Try the full key first, then map the model name to its family key.
    geom_data = profiler._load_profile_json(profiler.geom_profile_json, "geometry", required=False)
    active_mods = {str(mod).lower() for mod in (active_modalities or [])}
    geom_stats = []
    if "geom" in active_mods:
        for model_name in (dna_model_name, protein_model_name):
            exact_key = f"{model_name}_center"
            item = profiler._resolve_nested_stats(geom_data, exact_key, split_name)
            if not item:
                family_key = f"{_base_geometry_model_name(model_name)}_center"
                item = profiler._resolve_nested_stats(geom_data, family_key, split_name)
            geom_stats.append(profiler._normalize_geom_stats(item))

    geom_gflops = sum(float(item.get("gflops_per_sample", 0.0)) for item in geom_stats)
    geom_latency = sum(float(item.get("inference_time_ms", 0.0)) for item in geom_stats)
    geom_peak = max((float(item.get("peak_memory_mb", 0.0)) for item in geom_stats), default=0.0)
    stats["Geom_GFLOPs/Sample"] = round(geom_gflops, 4)
    stats["Geom_Latency_ms/Sample"] = round(geom_latency, 2)
    stats["E2E_GFLOPs/Sample"] = round(
        float(stats.get("FM_GFLOPs/Sample", 0.0))
        + geom_gflops
        + float(stats.get("Classifier_GFLOPs/Sample", 0.0)),
        4,
    )
    stats["E2E_Latency_ms/Sample"] = round(
        float(stats.get("FM_Latency_ms/Sample", 0.0))
        + geom_latency
        + float(stats.get("Classifier_Latency_ms/Sample", 0.0)),
        2,
    )
    stats["Max_Peak_VRAM_MB"] = round(max(
        float(stats.get("Max_Peak_VRAM_MB", 0.0)), geom_peak
    ), 2)
    # Classifier_XGB_Model_Mem_MB is intentionally omitted: Module 6 models
    # handled by this runner are PyTorch classifiers, not XGBoost.
    return {key: stats.get(key, np.nan) for key in PROFILE_METRIC_COLUMNS}


def run_architecture_experiment(
    context, experiment_config, model_builder, architecture, variant, experiment_key,
    source, display_fn=None, manifest_architecture=None, trial_name=None,
    embedding_representation="delta", evaluate_test=True,
):
    """Run one architecture on configs A/B and the configured test datasets.

    Completed train/validation checkpoints and test rows are reused on resume.
    Interrupted training resumes at the next completed-epoch boundary, restoring
    model, optimizer, scheduler, early-stopping, data-loader, and RNG states.
    The stage manifest rejects resuming with changed model or training settings.
    """
    pipeline = context["pipeline"]
    targets = context["TARGET_CONFIGS"]
    datasets = context["datasets"]
    dna_model = context["DNA_MODEL"]
    protein_model = context["PROTEIN_MODEL"]
    device = torch.device(context.get("DEVICE", pipeline.device))
    config = dict(experiment_config)
    display = display_fn or print
    window_radius = (
        int(config["k_radius"])
        if (
            source == "fm_ablation_step1"
            and variant == "ref_alt_concat_window_33_fixed_hparams"
        ) or (
            source == "fm_ablation_step2"
            and variant == "ref_alt_concat_window_33_sequence_length_ablation"
        )
        else None
    )

    stage_root = Path(context.get("stage_root", Path(pipeline.batch_run_dir) / "module06_architecture_experiments"))
    stage_dir = stage_root / experiment_key
    if trial_name is not None:
        stage_dir = stage_dir / "trials" / str(trial_name)
    checkpoint_root = stage_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    validation_path = stage_dir / f"{experiment_key}_validation_metrics.csv"
    test_path = stage_dir / f"{experiment_key}_test_metrics.csv"
    history_path = stage_dir / f"{experiment_key}_epoch_history.csv"
    predictions_path = stage_dir / f"{experiment_key}_test_predictions.csv"

    signature = {
        "architecture": manifest_architecture or architecture,
        f"{experiment_key}_config": config,
        "pipeline_config": pipeline.config,
        "target_configs": targets,
        "datasets": datasets,
        "dna_model": dna_model,
        "protein_model": protein_model,
        "pooling": "center",
        "embedding_representation": embedding_representation,
        "checkpoint_selection": "validation_mcc_optimized_threshold_v2",
    }
    if context.get("input_data_signature") is not None:
        signature["input_data_signature"] = context["input_data_signature"]
    if experiment_key != "film":
        signature["variant"] = variant
    ensure_stage_signature(stage_dir, f"module06_{experiment_key}", signature)

    validation_rows = pd.read_csv(validation_path).to_dict("records") if validation_path.exists() else []
    test_rows = pd.read_csv(test_path).to_dict("records") if test_path.exists() else []
    history_rows = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    prediction_rows = pd.read_csv(predictions_path).to_dict("records") if predictions_path.exists() else []

    for target in targets:
        target_name = str(target["name"])
        model_dir = checkpoint_root / target_name
        model_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint = model_dir / "best_model.pth"
        training_state_path = model_dir / "training_state.pth"
        validation_row = next(
            (row for row in validation_rows if str(row.get("Target_Config")) == target_name), None
        )
        reuse_completed_model = validation_row is not None and best_checkpoint.exists()

        # Seed before constructing the model so its initial weights match the
        # recorded seed. On resume, the saved RNG state is restored afterward.
        _seed_everything(int(config["seed"]))
        loader_start = time.perf_counter()
        print(f"[{architecture}] Preparing train dataset and token-cache loader...")
        train_loader = _architecture_loader(
            pipeline, context, target, "train3", "center", dna_model, protein_model,
            True, "train3", int(config["seed"]), embedding_representation,
        )
        train_loader = _attach_token_cache(
            train_loader, context, "train3", is_train=True,
            window_radius=None if reuse_completed_model else window_radius,
        )
        print(
            f"[{architecture}] Train loader ready: {len(train_loader)} batches "
            f"in {time.perf_counter() - loader_start:.1f}s"
        )
        if not (window_radius is not None and reuse_completed_model):
            loader_start = time.perf_counter()
            print(f"[{architecture}] Preparing validation dataset and token-cache loader...")
            val_loader = _architecture_loader(
                pipeline, context, target, "val", "center", dna_model, protein_model,
                False, "train3", int(config["seed"]), embedding_representation,
            )
            val_loader = _attach_token_cache(val_loader, context, "val", window_radius=window_radius)
            print(
                f"[{architecture}] Validation loader ready: {len(val_loader)} batches "
                f"in {time.perf_counter() - loader_start:.1f}s"
            )
        train_dataset = train_loader.dataset
        dimensions = {
            "dna": int(train_dataset.v_dna_tensor.shape[1]),
            "protein": int(train_dataset.v_prot_tensor.shape[1]),
            "bio": int(train_dataset.bio_tensor.shape[1]),
            "geometry": int(train_dataset.geom_tensor.shape[1]) if train_dataset.has_geom else 0,
            "active_modalities": list(target["active_modalities"]),
        }
        if context.get("token_embedding_cache_index") is not None:
            dimensions.update({
                "token_dna": int(train_loader.dataset.token_hidden_sizes["dna"]),
                "token_protein": int(train_loader.dataset.token_hidden_sizes["protein"]),
                "token_cache": True,
            })
        biological_dim = dimensions["bio"] + dimensions["geometry"]
        model = model_builder(dimensions, config).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        if validation_row is None or not best_checkpoint.exists():
            if getattr(train_loader, "generator", None) is not None:
                train_loader.generator.manual_seed(int(config["seed"]))
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(config["lr_factor"]),
                patience=int(config["lr_patience"]),
            )
            if training_state_path.exists():
                training_state = torch.load(
                    training_state_path, map_location="cpu", weights_only=False
                )
                if training_state.get("state_version") != 1:
                    raise ValueError(
                        f"Unsupported training checkpoint version: {training_state_path}"
                    )
                model.load_state_dict(training_state["model_state"])
                optimizer.load_state_dict(training_state["optimizer_state"])
                scheduler.load_state_dict(training_state["scheduler_state"])
                history_rows = [
                    row for row in history_rows
                    if str(row.get("Target_Config")) != target_name
                ] + training_state["history_rows"]
                start_epoch = int(training_state["completed_epoch"])
                best_mcc = float(training_state["best_mcc"])
                best_epoch = int(training_state["best_epoch"])
                stale_epochs = int(training_state["stale_epochs"])
                best_model_state = training_state["best_model_state"]
                fit_seconds_before_resume = float(training_state["fit_seconds"])
                training_complete = bool(training_state["training_complete"])
                _restore_rng_state(training_state["rng_state"], train_loader)
                save_progress_csv(history_rows, history_path)
                print(
                    f"[RESUME] Continuing {architecture} {target_name} from epoch "
                    f"{start_epoch + 1}/{config['max_epochs']}"
                )
            else:
                start_epoch = 0
                best_mcc = -float("inf")
                best_epoch = 0
                stale_epochs = 0
                best_model_state = None
                fit_seconds_before_resume = 0.0
                training_complete = False

            segment_start = time.perf_counter()
            if not training_complete:
                epoch_progress = tqdm(
                    range(start_epoch, int(config["max_epochs"])),
                    total=int(config["max_epochs"]),
                    initial=start_epoch,
                    desc=f"{architecture} | {variant} | {target_name}",
                    unit="epoch",
                    leave=True,
                )
                for epoch in epoch_progress:
                    epoch_start = time.perf_counter()
                    model.train()
                    train_loss_sum, train_count = 0.0, 0
                    train_batches = tqdm(
                        train_loader,
                        desc=f"Train epoch {epoch + 1}/{config['max_epochs']}",
                        unit="batch",
                        leave=False,
                    )
                    for batch in train_batches:
                        optimizer.zero_grad(set_to_none=True)
                        logits = _forward(model, batch, device)
                        labels = _move(batch, "label", device, required=True).reshape_as(logits)
                        loss = criterion(logits, labels)
                        loss.backward()
                        optimizer.step()
                        train_batches.set_postfix(loss=f"{loss.item():.4f}")
                        count = int(labels.shape[0])
                        train_loss_sum += float(loss.item()) * count
                        train_count += count

                    train_loss = train_loss_sum / max(train_count, 1)
                    validation_start = time.perf_counter()
                    val_loss, y_val, p_val = _collect(
                        model,
                        val_loader,
                        device,
                        criterion,
                        progress_desc=f"Validation epoch {epoch + 1}/{config['max_epochs']}",
                    )
                    threshold, val_mcc, val_metrics = _metrics(pipeline, y_val, p_val)
                    scheduler.step(val_mcc)
                    epoch_row = {
                        "Target_Config": target_name,
                        "Architecture": architecture,
                        "Variant": variant,
                        "Epoch": epoch + 1,
                        "Train_Loss": train_loss,
                        "Val_Loss": val_loss,
                        "Val_MCC": val_mcc,
                        "Val_Threshold": threshold,
                        "Val_AUROC": val_metrics["AUROC"],
                        "Val_AUPRC": val_metrics["AUPRC"],
                        "Learning_Rate": float(optimizer.param_groups[0]["lr"]),
                    }
                    history_rows.append(epoch_row)

                    if val_mcc > best_mcc:
                        best_mcc = val_mcc
                        best_epoch = epoch + 1
                        stale_epochs = 0
                        best_model_state = _cpu_state_dict(model)
                    else:
                        stale_epochs += 1
                    training_complete = (
                        stale_epochs >= int(config["early_stop_patience"])
                        or epoch + 1 >= int(config["max_epochs"])
                    )
                    target_history = [
                        row for row in history_rows
                        if str(row.get("Target_Config")) == target_name
                    ]
                    training_state = {
                        "state_version": 1,
                        "completed_epoch": epoch + 1,
                        "training_complete": training_complete,
                        "model_state": _cpu_state_dict(model),
                        "best_model_state": best_model_state,
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_mcc": best_mcc,
                        "best_epoch": best_epoch,
                        "stale_epochs": stale_epochs,
                        "fit_seconds": fit_seconds_before_resume + time.perf_counter() - segment_start,
                        "history_rows": target_history,
                        "rng_state": _capture_rng_state(train_loader),
                    }
                    # Save the full resumable state before the CSV so a restart can
                    # reconstruct the history if interruption occurs between writes.
                    _atomic_torch_save(training_state, training_state_path)
                    if best_model_state is not None:
                        _atomic_torch_save(best_model_state, best_checkpoint)
                    save_progress_csv(history_rows, history_path)
                    if window_radius is not None:
                        print(
                            f"[Epoch {epoch + 1}] train={validation_start - epoch_start:.1f}s; "
                            f"total including validation and checkpoint="
                            f"{time.perf_counter() - epoch_start:.1f}s"
                        )
                    epoch_progress.set_postfix(
                        train_loss=f"{train_loss:.4f}",
                        val_loss=f"{val_loss:.4f}",
                        val_MCC=f"{val_mcc:.4f}",
                        threshold=f"{threshold:.4f}",
                        best_epoch=best_epoch,
                    )
                    if training_complete:
                        break

            if best_epoch == 0:
                raise RuntimeError(
                    f"No validation-selected {architecture} checkpoint produced for {target_name}"
                )
            _atomic_torch_save(best_model_state, best_checkpoint)
            model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=True))
            _, y_val, p_val = _collect(
                model, val_loader, device, progress_desc="Final validation"
            )
            threshold, val_mcc, val_metrics = _metrics(pipeline, y_val, p_val)
            validation_row = {
                "Run_ID": Path(pipeline.batch_run_dir).name,
                "Branch": "DL",
                "Evaluation_Split": "validation",
                "Target_Config": target_name,
                "Ablation": target["ablation"],
                "Pooling": "center",
                "DNA_Model": dna_model["name"],
                "Prot_Model": protein_model["name"],
                "Network": architecture,
                "Architecture": architecture,
                "Variant": variant,
                "Input_Representation": embedding_representation,
                "Classifier": "Residual_MLP",
                "Feature_Set": "DNA+Protein+Bio+Geometry" if dimensions["geometry"] else "DNA+Protein+Bio",
                "Source": source,
                "Candidate_Level": "single_run",
                "Seed": int(config["seed"]),
                "N_Seeds": 1,
                "Best_Epoch": best_epoch,
                "Fit_Seconds": fit_seconds_before_resume + time.perf_counter() - segment_start,
                "Decision_Threshold": threshold,
                "Val_MCC": val_mcc,
                "Parameters": parameter_count,
                "Hyperparameters": json.dumps(config, sort_keys=True),
                "Source_File": str(validation_path),
                **{f"Val_{key}": value for key, value in val_metrics.items()},
            }
            validation_rows = upsert_record(validation_rows, validation_row, ["Target_Config"])
            save_progress_csv(validation_rows, validation_path)
        else:
            model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=True))
            print(f"[RESUME] Reusing {architecture} checkpoint for {target_name}")

        if evaluate_test:
            threshold = float(validation_row["Decision_Threshold"])
            profiler = FusionEvaluatorProfiler(
                tensorboard_dir=None,
                fm_profile_json=getattr(pipeline, "fm_profile_json", None),
                geom_profile_json=getattr(pipeline, "geom_profile_json", None),
                device=device,
            )
            dummy_inputs = [
                torch.zeros((1, dimensions["dna"]), device=device),
                torch.zeros((1, dimensions["protein"]), device=device),
                torch.zeros((1, dimensions["bio"]), device=device),
            ]
            if dimensions["geometry"]:
                dummy_inputs.append(torch.zeros((1, dimensions["geometry"]), device=device))
            if dimensions.get("token_cache"):
                if not dimensions["geometry"]:
                    # Keep the optional geom argument in its positional slot before
                    # the six token tensors consumed by token-aware models.
                    dummy_inputs.append(torch.zeros((1, 0), device=device))
                profile_window = 2 * int(config.get("k_radius", 0)) + 1
                profile_center = int(config.get("k_radius", 0))
                dummy_inputs.extend([
                    torch.zeros((1, 2, profile_window, dimensions["token_dna"]), device=device),
                    torch.ones((1, 2, profile_window), dtype=torch.bool, device=device),
                    torch.full((1, 2), profile_center, dtype=torch.long, device=device),
                    torch.zeros((1, 2, profile_window, dimensions["token_protein"]), device=device),
                    torch.ones((1, 2, profile_window), dtype=torch.bool, device=device),
                    torch.full((1, 2), profile_center, dtype=torch.long, device=device),
                ])
            profiler.profile_pytorch_fusion(model, tuple(dummy_inputs))
    
            for dataset_cfg in datasets:
                dataset_name = str(dataset_cfg["name"])
                saved_test_row = next((
                    row for row in test_rows
                    if str(row.get("Target_Config")) == target_name
                    and str(row.get("Dataset")) == dataset_name
                ), None)
                if saved_test_row is not None and all(
                    column in saved_test_row and pd.notna(saved_test_row[column])
                    for column in PROFILE_METRIC_COLUMNS
                ):
                    print(f"[{architecture}] Reusing saved test metrics: {target_name} / {dataset_name}")
                    continue
                if saved_test_row is not None:
                    # Older Module 6 rows may have valid predictions/metrics but no
                    # hardware columns. Re-evaluate from the saved checkpoint and
                    # replace those rows without duplicating per-variant predictions.
                    test_rows = [
                        row for row in test_rows
                        if not (
                            str(row.get("Target_Config")) == target_name
                            and str(row.get("Dataset")) == dataset_name
                        )
                    ]
                    prediction_rows = [
                        row for row in prediction_rows
                        if not (
                            str(row.get("Target_Config")) == target_name
                            and str(row.get("Dataset")) == dataset_name
                        )
                    ]
                if any(
                    str(row.get("Target_Config")) == target_name
                    and str(row.get("Dataset")) == dataset_name
                    for row in test_rows
                ):
                    continue
                print(f"[{architecture}] Evaluating test set: {target_name} / {dataset_name}")
                test_loader = _architecture_loader(
                    pipeline, context, target, dataset_cfg["test"], "center", dna_model,
                    protein_model, False, dataset_cfg["train"], int(config["seed"]),
                    embedding_representation,
                )
                test_loader = _attach_token_cache(
                    test_loader, context, dataset_cfg["test"], window_radius=window_radius
                )
                _, y_test, p_test = _collect(
                    model,
                    test_loader,
                    device,
                    profiler=profiler,
                    progress_desc=f"Test {dataset_name}",
                )
                y_pred = pipeline._threshold_predictions(p_test, threshold)
                metrics = FusionEvaluatorProfiler.compute_metrics(y_test, p_test, y_pred)
                profile_metrics = _get_test_profile_metrics(
                    profiler=profiler,
                    dna_model_name=dna_model["name"],
                    protein_model_name=protein_model["name"],
                    active_modalities=target["active_modalities"],
                    split_name=dataset_cfg["test"],
                )
                test_row = {
                    "Run_ID": Path(pipeline.batch_run_dir).name,
                    "Branch": "DL",
                    "Evaluation_Split": "test",
                    "Target_Config": target_name,
                    "Ablation": target["ablation"],
                    "Dataset": dataset_name,
                    "Pooling": "center",
                    "DNA_Model": dna_model["name"],
                    "Prot_Model": protein_model["name"],
                    "Network": architecture,
                    "Architecture": architecture,
                    "Variant": variant,
                    "Input_Representation": embedding_representation,
                    "Classifier": "Residual_MLP",
                    "Feature_Set": "DNA+Protein+Bio+Geometry" if dimensions["geometry"] else "DNA+Protein+Bio",
                    "Source": source,
                    "Candidate_Level": "single_run",
                    "Seed": int(config["seed"]),
                    "N_Seeds": 1,
                    "Decision_Threshold": threshold,
                    "Best_Epoch": validation_row["Best_Epoch"],
                    "Fit_Seconds": validation_row["Fit_Seconds"],
                    "Val_MCC": validation_row["Val_MCC"],
                    "Parameters": parameter_count,
                    "Hyperparameters": json.dumps(config, sort_keys=True),
                    "Source_File": str(test_path),
                    **{
                        f"Val_{key}": validation_row.get(f"Val_{key}", np.nan)
                        for key in (
                            "Accuracy", "Precision", "Recall", "Specificity",
                            "F1_Score", "MCC", "AUROC", "AUPRC",
                        )
                    },
                    **metrics,
                    **profile_metrics,
                }
                prediction_rows.extend({
                    "Target_Config": target_name,
                    "Dataset": dataset_name,
                    "Variant_ID": str(variant_id),
                    "Label": int(label),
                    "Probability_Pathogenic": float(probability),
                    "Prediction": int(prediction),
                } for variant_id, label, probability, prediction in zip(
                    test_loader.dataset.df["Variant_ID"].astype(str), y_test, p_test, y_pred
                ))
                save_progress_csv(prediction_rows, predictions_path)
                test_rows = upsert_record(test_rows, test_row, ["Target_Config", "Dataset"])
                save_progress_csv(test_rows, test_path)
                print(
                    f"[{architecture}] Test complete: {target_name} / {dataset_name} "
                    f"MCC={metrics['MCC']:.4f} AUROC={metrics['AUROC']:.4f} "
                    f"AUPRC={metrics['AUPRC']:.4f}"
                )
            profiler.close()

    results = {
        "output_dir": str(stage_dir),
        "validation_metrics": pd.DataFrame(validation_rows),
        "test_metrics": pd.DataFrame(test_rows),
        "epoch_history": pd.DataFrame(history_rows),
        "test_predictions": pd.DataFrame(prediction_rows),
    }
    display(f"{architecture} training, validation, and test evaluation complete. Results and checkpoints: {stage_dir}")
    if not results["test_metrics"].empty:
        test_columns = [
            column for column in (
                "Target_Config", "Dataset", "Architecture", "Variant",
                "Decision_Threshold", "Accuracy", "Precision", "Recall",
                "Specificity", "F1_Score", "MCC", "AUROC", "AUPRC",
            )
            if column in results["test_metrics"].columns
        ]
        display(results["test_metrics"][test_columns].sort_values(["Target_Config", "Dataset"]))
    return results


def run_architecture_optuna(
    context,
    base_config,
    search_space,
    n_trials,
    model_builder,
    architecture,
    variant_builder,
    experiment_key,
    source,
    display_fn=None,
    manifest_architecture=None,
    embedding_representation="delta",
):
    """Optimize validation MCC with resumable TPE trials; test rows are reports only."""
    pipeline = context["pipeline"]
    display = display_fn or print
    unknown = [key for key in search_space if key not in base_config]
    if unknown:
        raise KeyError(f"Optuna parameters are absent from the base config: {unknown}")

    objective_name = str(context.get(
        "optuna_objective_name", "mean_validation_mcc_across_A_and_B"
    ))
    root_dir = Path(pipeline.batch_run_dir) / "module06_architecture_experiments" / experiment_key
    root_dir.mkdir(parents=True, exist_ok=True)
    study_dir = root_dir / "optuna"
    # A changed search space must not silently reuse incompatible trial artifacts.
    ensure_stage_signature(study_dir, f"module06_{experiment_key}_optuna", {
        "architecture": manifest_architecture or architecture,
        "base_config": base_config,
        "search_space": search_space,
        "target_configs": context["TARGET_CONFIGS"],
        "datasets": context["datasets"],
        "embedding_representation": embedding_representation,
        "objective": objective_name,
        "input_data_signature": context.get("input_data_signature"),
    })

    aggregate_paths = {
        "validation_metrics": root_dir / f"{experiment_key}_optuna_validation_metrics.csv",
        "test_metrics": root_dir / f"{experiment_key}_optuna_test_metrics.csv",
        "epoch_history": root_dir / f"{experiment_key}_optuna_epoch_history.csv",
    }
    aggregate_keys = {
        "validation_metrics": ["Trial_ID", "Target_Config"],
        "test_metrics": ["Trial_ID", "Target_Config", "Dataset"],
        "epoch_history": ["Trial_ID", "Target_Config", "Epoch"],
    }
    aggregates = {
        name: pd.read_csv(path).to_dict("records") if path.exists() else []
        for name, path in aggregate_paths.items()
    }
    trial_log_path = root_dir / f"{experiment_key}_optuna_trials.csv"
    trial_log = pd.read_csv(trial_log_path).to_dict("records") if trial_log_path.exists() else []

    sampler_seed = int(base_config.get("seed", 42))
    plan = optuna_trial_plan(
        study_dir,
        f"module06_{experiment_key}",
        int(n_trials),
        search_space,
        seed=sampler_seed,
    )
    for study, opt_trial, params, is_new in plan:
        trial_id = f"optuna_{opt_trial.number:04d}"
        trial_config = dict(base_config)
        trial_config.update(params)
        stable_params = json.dumps(params, sort_keys=True, separators=(",", ":"))
        artifact_id = "params_" + hashlib.sha1(stable_params.encode("utf-8")).hexdigest()[:12]
        result = run_architecture_experiment(
            context=context,
            experiment_config=trial_config,
            model_builder=model_builder,
            architecture=architecture,
            variant=variant_builder(trial_config),
            experiment_key=experiment_key,
            source=source,
            display_fn=lambda *_args, **_kwargs: None,
            manifest_architecture=manifest_architecture,
            # A stable parameter key lets a retried RUNNING Optuna trial resume
            # its epoch checkpoint even when Optuna assigns it a new trial number.
            trial_name=artifact_id,
            embedding_representation=embedding_representation,
        )

        validation = result["validation_metrics"]
        if validation.empty or "Val_MCC" not in validation:
            raise RuntimeError(f"Optuna trial {trial_id} produced no validation MCC")
        validation = validation.copy()
        validation["Val_MCC"] = pd.to_numeric(validation["Val_MCC"], errors="coerce")
        objective = float(validation["Val_MCC"].mean())
        by_config = {
            str(row["Target_Config"]): float(row["Val_MCC"])
            for _, row in validation.iterrows()
            if pd.notna(row["Val_MCC"])
        }

        for name, output_path in aggregate_paths.items():
            frame = result[name].copy()
            if frame.empty:
                continue
            frame["Trial_ID"] = trial_id
            frame["Optuna_Trial"] = int(opt_trial.number)
            frame["Optuna_Hyperparameters"] = json.dumps(params, sort_keys=True)
            frame["Optuna_Objective_Mean_Val_MCC"] = objective
            merged = pd.concat([pd.DataFrame(aggregates[name]), frame], ignore_index=True, sort=False)
            merged = merged.drop_duplicates(subset=aggregate_keys[name], keep="last")
            aggregates[name] = merged.to_dict("records")
            save_progress_csv(aggregates[name], output_path)

        if is_new:
            opt_trial.set_user_attr("validation_mcc_by_config", by_config)
            tell_trial(study, opt_trial, objective)

        trial_row = {
            "Optuna_Trial": int(opt_trial.number),
            "Trial_ID": trial_id,
            "State": "COMPLETE",
            "Objective_Name": objective_name,
            "Objective_Value": objective,
            "Val_MCC_A": by_config.get("A_no_geometry", np.nan),
            "Val_MCC_B": by_config.get("B_with_geometry", np.nan),
            "Hyperparameters": json.dumps(params, sort_keys=True),
        }
        trial_log = [row for row in trial_log if row.get("Trial_ID") != trial_id]
        trial_log.append(trial_row)
        save_progress_csv(trial_log, trial_log_path)
        display(
            f"[{architecture} Optuna] {trial_id}; mean validation MCC={objective:.4f}; "
            f"A={by_config.get('A_no_geometry', float('nan')):.4f}; "
            f"B={by_config.get('B_with_geometry', float('nan')):.4f}; params={params}"
        )

    results = {
        "output_dir": str(root_dir),
        "trials": pd.DataFrame(trial_log).sort_values("Optuna_Trial") if trial_log else pd.DataFrame(),
        **{name: pd.DataFrame(rows) for name, rows in aggregates.items()},
    }
    if not results["validation_metrics"].empty:
        display(results["validation_metrics"].sort_values(
            ["Target_Config", "Val_MCC"], ascending=[True, False]
        ))
    if not results["test_metrics"].empty:
        display(results["test_metrics"].sort_values(
            ["Target_Config", "Dataset", "MCC"], ascending=[True, True, False]
        ))
    display(f"{architecture} Optuna study complete. SQLite and CSV results: {root_dir}")
    return results
