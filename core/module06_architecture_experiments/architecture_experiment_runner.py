"""Shared train, validation, checkpoint, and resume engine for Module 6 models."""
from __future__ import annotations

import json
from itertools import product
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from core.module05v2_fusion_classifier.evaluator_profiler import FusionEvaluatorProfiler
from core.module05v2_fusion_classifier.resume_utils import (
    ensure_stage_signature,
    save_progress_csv,
    upsert_record,
)



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
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _move(batch, key, device, required=False):
    value = batch.get(key)
    if value is None and required:
        raise KeyError(f"Required architecture batch input is missing: {key}")
    return value.to(device, non_blocking=True) if value is not None else None


def _forward(model, batch, device):
    return model(
        _move(batch, "v_dna", device, required=True),
        _move(batch, "v_prot", device, required=True),
        _move(batch, "bio_features", device, required=True),
        _move(batch, "geom_features", device),
    )


def _collect(model, loader, device, criterion=None):
    model.eval()
    labels, probabilities = [], []
    loss_total, n_total = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            logits = _forward(model, batch, device)
            y = _move(batch, "label", device, required=True)
            if criterion is not None:
                loss_total += float(criterion(logits, y).item()) * int(y.shape[0])
                n_total += int(y.shape[0])
            labels.extend(y.reshape(-1).cpu().numpy())
            probabilities.extend(torch.sigmoid(logits).reshape(-1).cpu().numpy())
    mean_loss = loss_total / max(n_total, 1) if criterion is not None else np.nan
    return mean_loss, np.asarray(labels, dtype=np.int32), np.asarray(probabilities, dtype=np.float32)


def _metrics(pipeline, labels, probabilities):
    threshold, mcc = pipeline._select_validation_threshold(labels, probabilities)
    predictions = pipeline._threshold_predictions(probabilities, threshold)
    values = FusionEvaluatorProfiler.compute_metrics(labels, probabilities, predictions)
    return float(threshold), float(mcc), values


def run_architecture_experiment(
    context, experiment_config, model_builder, architecture, variant, experiment_key,
    source, display_fn=None, manifest_architecture=None, trial_name=None,
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

    stage_dir = Path(pipeline.batch_run_dir) / "module06_architecture_experiments" / experiment_key
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
        "checkpoint_selection": "validation_mcc_optimized_threshold_v2",
    }
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

        # Seed before constructing the model so its initial weights match the
        # recorded seed. On resume, the saved RNG state is restored afterward.
        _seed_everything(int(config["seed"]))
        train_loader = pipeline._get_dataloader(
            "train3", "center", dna_model["name"], protein_model["name"],
            target["active_modalities"], True, target.get("geom_feature_names"),
            "train3", int(config["seed"]),
        )
        val_loader = pipeline._get_dataloader(
            "val", "center", dna_model["name"], protein_model["name"],
            target["active_modalities"], False, target.get("geom_feature_names"),
            "train3", int(config["seed"]),
        )
        train_dataset = train_loader.dataset
        dimensions = {
            "dna": int(train_dataset.v_dna_tensor.shape[1]),
            "protein": int(train_dataset.v_prot_tensor.shape[1]),
            "bio": int(train_dataset.bio_tensor.shape[1]),
            "geometry": int(train_dataset.geom_tensor.shape[1]) if train_dataset.has_geom else 0,
        }
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
                for epoch in range(start_epoch, int(config["max_epochs"])):
                    model.train()
                    train_loss_sum, train_count = 0.0, 0
                    for batch in train_loader:
                        optimizer.zero_grad(set_to_none=True)
                        logits = _forward(model, batch, device)
                        labels = _move(batch, "label", device, required=True)
                        loss = criterion(logits, labels)
                        loss.backward()
                        optimizer.step()
                        count = int(labels.shape[0])
                        train_loss_sum += float(loss.item()) * count
                        train_count += count

                    train_loss = train_loss_sum / max(train_count, 1)
                    val_loss, y_val, p_val = _collect(model, val_loader, device, criterion)
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
                    print(
                        f"[{architecture}] {target_name} epoch {epoch + 1}/{config['max_epochs']} "
                        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                        f"val_MCC={val_mcc:.4f} threshold={threshold:.4f}"
                    )
                    if training_complete:
                        break

            if best_epoch == 0:
                raise RuntimeError(
                    f"No validation-selected {architecture} checkpoint produced for {target_name}"
                )
            _atomic_torch_save(best_model_state, best_checkpoint)
            model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=True))
            _, y_val, p_val = _collect(model, val_loader, device)
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

        threshold = float(validation_row["Decision_Threshold"])
        for dataset_cfg in datasets:
            dataset_name = str(dataset_cfg["name"])
            if any(
                str(row.get("Target_Config")) == target_name
                and str(row.get("Dataset")) == dataset_name
                for row in test_rows
            ):
                print(f"[{architecture}] Reusing saved test metrics: {target_name} / {dataset_name}")
                continue
            print(f"[{architecture}] Evaluating test set: {target_name} / {dataset_name}")
            test_loader = pipeline._get_dataloader(
                dataset_cfg["test"], "center", dna_model["name"], protein_model["name"],
                target["active_modalities"], False, target.get("geom_feature_names"),
                dataset_cfg["train"], int(config["seed"]),
            )
            _, y_test, p_test = _collect(model, test_loader, device)
            y_pred = pipeline._threshold_predictions(p_test, threshold)
            metrics = FusionEvaluatorProfiler.compute_metrics(y_test, p_test, y_pred)
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


def run_architecture_grid(
    context,
    base_config,
    search_space,
    model_builder,
    architecture,
    variant_builder,
    experiment_key,
    source,
    display_fn=None,
    manifest_architecture=None,
):
    """Run a Cartesian hyperparameter grid, resuming each trial independently."""
    pipeline = context["pipeline"]
    display = display_fn or print
    search_keys = list(search_space)
    search_values = [list(search_space[key]) for key in search_keys]
    if any(not values for values in search_values):
        raise ValueError("Every grid-search parameter must have at least one candidate value")
    unknown_keys = [key for key in search_keys if key not in base_config]
    if unknown_keys:
        raise KeyError(f"Grid parameters are absent from the base config: {unknown_keys}")

    root_dir = Path(pipeline.batch_run_dir) / "module06_architecture_experiments" / experiment_key
    root_dir.mkdir(parents=True, exist_ok=True)
    aggregate_paths = {
        "validation_metrics": root_dir / f"{experiment_key}_grid_validation_metrics.csv",
        "test_metrics": root_dir / f"{experiment_key}_grid_test_metrics.csv",
        "epoch_history": root_dir / f"{experiment_key}_grid_epoch_history.csv",
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

    combinations = list(product(*search_values)) if search_keys else [()]
    for index, values in enumerate(combinations, start=1):
        trial_id = f"trial_{index:03d}"
        trial_config = dict(base_config)
        trial_config.update(dict(zip(search_keys, values)))
        trial_params = {key: trial_config[key] for key in search_keys}
        variant = variant_builder(trial_config)
        result = run_architecture_experiment(
            context=context,
            experiment_config=trial_config,
            model_builder=model_builder,
            architecture=architecture,
            variant=variant,
            experiment_key=experiment_key,
            source=source,
            # Show one compact grid summary below, not a separate table per trial.
            display_fn=lambda *_args, **_kwargs: None,
            manifest_architecture=manifest_architecture,
            trial_name=trial_id,
        )

        for name in aggregate_paths:
            frame = result[name].copy()
            if frame.empty:
                continue
            frame["Trial_ID"] = trial_id
            frame["Grid_Hyperparameters"] = json.dumps(trial_params, sort_keys=True)
            merged = pd.concat([pd.DataFrame(aggregates[name]), frame], ignore_index=True, sort=False)
            keys = aggregate_keys[name]
            merged = merged.drop_duplicates(subset=keys, keep="last")
            aggregates[name] = merged.to_dict("records")
            save_progress_csv(aggregates[name], aggregate_paths[name])

        display(f"[{architecture} grid] {trial_id}/{len(combinations)} complete: {trial_params}")

    results = {
        "output_dir": str(root_dir),
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
    display(f"{architecture} grid complete ({len(combinations)} trials). Aggregate CSVs: {root_dir}")
    return results

