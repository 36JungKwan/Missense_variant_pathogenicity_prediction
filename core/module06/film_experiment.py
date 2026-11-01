"""Train, evaluate, checkpoint, and resume the Module 6 FiLM experiment."""
from __future__ import annotations

import json
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

from .film_model import BiologyConditionedFiLM


DEFAULT_FILM_CONFIG = {
    "hidden_dim": 512,
    "bio_hidden_dim": 512,
    "head_dim": 128,
    "dropout": 0.2,
    "max_epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "early_stop_patience": 6,
    "lr_factor": 0.5,
    "lr_patience": 3,
    "seed": 42,
}


def get_default_film_config():
    """Return an independent, notebook-editable copy of FiLM settings."""
    return dict(DEFAULT_FILM_CONFIG)


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move(batch, key, device, required=False):
    value = batch.get(key)
    if value is None and required:
        raise KeyError(f"Required FiLM batch input is missing: {key}")
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


def run_film_experiment(context, film_config=None, display_fn=None):
    """Run the single FiLM idea on configs A/B and the configured test datasets.

    Completed train/validation checkpoints and test rows are reused on resume.
    The stage manifest rejects resuming with changed model or training settings.
    """
    pipeline = context["pipeline"]
    targets = context["TARGET_CONFIGS"]
    datasets = context["datasets"]
    dna_model = context["DNA_MODEL"]
    protein_model = context["PROTEIN_MODEL"]
    device = torch.device(context.get("DEVICE", pipeline.device))
    config = dict(DEFAULT_FILM_CONFIG if film_config is None else film_config)
    display = display_fn or print

    stage_dir = Path(pipeline.batch_run_dir) / "module06_architecture_experiments" / "film"
    checkpoint_root = stage_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    validation_path = stage_dir / "film_validation_metrics.csv"
    test_path = stage_dir / "film_test_metrics.csv"
    history_path = stage_dir / "film_epoch_history.csv"
    predictions_path = stage_dir / "film_test_predictions.csv"

    signature = {
        "architecture": "biology_conditioned_film",
        "film_config": config,
        "pipeline_config": pipeline.config,
        "target_configs": targets,
        "datasets": datasets,
        "dna_model": dna_model,
        "protein_model": protein_model,
        "pooling": "center",
        "checkpoint_selection": "validation_mcc_optimized_threshold_v2",
    }
    ensure_stage_signature(stage_dir, "module06_film", signature)

    validation_rows = pd.read_csv(validation_path).to_dict("records") if validation_path.exists() else []
    test_rows = pd.read_csv(test_path).to_dict("records") if test_path.exists() else []
    history_rows = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    prediction_rows = pd.read_csv(predictions_path).to_dict("records") if predictions_path.exists() else []

    for target in targets:
        target_name = str(target["name"])
        model_dir = checkpoint_root / target_name
        model_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint = model_dir / "best_model.pth"
        validation_row = next(
            (row for row in validation_rows if str(row.get("Target_Config")) == target_name), None
        )

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
        model = BiologyConditionedFiLM(
            dna_dim=dimensions["dna"],
            protein_dim=dimensions["protein"],
            biological_dim=biological_dim,
            hidden_dim=int(config["hidden_dim"]),
            bio_hidden_dim=int(config["bio_hidden_dim"]),
            head_dim=int(config["head_dim"]),
            dropout=float(config["dropout"]),
        ).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        if validation_row is None or not best_checkpoint.exists():
            _seed_everything(int(config["seed"]))
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
            best_mcc = -float("inf")
            best_epoch = 0
            stale_epochs = 0
            start_time = time.perf_counter()

            for epoch in range(int(config["max_epochs"])):
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
                history_rows.append({
                    "Target_Config": target_name,
                    "Architecture": "FiLM",
                    "Variant": "bio_conditioned_residual_film",
                    "Epoch": epoch + 1,
                    "Train_Loss": train_loss,
                    "Val_Loss": val_loss,
                    "Val_MCC": val_mcc,
                    "Val_Threshold": threshold,
                    "Val_AUROC": val_metrics["AUROC"],
                    "Val_AUPRC": val_metrics["AUPRC"],
                    "Learning_Rate": float(optimizer.param_groups[0]["lr"]),
                })
                save_progress_csv(history_rows, history_path)

                if val_mcc > best_mcc:
                    best_mcc = val_mcc
                    best_epoch = epoch + 1
                    stale_epochs = 0
                    torch.save(model.state_dict(), best_checkpoint)
                else:
                    stale_epochs += 1
                print(
                    f"[FiLM] {target_name} epoch {epoch + 1}/{config['max_epochs']} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"val_MCC={val_mcc:.4f} threshold={threshold:.4f}"
                )
                if stale_epochs >= int(config["early_stop_patience"]):
                    break

            if best_epoch == 0:
                raise RuntimeError(f"No validation-selected FiLM checkpoint produced for {target_name}")
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
                "Network": "FiLM",
                "Architecture": "FiLM",
                "Variant": "bio_conditioned_residual_film",
                "Classifier": "Residual_MLP",
                "Feature_Set": "DNA+Protein+Bio+Geometry" if dimensions["geometry"] else "DNA+Protein+Bio",
                "Source": "module06_film",
                "Candidate_Level": "single_run",
                "Seed": int(config["seed"]),
                "N_Seeds": 1,
                "Best_Epoch": best_epoch,
                "Fit_Seconds": time.perf_counter() - start_time,
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
            print(f"[RESUME] Reusing FiLM checkpoint for {target_name}")

        threshold = float(validation_row["Decision_Threshold"])
        for dataset_cfg in datasets:
            dataset_name = str(dataset_cfg["name"])
            if any(
                str(row.get("Target_Config")) == target_name
                and str(row.get("Dataset")) == dataset_name
                for row in test_rows
            ):
                continue
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
                "Network": "FiLM",
                "Architecture": "FiLM",
                "Variant": "bio_conditioned_residual_film",
                "Classifier": "Residual_MLP",
                "Feature_Set": "DNA+Protein+Bio+Geometry" if dimensions["geometry"] else "DNA+Protein+Bio",
                "Source": "module06_film",
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

    results = {
        "output_dir": str(stage_dir),
        "validation_metrics": pd.DataFrame(validation_rows),
        "test_metrics": pd.DataFrame(test_rows),
        "epoch_history": pd.DataFrame(history_rows),
        "test_predictions": pd.DataFrame(prediction_rows),
    }
    display(f"FiLM stage complete. Results and checkpoints: {stage_dir}")
    return results

