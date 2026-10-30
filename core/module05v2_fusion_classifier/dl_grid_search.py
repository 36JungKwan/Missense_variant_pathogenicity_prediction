from .resume_utils import ensure_stage_signature, reject_legacy_stage_artifacts, save_progress_csv, upsert_record
from .optuna_search import optuna_trial_plan, tell_trial, prune_trial
import pandas as pd

"""Grid searches for pure-DL fusion baselines and regularization hyperparameters."""

def run_pure_dl_optuna(context, search_config, stage, display_fn=None):
    pipeline = context["pipeline"]
    BASELINE_BATCH_RUN_DIR = context.get("BASELINE_BATCH_RUN_DIR")
    CONFIG = context["CONFIG"]
    DEVICE = context["DEVICE"]
    DNA_MODEL = context["DNA_MODEL"]
    PROTEIN_MODEL = context["PROTEIN_MODEL"]
    TARGET_CONFIGS = context["TARGET_CONFIGS"]
    datasets = context["datasets"]
    DL_SEARCH_CONFIG = search_config
    display = display_fn or print
    if stage not in {"v1", "v2"}:
        raise ValueError("stage must be 'v1' or 'v2'")
    # Shared Optuna implementation for the two regularization-search stages.
    # ==============================================================================
    import itertools
    import json
    import random
    import time
    from pathlib import Path
    
    import numpy as np
    import torch
    import torch.nn as nn
    
    from core.module05v2_fusion_classifier.evaluator_profiler import FusionEvaluatorProfiler
    from core.module05v2_fusion_classifier.fusion_model import MultiStrategyFusionModel
    
    PURE_DL_DIR = Path(pipeline.batch_run_dir) / ("pure_dl_regularization_grid" if stage == "v1" else "pure_dl_regularization_grid_v2")
    PURE_DL_DIR.mkdir(parents=True, exist_ok=True)
    PURE_DL_SPACE = DL_SEARCH_CONFIG[f'regularization_space_{stage}']
    PURE_DL_N_TRIALS = int(DL_SEARCH_CONFIG.get("n_trials", {}).get(f"regularization_{stage}", 12 if stage == "v1" else 24))
    PURE_DL_ARCHITECTURES = DL_SEARCH_CONFIG[f'pure_dl_architectures_{stage}']
    PURE_DL_OUTPUTS = [
        PURE_DL_DIR / "pure_dl_grid_trials.csv",
        PURE_DL_DIR / "pure_dl_validation_metrics.csv",
        PURE_DL_DIR / "pure_dl_test_metrics.csv",
        PURE_DL_DIR / "pure_dl_epoch_history.csv",
        PURE_DL_DIR / "pure_dl_vs_baseline.csv",
    ]
    
    reject_legacy_stage_artifacts(PURE_DL_DIR, f"pure_dl_optuna_{stage}")
    ensure_stage_signature(PURE_DL_DIR, f"pure_dl_optuna_{stage}", {
        "sampler": "TPE", "search_space": PURE_DL_SPACE, "n_trials": PURE_DL_N_TRIALS,
        "architectures": PURE_DL_ARCHITECTURES, "config": CONFIG,
        "targets": TARGET_CONFIGS, "datasets": datasets, "dna_model": DNA_MODEL, "protein_model": PROTEIN_MODEL,
        "checkpoint_selection": "validation_mcc_optimized_threshold_v2",
    })
    if all(path.exists() for path in PURE_DL_OUTPUTS):
        print(f"[REUSE] Completed pure-DL Optuna results: {PURE_DL_DIR}")
        pure_dl_trials_df = pd.read_csv(PURE_DL_OUTPUTS[0])
        pure_dl_validation_df = pd.read_csv(PURE_DL_OUTPUTS[1])
        pure_dl_test_df = pd.read_csv(PURE_DL_OUTPUTS[2])
        pure_dl_history_df = pd.read_csv(PURE_DL_OUTPUTS[3])
        pure_dl_comparison_df = pd.read_csv(PURE_DL_OUTPUTS[4])
        display(pure_dl_validation_df)
        display(pure_dl_comparison_df)
    else:
        def set_pure_dl_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
        def pure_dl_forward(model, batch):
            def move(key):
                value = batch.get(key)
                return value.to(DEVICE, non_blocking=True) if value is not None else None
            return model(
                move("v_dna"), move("v_prot"),
                move("bio_features"), move("geom_features"),
            )
    
        def pure_dl_probs(model, loader):
            model.eval()
            labels, probabilities = [], []
            with torch.no_grad():
                for batch in loader:
                    logits = pure_dl_forward(model, batch)
                    probabilities.extend(torch.sigmoid(logits).reshape(-1).cpu().numpy())
                    labels.extend(batch["label"].reshape(-1).cpu().numpy())
            return np.asarray(labels, dtype=np.int32), np.asarray(probabilities, dtype=np.float32)
    
        def build_pure_dl_model(target_config, dims, architecture, params):
            return MultiStrategyFusionModel(
                dna_in_dim=dims["dna"],
                prot_in_dim=dims["prot"],
                active_modalities=target_config["active_modalities"],
                bio_in_dim=dims["bio"],
                geom_in_dim=dims["geom"],
                fusion_strategy=architecture["fusion"],
                fusion_dim=params.get("fusion_dim", 512),
                dropout=params["dropout"],
            ).to(DEVICE)
    
        def fit_pure_dl_trial(target_config, dims, architecture, params, trial_checkpoint, history_rows, opt_trial=None):
            seed = int(CONFIG["seed"])
            set_pure_dl_seed(seed)
            train_loader.generator.manual_seed(seed)
            model = build_pure_dl_model(target_config, dims, architecture, params)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=params.get("learning_rate", CONFIG["lr"]), weight_decay=params["weight_decay"]
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=CONFIG["lr_factor"], patience=CONFIG["lr_patience"]
            )
            best_val_mcc_optimized = -float("inf")
            best_val_mcc_at_half = float("nan")
            best_epoch = 0
            best_train_loss = float("nan")
            best_val_loss = float("nan")
            epochs_without_improvement = 0
            started = time.perf_counter()
    
            for epoch in range(int(CONFIG["epochs"])):
                model.train()
                train_loss_sum = 0.0
                train_count = 0
                current_lr = float(optimizer.param_groups[0]["lr"])
                for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = pure_dl_forward(model, batch)
                    labels = batch["label"].to(DEVICE, non_blocking=True)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    count = int(labels.shape[0])
                    train_loss_sum += float(loss.item()) * count
                    train_count += count
                train_loss = train_loss_sum / max(train_count, 1)
    
                model.eval()
                val_loss_sum = 0.0
                val_count = 0
                val_labels, val_probs = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        logits = pure_dl_forward(model, batch)
                        labels = batch["label"].to(DEVICE, non_blocking=True)
                        val_loss_sum += float(criterion(logits, labels).item()) * int(labels.shape[0])
                        val_count += int(labels.shape[0])
                        val_probs.extend(torch.sigmoid(logits).reshape(-1).cpu().numpy())
                        val_labels.extend(labels.reshape(-1).cpu().numpy())
                val_loss = val_loss_sum / max(val_count, 1)
                epoch_labels = np.asarray(val_labels, dtype=np.int32)
                epoch_probs = np.asarray(val_probs, dtype=np.float32)
                epoch_preds = (epoch_probs >= 0.5).astype(np.int32)
                epoch_metrics = FusionEvaluatorProfiler.compute_metrics(epoch_labels, epoch_probs, epoch_preds)
                epoch_mcc_at_half = float(epoch_metrics["MCC"])
                epoch_threshold, epoch_mcc_optimized = pipeline._select_validation_threshold(epoch_labels, epoch_probs)
                epoch_metrics_optimized = FusionEvaluatorProfiler.compute_metrics(
                    epoch_labels, epoch_probs,
                    pipeline._threshold_predictions(epoch_probs, epoch_threshold),
                )
                if opt_trial is not None:
                    opt_trial.report(float(epoch_mcc_optimized), step=epoch + 1)
                    if opt_trial.should_prune():
                        return None, {"pruned": True, "best_epoch": epoch + 1, "val_mcc": float(epoch_mcc_optimized)}
                scheduler.step(epoch_mcc_optimized)
    
                history_rows.append({
                    "Target_Config": target_config["name"],
                    "Ablation": target_config["ablation"],
                    "Architecture": architecture["name"],
                    "Fusion_Dim": params.get("fusion_dim", 512),
                    "Dropout": params["dropout"],
                    "Weight_Decay": params["weight_decay"],
                    "Epoch": epoch + 1,
                    "Learning_Rate": current_lr,
                    "Train_Loss_Train_Mode": train_loss,
                    "Val_Loss_Eval_Mode": val_loss,
                    "Val_MCC_at_0_5": epoch_mcc_at_half,
                    "Val_MCC_at_Optimized_Threshold": epoch_mcc_optimized,
                    "Val_Threshold": epoch_threshold,
                    "Val_AUPRC": epoch_metrics["AUPRC"],
                })
    
                # Select checkpoints by the same validation-optimized threshold used in reporting.
                if epoch_mcc_optimized > best_val_mcc_optimized:
                    best_val_mcc_optimized = epoch_mcc_optimized
                    best_val_mcc_at_half = epoch_mcc_at_half
                    best_epoch = epoch + 1
                    best_train_loss = train_loss
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), trial_checkpoint)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= int(CONFIG["early_stop_patience"]):
                    break
    
            fit_seconds = time.perf_counter() - started
            if best_epoch == 0:
                raise RuntimeError(f"No valid checkpoint for {target_config['name']}/{architecture['name']}")
            model.load_state_dict(torch.load(trial_checkpoint, map_location=DEVICE, weights_only=True))
            y_val, val_prob = pure_dl_probs(model, val_loader)
            threshold, val_mcc = pipeline._select_validation_threshold(y_val, val_prob)
            val_predictions = pipeline._threshold_predictions(val_prob, threshold)
            val_metrics = FusionEvaluatorProfiler.compute_metrics(y_val, val_prob, val_predictions)
            return model, {
                "best_epoch": best_epoch,
                "best_val_mcc_at_0_5": best_val_mcc_at_half,
                "best_val_mcc_optimized": best_val_mcc_optimized,
                "train_loss": best_train_loss,
                "val_loss": best_val_loss,
                "fit_seconds": fit_seconds,
                "threshold": float(threshold),
                "val_mcc": float(val_mcc),
                "val_metrics": val_metrics,
            }
    
        pure_trial_rows = (pd.read_csv(PURE_DL_OUTPUTS[0]).to_dict("records")
                           if PURE_DL_OUTPUTS[0].exists() else [])
        pure_history_rows = (pd.read_csv(PURE_DL_OUTPUTS[3]).to_dict("records")
                             if PURE_DL_OUTPUTS[3].exists() else [])
        pure_validation_rows = []
        pure_winners = {}
        for target_config in TARGET_CONFIGS:
            config_name = target_config["name"]
            print(f"\\n[Pure DL Optuna/TPE] {config_name}")
            train_loader = pipeline._get_dataloader(
                "train3", "center", DNA_MODEL["name"], PROTEIN_MODEL["name"],
                target_config["active_modalities"], True,
                target_config["geom_feature_names"], "train3", CONFIG["seed"],
            )
            val_loader = pipeline._get_dataloader(
                "val", "center", DNA_MODEL["name"], PROTEIN_MODEL["name"],
                target_config["active_modalities"], False,
                target_config["geom_feature_names"], "train3", CONFIG["seed"],
            )
            train_dataset = train_loader.dataset
            dims = {
                "dna": int(train_dataset.v_dna_tensor.shape[1]) if hasattr(train_dataset, "v_dna_tensor") else 0,
                "prot": int(train_dataset.v_prot_tensor.shape[1]) if hasattr(train_dataset, "v_prot_tensor") else 0,
                "bio": int(train_dataset.bio_tensor.shape[1]) if hasattr(train_dataset, "bio_tensor") else 0,
                "geom": int(train_dataset.geom_tensor.shape[1]) if hasattr(train_dataset, "geom_tensor") else 0,
            }
            print(f"  input dimensions: {dims}")
    
            for architecture in PURE_DL_ARCHITECTURES:
                arch_key = (config_name, architecture["name"])
                arch_dir = PURE_DL_DIR / config_name / architecture["name"]
                arch_dir.mkdir(parents=True, exist_ok=True)
                best_score = (-float("inf"), -float("inf"))
                best_info = None
                best_model_path = arch_dir / "best_model.pth"
                temporary_checkpoint = arch_dir / "current_trial_best.pth"
    
                optuna_trials = optuna_trial_plan(arch_dir, f"pure_dl_{stage}_{config_name}_{architecture['name']}", PURE_DL_N_TRIALS, PURE_DL_SPACE, CONFIG["seed"])
                for trial_index, (study, opt_trial, params, is_new_trial) in enumerate(optuna_trials, start=1):
                    trial_label = f"optuna_{opt_trial.number:04d}"
                    trial_model_path = arch_dir / f"trial_{opt_trial.number:04d}.pth"
                    cached = next((row for row in pure_trial_rows
                                   if row.get("Target_Config") == config_name
                                   and row.get("Architecture") == architecture["name"]
                                   and row.get("Trial") == trial_label), None)
                    if cached is not None and trial_model_path.exists():
                        val_metrics = {key: cached[f"Val_{key}"] for key in
                                       ("Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "MCC", "AUROC", "AUPRC")}
                        fit_info = {
                            "best_epoch": int(cached["Best_Epoch"]),
                            "best_val_mcc_at_0_5": float(cached["Val_MCC_at_0_5"]),
                            "best_val_mcc_optimized": float(cached.get("Val_MCC_at_Optimized_Threshold", cached["Val_MCC_at_Threshold"])),
                            "train_loss": float(cached["Train_Loss_Train_Mode_at_Best_Epoch"]),
                            "val_loss": float(cached["Val_Loss_Eval_Mode_at_Best_Epoch"]),
                            "fit_seconds": float(cached["Fit_Seconds"]),
                            "threshold": float(cached["Decision_Threshold"]),
                            "val_mcc": float(cached["Val_MCC_at_Threshold"]),
                            "val_metrics": val_metrics,
                        }
                        model = build_pure_dl_model(target_config, dims, architecture, params)
                        model.load_state_dict(torch.load(trial_model_path, map_location=DEVICE, weights_only=True))
                        print(f"[RESUME] {config_name}/{architecture['name']}/{trial_label}")
                        tell_trial(study, opt_trial, fit_info["val_mcc"]) if is_new_trial else None
                    else:
                        fit_info_model, fit_info = None, None
                        model, fit_info = fit_pure_dl_trial(
                            target_config, dims, architecture, params,
                            temporary_checkpoint, pure_history_rows, opt_trial,
                        )
                        if fit_info.get("pruned"):
                            prune_trial(study, opt_trial)
                            save_progress_csv(pure_history_rows, PURE_DL_OUTPUTS[3])
                            print(f"  [PRUNED] {architecture['name']} trial {opt_trial.number}: MCC={fit_info['val_mcc']:.4f}")
                            continue
                        torch.save(model.state_dict(), trial_model_path)
                    score = (fit_info["val_mcc"], float(fit_info["val_metrics"]["AUPRC"]))
                    trial_row = {
                        "Target_Config": config_name,
                        "Ablation": target_config["ablation"],
                        "Architecture": architecture["name"],
                        "Trial": trial_label,
                        "Fusion_Dim": params.get("fusion_dim", 512),
                        "Dropout": params["dropout"],
                        "Weight_Decay": params["weight_decay"],
                        "Learning_Rate": params.get("learning_rate", CONFIG["lr"]),
                        "Batch_Size": CONFIG["batch_size"],
                        "Max_Epochs": CONFIG["epochs"],
                        "Best_Epoch": fit_info["best_epoch"],
                        "Early_Stop_Patience": CONFIG["early_stop_patience"],
                        "Train_Loss_Train_Mode_at_Best_Epoch": fit_info["train_loss"],
                        "Val_Loss_Eval_Mode_at_Best_Epoch": fit_info["val_loss"],
                        "Val_MCC_at_0_5": fit_info["best_val_mcc_at_0_5"],
                        "Val_MCC_at_Optimized_Threshold": fit_info["best_val_mcc_optimized"],
                        "Decision_Threshold": fit_info["threshold"],
                        "Val_MCC_at_Threshold": fit_info["val_mcc"],
                        **{f"Val_{key}": value for key, value in fit_info["val_metrics"].items()},
                        "Fit_Seconds": round(fit_info["fit_seconds"], 3),
                    }
                    pure_trial_rows = upsert_record(
                        pure_trial_rows, trial_row, ("Target_Config", "Architecture", "Trial")
                    )
                    save_progress_csv(pure_trial_rows, PURE_DL_OUTPUTS[0])
                    save_progress_csv(pure_history_rows, PURE_DL_OUTPUTS[3])
                    tell_trial(study, opt_trial, fit_info["val_mcc"]) if is_new_trial else None
                    if score > best_score:
                        best_score = score
                        best_info = {**fit_info, "params": dict(params), "trial": trial_label}
                        torch.save(model.state_dict(), best_model_path)
                    del model
                    print(
                        f"  {architecture['name']} [{trial_index}/{PURE_DL_N_TRIALS}] "
                        f"fusion_dim={params.get('fusion_dim', 512)} dropout={params['dropout']} "
                        f"wd={params['weight_decay']} lr={params.get('learning_rate', CONFIG['lr'])} "
                        f"Val_MCC={fit_info['val_mcc']:.4f} "
                        f"(epoch={fit_info['best_epoch']}, val_loss={fit_info['val_loss']:.4f})"
                    )
    
                if best_info is None:
                    raise RuntimeError(f"No DL Optuna result for {config_name}/{architecture['name']}")
                torch.save({"params": best_info["params"], "trial": best_info["trial"], "dims": dims}, arch_dir / "best_model_metadata.pt")
                pure_validation_rows.append({
                    "Target_Config": config_name,
                    "Ablation": target_config["ablation"],
                    "Architecture": architecture["name"],
                    "Model": f"{architecture['name']}_GridBest",
                    "Selected_Trial": best_info["trial"],
                    "Fusion_Dim": best_info["params"].get("fusion_dim", 512),
                    "Dropout": best_info["params"]["dropout"],
                    "Weight_Decay": best_info["params"]["weight_decay"],
                    "Learning_Rate": best_info["params"].get("learning_rate", CONFIG["lr"]),
                    "Batch_Size": CONFIG["batch_size"],
                    "Best_Epoch": best_info["best_epoch"],
                    "Train_Loss_at_Best_Epoch": best_info["train_loss"],
                    "Val_Loss_at_Best_Epoch": best_info["val_loss"],
                    "Val_Train_Loss_Gap": best_info["val_loss"] - best_info["train_loss"],
                    "Val_MCC_at_0_5": best_info["best_val_mcc_at_0_5"],
                    "Val_MCC_at_Optimized_Threshold": best_info["best_val_mcc_optimized"],
                    "Decision_Threshold": best_info["threshold"],
                    **best_info["val_metrics"],
                    "Fit_Seconds": round(best_info["fit_seconds"], 3),
                })
                winner_model = build_pure_dl_model(
                    target_config, dims, architecture, best_info["params"]
                )
                winner_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
                winner_model.eval()
                pure_winners[arch_key] = {
                    "model": winner_model,
                    "target_config": target_config,
                    "architecture": architecture,
                    "threshold": best_info["threshold"],
                    "params": best_info["params"],
                    "trial": best_info["trial"],
                }
                if temporary_checkpoint.exists():
                    temporary_checkpoint.unlink()
            del train_loader, val_loader, train_dataset
    
        pure_test_rows = []
        for target_config in TARGET_CONFIGS:
            config_name = target_config["name"]
            config_winners = [
                (key, item) for key, item in pure_winners.items() if key[0] == config_name
            ]
            for dataset_config in datasets:
                split_name = dataset_config["test"]
                test_loader = pipeline._get_dataloader(
                    split_name, "center", DNA_MODEL["name"], PROTEIN_MODEL["name"],
                    target_config["active_modalities"], False,
                    target_config["geom_feature_names"], "train3", CONFIG["seed"],
                )
                y_true = []
                probabilities = {key[1]: [] for key, _ in config_winners}
                with torch.no_grad():
                    for batch in test_loader:
                        y_true.extend(batch["label"].reshape(-1).cpu().numpy())
                        for key, winner in config_winners:
                            logits = pure_dl_forward(winner["model"], batch)
                            probabilities[key[1]].extend(torch.sigmoid(logits).reshape(-1).cpu().numpy())
                y_true = np.asarray(y_true, dtype=np.int32)
                for key, winner in config_winners:
                    architecture_name = key[1]
                    probs = np.asarray(probabilities[architecture_name], dtype=np.float32)
                    preds = pipeline._threshold_predictions(probs, winner["threshold"])
                    metrics = FusionEvaluatorProfiler.compute_metrics(y_true, probs, preds)
                    pure_test_rows.append({
                        "Target_Config": config_name,
                        "Ablation": target_config["ablation"],
                        "Dataset": dataset_config["name"],
                        "Architecture": architecture_name,
                        "Model": f"{architecture_name}_GridBest",
                        "Selected_Trial": winner["trial"],
                        "Fusion_Dim": winner["params"].get("fusion_dim", 512),
                        "Dropout": winner["params"]["dropout"],
                        "Weight_Decay": winner["params"]["weight_decay"],
                        "Learning_Rate": winner["params"].get("learning_rate", CONFIG["lr"]),
                        "Decision_Threshold": winner["threshold"],
                        **metrics,
                    })
                    pred_dir = PURE_DL_DIR / config_name / architecture_name / "tests" / dataset_config["name"]
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({
                        "True_Label": y_true,
                        "Predicted_Probability": probs,
                        "Decision_Threshold": winner["threshold"],
                        "Prediction_Class": preds,
                    }).to_csv(pred_dir / "predictions.csv", index=False)
                del test_loader
    
        pure_dl_trials_df = pd.DataFrame(pure_trial_rows)
        pure_dl_validation_df = pd.DataFrame(pure_validation_rows)
        pure_dl_test_df = pd.DataFrame(pure_test_rows)
        pure_dl_history_df = pd.DataFrame(pure_history_rows)
        pure_dl_trials_df.to_csv(PURE_DL_OUTPUTS[0], index=False)
        pure_dl_validation_df.to_csv(PURE_DL_OUTPUTS[1], index=False)
        pure_dl_test_df.to_csv(PURE_DL_OUTPUTS[2], index=False)
        pure_dl_history_df.to_csv(PURE_DL_OUTPUTS[3], index=False)
    
        baseline_df = pd.read_csv(Path(pipeline.batch_run_dir) / "fixed_config_baseline.csv")
        baseline_df = baseline_df.loc[
            baseline_df["Network"].isin([arch["name"] for arch in PURE_DL_ARCHITECTURES])
        ].copy()
        baseline_df["Target_Config"] = baseline_df["Ablation"].map(
            {cfg["ablation"]: cfg["name"] for cfg in TARGET_CONFIGS}
        )
        baseline_df["Model"] = baseline_df["Network"] + "_Baseline"
        baseline_cols = ["Target_Config", "Ablation", "Dataset", "Model", "Decision_Threshold", "MCC", "AUROC", "AUPRC"]
        grid_cols = ["Target_Config", "Ablation", "Dataset", "Model", "Decision_Threshold", "MCC", "AUROC", "AUPRC"]
        pure_dl_comparison_df = pd.concat(
            [baseline_df[baseline_cols], pure_dl_test_df[grid_cols]], ignore_index=True
        ).sort_values(["Target_Config", "Dataset", "Model"])
        pure_dl_comparison_df.to_csv(PURE_DL_OUTPUTS[4], index=False)
        print(f"\\n[COMPLETE] Pure-DL Optuna search saved in {PURE_DL_DIR}")
        print("Grid trials:", len(pure_dl_trials_df), "| expected 288 | epoch history rows:", len(pure_dl_history_df))
        display(pure_dl_validation_df)
        display(pure_dl_comparison_df)
    
    # ==============================================================================
