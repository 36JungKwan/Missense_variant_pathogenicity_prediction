from .resume_utils import ensure_stage_signature, save_progress_csv, upsert_record
from .optuna_search import optuna_trial_plan, tell_trial, prune_trial
from .dl_architecture_model import FusionArchitectureVariant
import pandas as pd

"""Architecture-variant and training-hyperparameter searches for pure deep learning."""

def run_architecture_stage(context, search_config, display_fn=None):
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
    # DL STAGE 6/7: Architecture and Optuna training-hyperparameter search
    # ==============================================================================
    import json
    import random
    import time
    from pathlib import Path
    
    import numpy as np
    import torch
    import torch.nn as nn
    
    from core.module05v2_fusion_classifier.evaluator_profiler import FusionEvaluatorProfiler
    
    DL_ARCH_VARIANT_DIR = Path(pipeline.batch_run_dir) / "pure_dl_architecture_hparam_search_v1"
    DL_ARCH_VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    DL_ARCH_VARIANT_OUTPUTS = [
        DL_ARCH_VARIANT_DIR / "architecture_variant_trials.csv",
        DL_ARCH_VARIANT_DIR / "architecture_variant_validation_metrics.csv",
        DL_ARCH_VARIANT_DIR / "architecture_variant_test_metrics.csv",
        DL_ARCH_VARIANT_DIR / "architecture_variant_epoch_history.csv",
        DL_ARCH_VARIANT_DIR / "architecture_variant_validation_comparison.csv",
        DL_ARCH_VARIANT_DIR / "architecture_variant_test_comparison.csv",
    ]
    DL_HPARAM_DIR = DL_ARCH_VARIANT_DIR / "training_hparam_optuna"
    DL_HPARAM_DIR.mkdir(parents=True, exist_ok=True)
    DL_HPARAM_OUTPUTS = [
        DL_HPARAM_DIR / "training_optuna_trials.csv",
        DL_HPARAM_DIR / "training_optuna_epoch_history.csv",
        DL_HPARAM_DIR / "training_optuna_best_validation.csv",
        DL_HPARAM_DIR / "training_optuna_best_test.csv",
    ]
    DL_ARCH_VARIANTS = DL_SEARCH_CONFIG['architecture_hparam_search_v1']

    confirmation_candidate_keys = {
        (str(candidate["Target_Config"]), str(candidate["Architecture"]), str(candidate["Variant"]))
        for candidate in DL_SEARCH_CONFIG.get("confirmation_candidates", [])
    }

    def has_confirmation_hparam_trials():
        if not confirmation_candidate_keys:
            return True
        if not DL_HPARAM_OUTPUTS[0].exists():
            return False
        try:
            trial_keys = pd.read_csv(
                DL_HPARAM_OUTPUTS[0],
                usecols=["Target_Config", "Architecture", "Variant"],
            )
        except (ValueError, pd.errors.EmptyDataError):
            return False
        completed_keys = set(
            trial_keys.astype(str).itertuples(index=False, name=None)
        )
        return confirmation_candidate_keys.issubset(completed_keys)
    
    def build_architecture_variant_comparisons(variant_validation_df, variant_test_df):
        tuned_root = Path(pipeline.batch_run_dir) / "pure_dl_regularization_grid_v2"
        original_validation = pd.read_csv(tuned_root / "pure_dl_validation_metrics.csv")
        original_validation["Variant"] = "original"
        original_validation["Model"] = original_validation["Architecture"] + "_original"
    
        variant_validation_compare = variant_validation_df.copy()
        variant_validation_compare["Model"] = (
            variant_validation_compare["Architecture"] + "_" + variant_validation_compare["Variant"]
        )
        # The variant table stores validation scores with a Val_ prefix.
        for metric in ("MCC", "AUROC", "AUPRC"):
            variant_validation_compare[metric] = variant_validation_compare[f"Val_{metric}"]
        compare_validation_cols = [
            "Target_Config", "Ablation", "Architecture", "Variant", "Model",
            "Decision_Threshold", "MCC", "AUROC", "AUPRC",
        ]
        validation_comparison = pd.concat([
            original_validation[compare_validation_cols],
            variant_validation_compare[compare_validation_cols],
        ], ignore_index=True).sort_values(["Target_Config", "Architecture", "Model"])
    
        original_test = pd.read_csv(tuned_root / "pure_dl_test_metrics.csv")
        original_test["Variant"] = "original"
        original_test["Model"] = original_test["Architecture"] + "_original"
        variant_test_compare = variant_test_df.copy()
        compare_test_cols = [
            "Target_Config", "Ablation", "Dataset", "Architecture", "Variant",
            "Model", "Decision_Threshold", "MCC", "AUROC", "AUPRC",
        ]
        test_comparison = pd.concat([
            original_test[compare_test_cols], variant_test_compare[compare_test_cols],
        ], ignore_index=True).sort_values(["Target_Config", "Architecture", "Dataset", "Model"])
        return validation_comparison, test_comparison
    
    
    ensure_stage_signature(DL_ARCH_VARIANT_DIR, "architecture_hparam_search_v1", {
        "variants": DL_ARCH_VARIANTS, "sampler": "TPE",
        "hparam_space": DL_SEARCH_CONFIG.get("training_hparam_space_v1"),
        "hparam_n_trials": DL_SEARCH_CONFIG.get("n_trials", {}).get("training_hparam", 16),
        "search_config": DL_SEARCH_CONFIG, "config": CONFIG, "targets": TARGET_CONFIGS, "datasets": datasets,
        "dna_model": DNA_MODEL, "protein_model": PROTEIN_MODEL,
    })
    if (
        all(path.exists() for path in DL_ARCH_VARIANT_OUTPUTS[:4])
        and has_confirmation_hparam_trials()
    ):
        print(f"[REUSE] Reusing completed architecture model runs: {DL_ARCH_VARIANT_DIR}")
        dl_arch_trials_df = pd.read_csv(DL_ARCH_VARIANT_OUTPUTS[0])
        dl_arch_validation_df = pd.read_csv(DL_ARCH_VARIANT_OUTPUTS[1])
        dl_arch_test_df = pd.read_csv(DL_ARCH_VARIANT_OUTPUTS[2])
        dl_arch_history_df = pd.read_csv(DL_ARCH_VARIANT_OUTPUTS[3])
        if all(path.exists() for path in DL_ARCH_VARIANT_OUTPUTS[4:]):
            dl_arch_validation_comparison_df = pd.read_csv(DL_ARCH_VARIANT_OUTPUTS[4])
            dl_arch_test_comparison_df = pd.read_csv(DL_ARCH_VARIANT_OUTPUTS[5])
        else:
            # Recover the comparison tables without retraining the completed variants.
            dl_arch_validation_comparison_df, dl_arch_test_comparison_df = build_architecture_variant_comparisons(
                dl_arch_validation_df, dl_arch_test_df
            )
            dl_arch_validation_comparison_df.to_csv(DL_ARCH_VARIANT_OUTPUTS[4], index=False)
            dl_arch_test_comparison_df.to_csv(DL_ARCH_VARIANT_OUTPUTS[5], index=False)
        display(dl_arch_validation_comparison_df)
        display(dl_arch_test_comparison_df)
        if all(path.exists() for path in DL_HPARAM_OUTPUTS):
            print(f"[REUSE] Reusing completed training Optuna search: {DL_HPARAM_DIR}")
            dl_hparam_trials_df = pd.read_csv(DL_HPARAM_OUTPUTS[0])
            dl_hparam_history_df = pd.read_csv(DL_HPARAM_OUTPUTS[1])
            dl_hparam_best_validation_df = pd.read_csv(DL_HPARAM_OUTPUTS[2])
            dl_hparam_best_test_df = pd.read_csv(DL_HPARAM_OUTPUTS[3])
            display(dl_hparam_best_validation_df)
            display(dl_hparam_best_test_df)
    else:
    
        def set_variant_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
        def variant_forward(model, batch, augmentation=None, input_stds=None):
            def move(key):
                value = batch.get(key)
                return value.to(DEVICE, non_blocking=True) if value is not None else None
            v_dna, v_prot = move("v_dna"), move("v_prot")
            bio, geom = move("bio_features"), move("geom_features")
            augmentation = augmentation or {}
            if model.training:
                for value_name, value, rate_name in (
                    ("dna", v_dna, "dna_feature_dropout"),
                    ("protein", v_prot, "protein_feature_dropout"),
                    ("bio", bio, "bio_feature_dropout"),
                    ("geom", geom, "geom_feature_dropout"),
                ):
                    rate = float(augmentation.get(rate_name, augmentation.get("feature_dropout", 0.0)))
                    if value is not None and rate > 0:
                        value = torch.nn.functional.dropout(value, p=rate, training=True)
                    if value_name == "dna": v_dna = value
                    elif value_name == "protein": v_prot = value
                    elif value_name == "bio": bio = value
                    else: geom = value
                modality_dropout = float(augmentation.get("modality_dropout", 0.0))
                if modality_dropout > 0:
                    keep_dna = torch.rand((v_dna.shape[0], 1), device=DEVICE) >= modality_dropout
                    keep_prot = torch.rand((v_prot.shape[0], 1), device=DEVICE) >= modality_dropout
                    both_dropped = ~(keep_dna | keep_prot)
                    if both_dropped.any():
                        restore_dna = torch.rand((v_dna.shape[0], 1), device=DEVICE) < 0.5
                        keep_dna = keep_dna | (both_dropped & restore_dna)
                        keep_prot = keep_prot | (both_dropped & ~restore_dna)
                    v_dna = v_dna * keep_dna
                    v_prot = v_prot * keep_prot
                noise_scale = float(augmentation.get("gaussian_noise", 0.0))
                if noise_scale > 0 and input_stds is not None:
                    v_dna = v_dna + torch.randn_like(v_dna) * input_stds[0] * noise_scale
                    v_prot = v_prot + torch.randn_like(v_prot) * input_stds[1] * noise_scale
                for value_name, value, scale_name in (
                    ("bio", bio, "bio_gaussian_noise"), ("geom", geom, "geom_gaussian_noise")
                ):
                    scale = float(augmentation.get(scale_name, 0.0))
                    if value is not None and scale > 0:
                        value = value + torch.randn_like(value) * value.float().std(unbiased=False).clamp_min(1e-8) * scale
                    if value_name == "bio": bio = value
                    else: geom = value
            return model(v_dna, v_prot, bio, geom)
    
        def variant_eval(model, loader, criterion=None):
            model.eval()
            loss_sum, sample_count = 0.0, 0
            labels, probs = [], []
            with torch.no_grad():
                for batch in loader:
                    logits = variant_forward(model, batch)
                    y = batch["label"].to(DEVICE, non_blocking=True)
                    if criterion is not None:
                        loss_sum += float(criterion(logits, y).item()) * int(y.shape[0])
                        sample_count += int(y.shape[0])
                    labels.extend(y.reshape(-1).cpu().numpy())
                    probs.extend(torch.sigmoid(logits).reshape(-1).cpu().numpy())
            mean_loss = loss_sum / max(sample_count, 1) if criterion is not None else np.nan
            return mean_loss, np.asarray(labels, dtype=np.int32), np.asarray(probs, dtype=np.float32)
    
        tuned_path = Path(pipeline.batch_run_dir) / "pure_dl_regularization_grid_v2" / "pure_dl_validation_metrics.csv"
        tuned_validation = pd.read_csv(tuned_path)
        variant_trial_rows = (pd.read_csv(DL_ARCH_VARIANT_OUTPUTS[0]).to_dict("records")
                              if DL_ARCH_VARIANT_OUTPUTS[0].exists() else [])
        variant_validation_rows = (pd.read_csv(DL_ARCH_VARIANT_OUTPUTS[1]).to_dict("records")
                                   if DL_ARCH_VARIANT_OUTPUTS[1].exists() else [])
        variant_history_rows = (pd.read_csv(DL_ARCH_VARIANT_OUTPUTS[3]).to_dict("records")
                                if DL_ARCH_VARIANT_OUTPUTS[3].exists() else [])
        variant_winners = {}
        criterion = nn.BCEWithLogitsLoss()
    
        for target_config in TARGET_CONFIGS:
            config_name = target_config["name"]
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
            dataset = train_loader.dataset
            dims = {
                "dna": int(dataset.v_dna_tensor.shape[1]),
                "prot": int(dataset.v_prot_tensor.shape[1]),
                "bio": int(dataset.bio_tensor.shape[1]) if hasattr(dataset, "bio_tensor") else 0,
                "geom": int(dataset.geom_tensor.shape[1]) if hasattr(dataset, "geom_tensor") else 0,
            }
    
            for architecture, variants in DL_ARCH_VARIANTS.items():
                tuned_rows = tuned_validation.loc[
                    tuned_validation["Target_Config"].eq(config_name)
                    & tuned_validation["Architecture"].eq(architecture)
                ]
                if len(tuned_rows) != 1:
                    raise ValueError(f"Expected one tuned row for {config_name}/{architecture}; found {len(tuned_rows)}")
                tuned = tuned_rows.iloc[0]
                params = {
                    "fusion_dim": int(tuned["Fusion_Dim"]),
                    "dropout": float(tuned["Dropout"]),
                    "weight_decay": float(tuned["Weight_Decay"]),
                    "learning_rate": float(tuned["Learning_Rate"]),
                }
                for variant in variants:
                    variant_dir = DL_ARCH_VARIANT_DIR / config_name / architecture / variant
                    variant_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint = variant_dir / "current_best.pth"
                    set_variant_seed(int(CONFIG["seed"]))
                    model = FusionArchitectureVariant(
                        dna_dim=dims["dna"], prot_dim=dims["prot"],
                        bio_dim=dims["bio"], geom_dim=dims["geom"],
                        active_modalities=target_config["active_modalities"],
                        variant=variant, fusion_dim=params["fusion_dim"], dropout=params["dropout"],
                    ).to(DEVICE)
                    winner_path = variant_dir / "best_model.pth"
                    if winner_path.exists():
                        # Reuse a completed checkpoint if a prior cell run stopped partway through.
                        model.load_state_dict(torch.load(winner_path, map_location=DEVICE, weights_only=True))
                        train_eval_loss, _, _ = variant_eval(model, train_loader, criterion)
                        val_eval_loss, y_val, p_val = variant_eval(model, val_loader, criterion)
                        threshold, val_mcc = pipeline._select_validation_threshold(y_val, p_val)
                        val_predictions = pipeline._threshold_predictions(p_val, threshold)
                        val_metrics = FusionEvaluatorProfiler.compute_metrics(y_val, p_val, val_predictions)
                        row = {
                            "Target_Config": config_name, "Ablation": target_config["ablation"],
                            "Architecture": architecture, "Variant": variant, **params,
                            "Best_Epoch": np.nan, "Train_Loss_Train_Mode_at_Best": np.nan,
                            "Train_Loss_Eval_Mode": train_eval_loss, "Val_Loss_Eval_Mode": val_eval_loss,
                            "Eval_Mode_Val_Train_Loss_Gap": val_eval_loss - train_eval_loss,
                            "Val_MCC_at_Optimized_Threshold": float(val_mcc),
                            "Decision_Threshold": float(threshold), "Val_MCC_at_Threshold": float(val_mcc),
                            **{f"Val_{key}": value for key, value in val_metrics.items()},
                            "Fit_Seconds": np.nan, "Resumed_From_Checkpoint": True,
                        }
                        variant_trial_rows = upsert_record(variant_trial_rows, row, ("Target_Config", "Architecture", "Variant"))
                        variant_validation_rows = upsert_record(variant_validation_rows, row.copy(), ("Target_Config", "Architecture", "Variant"))
                        save_progress_csv(variant_trial_rows, DL_ARCH_VARIANT_OUTPUTS[0])
                        save_progress_csv(variant_validation_rows, DL_ARCH_VARIANT_OUTPUTS[1])
                        save_progress_csv(variant_history_rows, DL_ARCH_VARIANT_OUTPUTS[3])
                        model.to("cpu").eval()
                        variant_winners[(config_name, architecture, variant)] = {
                            "model": model, "variant": variant, "params": params,
                            "threshold": float(threshold),
                        }
                        print(f"[RESUME] {config_name}/{architecture}/{variant}; MCC_val={val_mcc:.4f}")
                        continue
                    train_loader.generator.manual_seed(int(CONFIG["seed"]))
                    optimizer = torch.optim.AdamW(
                        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
                    )
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="max", factor=CONFIG["lr_factor"], patience=CONFIG["lr_patience"]
                    )
                    best_mcc_half = -float("inf")
                    best_epoch = 0
                    best_train_mode_loss = np.nan
                    no_improve = 0
                    started = time.perf_counter()
    
                    for epoch in range(int(CONFIG["epochs"])):
                        model.train()
                        train_sum, train_count = 0.0, 0
                        lr_now = float(optimizer.param_groups[0]["lr"])
                        for batch in train_loader:
                            optimizer.zero_grad(set_to_none=True)
                            logits = variant_forward(model, batch)
                            y = batch["label"].to(DEVICE, non_blocking=True)
                            loss = criterion(logits, y)
                            loss.backward()
                            optimizer.step()
                            n = int(y.shape[0])
                            train_sum += float(loss.item()) * n
                            train_count += n
                        train_mode_loss = train_sum / max(train_count, 1)
                        val_loss, y_val_epoch, p_val_epoch = variant_eval(model, val_loader, criterion)
                        epoch_threshold, epoch_val_mcc = pipeline._select_validation_threshold(y_val_epoch, p_val_epoch)
                        val_predictions_epoch = pipeline._threshold_predictions(p_val_epoch, epoch_threshold)
                        metrics_epoch = FusionEvaluatorProfiler.compute_metrics(
                            y_val_epoch, p_val_epoch, val_predictions_epoch
                        )
                        val_mcc_half = float(epoch_val_mcc)
                        scheduler.step(val_mcc_half)
                        variant_history_rows.append({
                            "Target_Config": config_name, "Ablation": target_config["ablation"],
                            "Architecture": architecture, "Variant": variant,
                            "Epoch": epoch + 1, "Learning_Rate": lr_now,
                            "Train_Loss_Train_Mode": train_mode_loss,
                            "Val_Loss_Eval_Mode": val_loss,
                            "Val_MCC_at_Optimized_Threshold": val_mcc_half,
                            "Val_Threshold": float(epoch_threshold),
                            "Val_AUROC": float(metrics_epoch["AUROC"]),
                            "Val_AUPRC": float(metrics_epoch["AUPRC"]),
                        })
                        if val_mcc_half > best_mcc_half:
                            best_mcc_half = val_mcc_half
                            best_epoch = epoch + 1
                            best_train_mode_loss = train_mode_loss
                            torch.save(model.state_dict(), checkpoint)
                            no_improve = 0
                        else:
                            no_improve += 1
                        if no_improve >= int(CONFIG["early_stop_patience"]):
                            break
    
                    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
                    train_eval_loss, _, _ = variant_eval(model, train_loader, criterion)
                    val_eval_loss, y_val, p_val = variant_eval(model, val_loader, criterion)
                    threshold, val_mcc = pipeline._select_validation_threshold(y_val, p_val)
                    val_predictions = pipeline._threshold_predictions(p_val, threshold)
                    val_metrics = FusionEvaluatorProfiler.compute_metrics(y_val, p_val, val_predictions)
                    fit_seconds = time.perf_counter() - started
                    row = {
                        "Target_Config": config_name, "Ablation": target_config["ablation"],
                        "Architecture": architecture, "Variant": variant, **params,
                        "Best_Epoch": best_epoch,
                        "Train_Loss_Train_Mode_at_Best": best_train_mode_loss,
                        "Train_Loss_Eval_Mode": train_eval_loss,
                        "Val_Loss_Eval_Mode": val_eval_loss,
                        "Eval_Mode_Val_Train_Loss_Gap": val_eval_loss - train_eval_loss,
                        "Val_MCC_at_Optimized_Threshold": best_mcc_half,
                        "Decision_Threshold": float(threshold),
                        "Val_MCC_at_Threshold": float(val_mcc),
                        **{f"Val_{key}": value for key, value in val_metrics.items()},
                        "Fit_Seconds": round(fit_seconds, 3), "Resumed_From_Checkpoint": False,
                    }
                    variant_trial_rows = upsert_record(variant_trial_rows, row, ("Target_Config", "Architecture", "Variant"))
                    variant_validation_rows = upsert_record(variant_validation_rows, row.copy(), ("Target_Config", "Architecture", "Variant"))
                    save_progress_csv(variant_trial_rows, DL_ARCH_VARIANT_OUTPUTS[0])
                    save_progress_csv(variant_validation_rows, DL_ARCH_VARIANT_OUTPUTS[1])
                    save_progress_csv(variant_history_rows, DL_ARCH_VARIANT_OUTPUTS[3])
                    winner_path = variant_dir / "best_model.pth"
                    torch.save(model.state_dict(), winner_path)
                    model.to("cpu").eval()
                    variant_winners[(config_name, architecture, variant)] = {
                        "model": model, "variant": variant, "params": params,
                        "threshold": float(threshold),
                    }
                    if checkpoint.exists():
                        checkpoint.unlink()
                    print(
                        f"[{config_name}/{architecture}/{variant}] MCC_val={val_mcc:.4f}; "
                        f"threshold={threshold:.4f}; epoch={best_epoch}; "
                        f"eval_loss_gap={val_eval_loss - train_eval_loss:.4f}"
                    )
                    del model
            del train_loader, val_loader, dataset
    
        variant_test_rows = []
        for target_config in TARGET_CONFIGS:
            config_name = target_config["name"]
            config_winners = [(key, item) for key, item in variant_winners.items() if key[0] == config_name]
            for _, winner in config_winners:
                winner["model"].to(DEVICE).eval()
            for dataset_config in datasets:
                test_loader = pipeline._get_dataloader(
                    dataset_config["test"], "center", DNA_MODEL["name"], PROTEIN_MODEL["name"],
                    target_config["active_modalities"], False,
                    target_config["geom_feature_names"], "train3", CONFIG["seed"],
                )
                y_true, variant_ids = [], []
                probabilities = {key[1:]: [] for key, _ in config_winners}
                with torch.no_grad():
                    for batch in test_loader:
                        y_true.extend(batch["label"].reshape(-1).cpu().numpy())
                        variant_ids.extend(batch["variant_id"])
                        for key, winner in config_winners:
                            logits = variant_forward(winner["model"], batch)
                            probabilities[key[1:]].extend(torch.sigmoid(logits).reshape(-1).cpu().numpy())
                y_true = np.asarray(y_true, dtype=np.int32)
                for key, winner in config_winners:
                    architecture, variant = key[1], key[2]
                    probs = np.asarray(probabilities[(architecture, variant)], dtype=np.float32)
                    predictions = pipeline._threshold_predictions(probs, winner["threshold"])
                    metrics = FusionEvaluatorProfiler.compute_metrics(y_true, probs, predictions)
                    variant_test_rows.append({
                        "Target_Config": config_name, "Ablation": target_config["ablation"],
                        "Dataset": dataset_config["name"], "Architecture": architecture,
                        "Variant": variant, "Model": f"{architecture}_{variant}",
                        "Decision_Threshold": winner["threshold"], **metrics,
                    })
                    pred_dir = DL_ARCH_VARIANT_DIR / config_name / architecture / variant / "tests" / dataset_config["name"]
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({
                        "Variant_ID": variant_ids, "True_Label": y_true,
                        "Predicted_Probability": probs, "Decision_Threshold": winner["threshold"],
                        "Prediction_Class": predictions,
                    }).to_csv(pred_dir / "predictions.csv", index=False)
                del test_loader
            for _, winner in config_winners:
                winner["model"].to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
        # Tune training factors only after the 100-variant structural screen.
        # The top two structures per fusion family/config enter a bounded TPE search.
        architecture_trials_for_tuning = pd.DataFrame(variant_trial_rows)
        duplicate_aliases = architecture_trials_for_tuning["Variant"].eq("cross_8token_attention_shared")
        duplicate_aliases |= (
            architecture_trials_for_tuning["Target_Config"].eq("A_no_geometry")
            & architecture_trials_for_tuning["Variant"].eq("transformer_separate_feature_tokens_2layer")
        )
        tuning_candidates = architecture_trials_for_tuning.loc[~duplicate_aliases].copy()
        tuning_candidates["Val_MCC_at_Threshold"] = pd.to_numeric(
            tuning_candidates["Val_MCC_at_Threshold"], errors="coerce"
        )
        top_tuning_candidates = (
            tuning_candidates.sort_values("Val_MCC_at_Threshold", ascending=False)
            .groupby(["Target_Config", "Architecture"], as_index=False, group_keys=False)
            .head(2)
        )
        candidate_columns = ["Target_Config", "Architecture", "Variant"]
        confirmation_keys = {
            (str(candidate["Target_Config"]), str(candidate["Architecture"]), str(candidate["Variant"]))
            for candidate in DL_SEARCH_CONFIG.get("confirmation_candidates", [])
        }
        available_keys = set(
            architecture_trials_for_tuning[candidate_columns]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        missing_confirmation_candidates = confirmation_keys - available_keys
        if missing_confirmation_candidates:
            raise ValueError(
                "Seed-confirmation candidates are missing from the architecture screen: "
                f"{sorted(missing_confirmation_candidates)}"
            )
        confirmation_rows = architecture_trials_for_tuning.loc[
            architecture_trials_for_tuning[candidate_columns]
            .astype(str)
            .apply(tuple, axis=1)
            .isin(confirmation_keys)
        ]
        tuning_candidates = pd.concat(
            [top_tuning_candidates, confirmation_rows], ignore_index=True
        ).drop_duplicates(candidate_columns, keep="first")
        hparam_space = DL_SEARCH_CONFIG["training_hparam_space_v1"]
        hparam_n_trials = int(DL_SEARCH_CONFIG.get("n_trials", {}).get("training_hparam", 16))
        hparam_trial_rows = (pd.read_csv(DL_HPARAM_OUTPUTS[0]).to_dict("records")
                             if DL_HPARAM_OUTPUTS[0].exists() else [])
        hparam_history_rows = (pd.read_csv(DL_HPARAM_OUTPUTS[1]).to_dict("records")
                               if DL_HPARAM_OUTPUTS[1].exists() else [])
        hparam_checkpoints = {}
        criterion_default = nn.BCEWithLogitsLoss()
    
        def hparam_loss(logits, labels, loss_type, pos_weight, positive_fraction, overrides):
            smoothing = float(overrides.get("label_smoothing", 0.0))
            if smoothing:
                labels = labels * (1.0 - smoothing) + 0.5 * smoothing
            if loss_type == "bce":
                return nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if loss_type == "weighted_bce":
                scaled_weight = pos_weight * float(overrides.get("pos_weight_multiplier", 1.0))
                return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=scaled_weight)
            per_item = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            probability_correct = torch.exp(-per_item)
            alpha_positive = float(overrides.get("focal_alpha", 1.0 - positive_fraction))
            alpha_negative = 1.0 - alpha_positive
            alpha = labels * alpha_positive + (1.0 - labels) * alpha_negative
            gamma = float(overrides.get("focal_gamma", 2.0))
            return (alpha * (1.0 - probability_correct).pow(gamma) * per_item).mean()

        def with_batch_size(loader, batch_size, seed):
            from torch.utils.data import DataLoader
            return DataLoader(
                loader.dataset, batch_size=int(batch_size), shuffle=True,
                num_workers=loader.num_workers, pin_memory=getattr(loader, "pin_memory", False),
                drop_last=False, generator=torch.Generator().manual_seed(int(seed)),
                collate_fn=loader.collate_fn,
            )
    
        for target_config in TARGET_CONFIGS:
            config_name = target_config["name"]
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
            dataset = train_loader.dataset
            dims = {
                "dna": int(dataset.v_dna_tensor.shape[1]),
                "prot": int(dataset.v_prot_tensor.shape[1]),
                "bio": int(dataset.bio_tensor.shape[1]) if hasattr(dataset, "bio_tensor") else 0,
                "geom": int(dataset.geom_tensor.shape[1]) if hasattr(dataset, "geom_tensor") else 0,
            }
            input_stds = (
                float(dataset.v_dna_tensor.float().std(unbiased=False).item()),
                float(dataset.v_prot_tensor.float().std(unbiased=False).item()),
            )
            labels = dataset.labels.float()
            positive_count = float(labels.sum().item())
            negative_count = float(labels.numel() - positive_count)
            positive_fraction = positive_count / max(positive_count + negative_count, 1.0)
            pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], device=DEVICE)
            config_candidates = tuning_candidates.loc[tuning_candidates["Target_Config"].eq(config_name)]
            for _, candidate in config_candidates.iterrows():
                architecture = candidate["Architecture"]
                variant = candidate["Variant"]
                tuned_rows = tuned_validation.loc[
                    tuned_validation["Target_Config"].eq(config_name)
                    & tuned_validation["Architecture"].eq(architecture)
                ]
                tuned = tuned_rows.iloc[0]
                base_params = {
                    "fusion_dim": int(tuned["Fusion_Dim"]), "dropout": float(tuned["Dropout"]),
                    "weight_decay": float(tuned["Weight_Decay"]), "learning_rate": float(tuned["Learning_Rate"]),
                    "head_scale": 1.0,
                }
                study_dir = DL_HPARAM_DIR / config_name / architecture / variant
                family_keys = set(DL_SEARCH_CONFIG.get("fusion_hparam_spaces_v1", {}).get(architecture, []))
                if architecture == "PyTorch_Concat" and "bilinear" not in variant:
                    family_keys.discard("concat_bilinear_rank")
                if architecture == "PyTorch_CrossAttn" and "token_attention" not in variant:
                    family_keys.discard("cross_attention_dropout")
                    family_keys.discard("cross_attention_heads")
                if architecture == "PyTorch_Gating" and not variant.startswith("gating_softmax"):
                    family_keys.discard("gating_temperature")
                candidate_space = {
                    key: value for key, value in hparam_space.items()
                    if key not in {k for keys in DL_SEARCH_CONFIG.get("fusion_hparam_spaces_v1", {}).values() for k in keys}
                    or key in family_keys
                }
                study = optuna_trial_plan(
                    study_dir, f"dl_hparam_{config_name}_{architecture}_{variant}",
                    hparam_n_trials, candidate_space, int(CONFIG["seed"]),
                )
                for trial_index, (opt_study, opt_trial, overrides, is_new_trial) in enumerate(study, 1):
                    profile_name = f"optuna_{opt_trial.number:04d}"
                    params = {**base_params, "dropout": float(overrides["dropout"]),
                              "weight_decay": float(overrides["weight_decay"]),
                              "learning_rate": float(overrides["learning_rate"])}
                    params["fusion_dim"] = max(128, int(round(base_params["fusion_dim"] * float(overrides["capacity_scale"]) / 64.0) * 64))
                    params["head_scale"] = float(overrides["head_scale"])
                    params["dna_embedding_dim"] = int(overrides.get("dna_embedding_dim", params["fusion_dim"]))
                    params["prot_embedding_dim"] = int(overrides.get("prot_embedding_dim", params["fusion_dim"]))
                    params["bio_embedding_dim"] = int(overrides.get("bio_embedding_dim", 128))
                    augmentation = {
                        "feature_dropout": float(overrides.get("feature_dropout", 0.0)),
                        "modality_dropout": float(overrides.get("modality_dropout", 0.0)),
                        "gaussian_noise": float(overrides.get("gaussian_noise", 0.0)),
                        **{key: float(overrides.get(key, 0.0)) for key in (
                            "dna_feature_dropout", "protein_feature_dropout", "bio_feature_dropout",
                            "geom_feature_dropout", "bio_gaussian_noise", "geom_gaussian_noise")},
                    }
                    batch_size = int(overrides.get("batch_size", CONFIG["batch_size"]))
                    max_epochs = int(overrides.get("max_epochs", CONFIG["epochs"]))
                    patience = int(overrides.get("early_stop_patience", CONFIG["early_stop_patience"]))
                    grad_clip_norm = float(overrides.get("grad_clip_norm", 1.0))
                    params.update({key: overrides[key] for key in family_keys if key in overrides})
                    params.update({"batch_size": batch_size, "max_epochs": max_epochs,
                                   "early_stop_patience": patience, "optimizer": overrides.get("optimizer", "adamw"),
                                   "grad_clip_norm": grad_clip_norm, "label_smoothing": float(overrides.get("label_smoothing", 0.0)),
                                   "pos_weight_multiplier": float(overrides.get("pos_weight_multiplier", 1.0)),
                                   "focal_gamma": float(overrides.get("focal_gamma", 2.0)),
                                   "focal_alpha": float(overrides.get("focal_alpha", 1.0 - positive_fraction))})
                    loss_type = str(overrides.get("loss_type", "bce"))
                    run_key = (config_name, architecture, variant, profile_name)
                    run_dir = DL_HPARAM_DIR / config_name / architecture / variant / profile_name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint = run_dir / "current_best.pth"
                    best_path = run_dir / "best_model.pth"
                    cached_row = next((row for row in hparam_trial_rows
                                       if row.get("Target_Config") == config_name
                                       and row.get("Architecture") == architecture
                                       and row.get("Variant") == variant
                                       and row.get("Profile") == profile_name), None)
                    if cached_row is not None and best_path.exists():
                        hparam_checkpoints[run_key] = str(best_path)
                        print(f"[RESUME] {config_name}/{architecture}/{variant}/{profile_name}")
                        tell_trial(opt_study, opt_trial, float(cached_row["Val_MCC"])) if is_new_trial else None
                        continue
                    set_variant_seed(int(CONFIG["seed"]))
                    model = FusionArchitectureVariant(
                        dna_dim=dims["dna"], prot_dim=dims["prot"], bio_dim=dims["bio"], geom_dim=dims["geom"],
                        active_modalities=target_config["active_modalities"], variant=variant,
                        fusion_dim=params["fusion_dim"], dropout=params["dropout"], head_scale=params["head_scale"],
                        dna_embedding_dim=params["dna_embedding_dim"],
                        prot_embedding_dim=params["prot_embedding_dim"],
                        bio_embedding_dim=params["bio_embedding_dim"],
                        fusion_hparams={key: params[key] for key in family_keys if key in params},
                    ).to(DEVICE)
                    trial_train_loader = with_batch_size(train_loader, batch_size, int(CONFIG["seed"]))
                    optimizer_class = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam,
                                       "radam": torch.optim.RAdam}[str(params["optimizer"])]
                    optimizer = optimizer_class(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="max", factor=CONFIG["lr_factor"], patience=CONFIG["lr_patience"]
                    )
                    best_mcc, best_epoch, no_improve = -float("inf"), 0, 0
                    best_train_loss = np.nan
                    started = time.perf_counter()
                    was_pruned = False
                    for epoch in range(max_epochs):
                        model.train()
                        train_sum, train_count = 0.0, 0
                        lr_now = float(optimizer.param_groups[0]["lr"])
                        for batch in trial_train_loader:
                            optimizer.zero_grad(set_to_none=True)
                            logits = variant_forward(model, batch, augmentation, input_stds)
                            y = batch["label"].to(DEVICE, non_blocking=True)
                            loss = hparam_loss(logits, y, loss_type, pos_weight, positive_fraction, overrides)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                            optimizer.step()
                            n = int(y.shape[0]); train_sum += float(loss.item()) * n; train_count += n
                        train_loss = train_sum / max(train_count, 1)
                        val_loss, y_val_epoch, p_val_epoch = variant_eval(model, val_loader, criterion_default)
                        epoch_threshold, epoch_mcc = pipeline._select_validation_threshold(y_val_epoch, p_val_epoch)
                        pred_epoch = pipeline._threshold_predictions(p_val_epoch, epoch_threshold)
                        epoch_metrics = FusionEvaluatorProfiler.compute_metrics(y_val_epoch, p_val_epoch, pred_epoch)
                        opt_trial.report(float(epoch_mcc), step=epoch + 1)
                        if opt_trial.should_prune():
                            prune_trial(opt_study, opt_trial)
                            was_pruned = True
                            print(f"[PRUNED] {config_name}/{architecture}/{variant}/{profile_name} at epoch {epoch + 1}")
                            break
                        scheduler.step(float(epoch_mcc))
                        hparam_history_rows.append({
                            "Target_Config": config_name, "Architecture": architecture, "Variant": variant,
                            "Profile": profile_name, "Epoch": epoch + 1, "Learning_Rate": lr_now,
                            "Train_Loss": train_loss, "Val_Loss": val_loss, "Val_MCC": float(epoch_mcc),
                            "Val_AUROC": float(epoch_metrics["AUROC"]), "Val_AUPRC": float(epoch_metrics["AUPRC"]),
                        })
                        if float(epoch_mcc) > best_mcc:
                            best_mcc, best_epoch, best_train_loss = float(epoch_mcc), epoch + 1, train_loss
                            torch.save(model.state_dict(), checkpoint)
                            no_improve = 0
                        else:
                            no_improve += 1
                        if no_improve >= patience:
                            break
                    if was_pruned:
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
                    train_eval_loss, _, _ = variant_eval(model, trial_train_loader, criterion_default)
                    val_eval_loss, y_val, p_val = variant_eval(model, val_loader, criterion_default)
                    threshold, val_mcc = pipeline._select_validation_threshold(y_val, p_val)
                    val_predictions = pipeline._threshold_predictions(p_val, threshold)
                    val_metrics = FusionEvaluatorProfiler.compute_metrics(y_val, p_val, val_predictions)
                    torch.save(model.state_dict(), best_path)
                    hparam_checkpoints[run_key] = str(best_path)
                    hparam_trial_rows = [row for row in hparam_trial_rows if not (
                        row.get("Target_Config") == config_name
                        and row.get("Architecture") == architecture
                        and row.get("Variant") == variant
                        and row.get("Profile") == profile_name
                    )]
                    hparam_trial_rows.append({
                        "Target_Config": config_name, "Ablation": target_config["ablation"],
                        "Architecture": architecture, "Variant": variant, "Profile": profile_name,
                        **params, **augmentation, "Loss_Type": loss_type, "Best_Epoch": best_epoch,
                        "Train_Loss_at_Best": best_train_loss, "Train_Loss_Eval_Mode": train_eval_loss,
                        "Val_Loss_Eval_Mode": val_eval_loss, "Val_MCC": float(val_mcc),
                        "Decision_Threshold": float(threshold),
                        **{f"Val_{k}": v for k, v in val_metrics.items()},
                        "Fit_Seconds": round(time.perf_counter() - started, 3),
                        "Checkpoint_Path": str(best_path),
                    })
                    save_progress_csv(hparam_trial_rows, DL_HPARAM_OUTPUTS[0])
                    save_progress_csv(hparam_history_rows, DL_HPARAM_OUTPUTS[1])
                    tell_trial(opt_study, opt_trial, val_mcc) if is_new_trial else None
                    print(f"[Optuna {config_name}/{architecture}/{variant}/{profile_name}] MCC_val={val_mcc:.4f}; epoch={best_epoch}")
                    del model, trial_train_loader
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            del train_loader, val_loader, dataset
    
        dl_hparam_trials_df = pd.DataFrame(hparam_trial_rows)
        dl_hparam_history_df = pd.DataFrame(hparam_history_rows)
        dl_hparam_trials_df = dl_hparam_trials_df.sort_values(
            ["Target_Config", "Architecture", "Val_MCC"], ascending=[True, True, False]
        )
        dl_hparam_best_validation_df = (
            dl_hparam_trials_df.groupby(["Target_Config", "Architecture"], as_index=False, group_keys=False)
            .head(1).reset_index(drop=True)
        )
        hparam_best_models = {}
        for _, row in dl_hparam_best_validation_df.iterrows():
            target_config = next(c for c in TARGET_CONFIGS if c["name"] == row["Target_Config"])
            config_name, architecture, variant = row["Target_Config"], row["Architecture"], row["Variant"]
            train_loader = pipeline._get_dataloader(
                "train3", "center", DNA_MODEL["name"], PROTEIN_MODEL["name"],
                target_config["active_modalities"], True, target_config["geom_feature_names"], "train3", CONFIG["seed"],
            )
            dataset = train_loader.dataset
            dims = {
                "dna": int(dataset.v_dna_tensor.shape[1]), "prot": int(dataset.v_prot_tensor.shape[1]),
                "bio": int(dataset.bio_tensor.shape[1]) if hasattr(dataset, "bio_tensor") else 0,
                "geom": int(dataset.geom_tensor.shape[1]) if hasattr(dataset, "geom_tensor") else 0,
            }
            model = FusionArchitectureVariant(
                dna_dim=dims["dna"], prot_dim=dims["prot"], bio_dim=dims["bio"], geom_dim=dims["geom"],
                active_modalities=target_config["active_modalities"], variant=variant,
                fusion_dim=int(row["fusion_dim"]), dropout=float(row["dropout"]), head_scale=float(row["head_scale"]),
                dna_embedding_dim=(
                    int(row["dna_embedding_dim"])
                    if pd.notna(row.get("dna_embedding_dim", np.nan))
                    else int(row["fusion_dim"])
                ),
                prot_embedding_dim=(
                    int(row["prot_embedding_dim"])
                    if pd.notna(row.get("prot_embedding_dim", np.nan))
                    else int(row["fusion_dim"])
                ),
                bio_embedding_dim=(
                    int(row["bio_embedding_dim"])
                    if pd.notna(row.get("bio_embedding_dim", np.nan))
                    else 128
                ),
                fusion_hparams={key: row[key] for key in (
                    "concat_bilinear_rank", "cross_attention_dropout", "cross_attention_heads",
                    "transformer_ffn_expansion", "transformer_num_heads", "transformer_depth", "gating_temperature"
                ) if key in row.index and pd.notna(row[key])},
            ).to(DEVICE)
            model.load_state_dict(torch.load(row["Checkpoint_Path"], map_location=DEVICE, weights_only=True))
            model.eval()
            hparam_best_models[(config_name, architecture)] = {
                "model": model, "variant": variant, "threshold": float(row["Decision_Threshold"]),
            }
            del train_loader, dataset
    
        dl_hparam_best_test_rows = []
        for target_config in TARGET_CONFIGS:
            config_name = target_config["name"]
            models = [(key, val) for key, val in hparam_best_models.items() if key[0] == config_name]
            for dataset_config in datasets:
                test_loader = pipeline._get_dataloader(
                    dataset_config["test"], "center", DNA_MODEL["name"], PROTEIN_MODEL["name"],
                    target_config["active_modalities"], False, target_config["geom_feature_names"], "train3", CONFIG["seed"],
                )
                y_true, probabilities = [], {key[1]: [] for key, _ in models}
                with torch.no_grad():
                    for batch in test_loader:
                        y_true.extend(batch["label"].reshape(-1).cpu().numpy())
                        for key, winner in models:
                            probs = torch.sigmoid(variant_forward(winner["model"], batch)).reshape(-1).cpu().numpy()
                            probabilities[key[1]].extend(probs)
                y_true = np.asarray(y_true, dtype=np.int32)
                for key, winner in models:
                    probs = np.asarray(probabilities[key[1]], dtype=np.float32)
                    predictions = pipeline._threshold_predictions(probs, winner["threshold"])
                    metrics = FusionEvaluatorProfiler.compute_metrics(y_true, probs, predictions)
                    dl_hparam_best_test_rows.append({
                        "Target_Config": config_name, "Dataset": dataset_config["name"],
                        "Architecture": key[1], "Variant": winner["variant"],
                        "Profile": next(
                            str(row["Profile"]) for _, row in dl_hparam_best_validation_df.iterrows()
                            if row["Target_Config"] == config_name and row["Architecture"] == key[1]
                        ),
                        "Decision_Threshold": winner["threshold"], **metrics,
                    })
                del test_loader
            for _, winner in models:
                winner["model"].to("cpu")
        dl_hparam_best_test_df = pd.DataFrame(dl_hparam_best_test_rows)
        dl_hparam_trials_df.to_csv(DL_HPARAM_OUTPUTS[0], index=False)
        dl_hparam_history_df.to_csv(DL_HPARAM_OUTPUTS[1], index=False)
        dl_hparam_best_validation_df.to_csv(DL_HPARAM_OUTPUTS[2], index=False)
        dl_hparam_best_test_df.to_csv(DL_HPARAM_OUTPUTS[3], index=False)
        print(f"\n[COMPLETE] Optuna training trials per candidate: {hparam_n_trials}; total trial rows: {len(dl_hparam_trials_df)}")
        display(dl_hparam_best_validation_df.drop(columns=["Checkpoint_Path"], errors="ignore"))
        display(dl_hparam_best_test_df)
    
        dl_arch_trials_df = pd.DataFrame(variant_trial_rows)
        dl_arch_validation_df = pd.DataFrame(variant_validation_rows)
        dl_arch_test_df = pd.DataFrame(variant_test_rows)
        dl_arch_history_df = pd.DataFrame(variant_history_rows)
        dl_arch_trials_df.to_csv(DL_ARCH_VARIANT_OUTPUTS[0], index=False)
        dl_arch_validation_df.to_csv(DL_ARCH_VARIANT_OUTPUTS[1], index=False)
        dl_arch_test_df.to_csv(DL_ARCH_VARIANT_OUTPUTS[2], index=False)
        dl_arch_history_df.to_csv(DL_ARCH_VARIANT_OUTPUTS[3], index=False)
    
        dl_arch_validation_comparison_df, dl_arch_test_comparison_df = build_architecture_variant_comparisons(
            dl_arch_validation_df, dl_arch_test_df
        )
        dl_arch_validation_comparison_df.to_csv(DL_ARCH_VARIANT_OUTPUTS[4], index=False)
        dl_arch_test_comparison_df.to_csv(DL_ARCH_VARIANT_OUTPUTS[5], index=False)
        print(f"\\n[COMPLETE] Architecture variants saved in {DL_ARCH_VARIANT_DIR}")
        print("Variants:", len(dl_arch_trials_df), "| expected 100")
        display(dl_arch_validation_comparison_df)
        display(dl_arch_test_comparison_df)
    
    # ==============================================================================
