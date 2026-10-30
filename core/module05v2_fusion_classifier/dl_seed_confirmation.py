from .resume_utils import ensure_stage_signature, save_progress_csv
from .dl_architecture_model import FusionArchitectureVariant
"""Repeated-seed confirmation for shortlisted pure-DL architectures."""

def run_seed_confirmation(context, search_config, display_fn=None):
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
    # DL STAGE 7/7: Repeated-seed confirmation
    # ==============================================================================
    # Repeated-seed confirmation for the shortlisted DL architectures.
    import json
    import random
    import time
    from pathlib import Path
    
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    
    from core.module05v2_fusion_classifier.evaluator_profiler import FusionEvaluatorProfiler
    
    SEED_CONFIRM_DIR = Path(pipeline.batch_run_dir) / "pure_dl_seed_confirmation_v1"
    SEED_CONFIRM_DIR.mkdir(parents=True, exist_ok=True)
    SEED_CONFIRM_FILES = {
        "trials": SEED_CONFIRM_DIR / "seed_confirmation_trials.csv",
        "history": SEED_CONFIRM_DIR / "seed_confirmation_epoch_history.csv",
        "test": SEED_CONFIRM_DIR / "seed_confirmation_test_metrics.csv",
        "summary": SEED_CONFIRM_DIR / "seed_confirmation_summary.csv",
    }
    
    # Seed 42 reuses the completed grid checkpoint; 20 additional seeds are fitted.
    CONFIRMATION_SEEDS = DL_SEARCH_CONFIG["confirmation_seeds"]
    CONFIRMATION_CANDIDATES = DL_SEARCH_CONFIG['confirmation_candidates']
    
    OPTUNA_DIR = Path(pipeline.batch_run_dir) / "pure_dl_architecture_hparam_search_v1" / "training_hparam_optuna"
    OPTUNA_TRIALS_PATH = OPTUNA_DIR / "training_optuna_trials.csv"
    if not OPTUNA_TRIALS_PATH.exists():
        raise FileNotFoundError(f"Cannot find completed DL Optuna results: {OPTUNA_TRIALS_PATH}")
    grid_trials_df = pd.read_csv(OPTUNA_TRIALS_PATH)
    # Confirm the strongest Optuna parameter sets for each preselected structure.
    confirmed_candidates = []
    for candidate in CONFIRMATION_CANDIDATES:
        rows = grid_trials_df.loc[
            grid_trials_df["Target_Config"].eq(candidate["Target_Config"])
            & grid_trials_df["Architecture"].eq(candidate["Architecture"])
            & grid_trials_df["Variant"].eq(candidate["Variant"])
        ].sort_values("Val_MCC", ascending=False)
        n_profiles = max(1, len(candidate.get("Profiles", [])))
        if rows.empty:
            raise KeyError(f"No Optuna trials for confirmation candidate {candidate['Target_Config']}/{candidate['Variant']}")
        selected = candidate.copy()
        selected["Profiles"] = rows.head(n_profiles)["Profile"].astype(str).tolist()
        confirmed_candidates.append(selected)
    CONFIRMATION_CANDIDATES = confirmed_candidates
    ensure_stage_signature(SEED_CONFIRM_DIR, "seed_confirmation_v1", {
        "seeds": CONFIRMATION_SEEDS, "candidates": CONFIRMATION_CANDIDATES, "config": CONFIG,
        "targets": TARGET_CONFIGS, "datasets": datasets, "dna_model": DNA_MODEL, "protein_model": PROTEIN_MODEL,
    })
    
    
    
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
            for key, value, rate_key in (
                ("dna", v_dna, "dna_feature_dropout"), ("protein", v_prot, "protein_feature_dropout"),
                ("bio", bio, "bio_feature_dropout"), ("geom", geom, "geom_feature_dropout"),
            ):
                rate = float(augmentation.get(rate_key, augmentation.get("feature_dropout", 0.0)))
                if value is not None and rate > 0:
                    value = torch.nn.functional.dropout(value, p=rate, training=True)
                if key == "dna": v_dna = value
                elif key == "protein": v_prot = value
                elif key == "bio": bio = value
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
            for key, value, scale_key in (("bio", bio, "bio_gaussian_noise"), ("geom", geom, "geom_gaussian_noise")):
                scale = float(augmentation.get(scale_key, 0.0))
                if value is not None and scale > 0:
                    value = value + torch.randn_like(value) * value.float().std(unbiased=False).clamp_min(1e-8) * scale
                if key == "bio": bio = value
                else: geom = value
        return model(v_dna, v_prot, bio, geom)
    
    
    def seed_eval(model, loader, criterion=None):
        model.eval()
        loss_sum, sample_count = 0.0, 0
        labels, probabilities = [], []
        with torch.no_grad():
            for batch in loader:
                logits = variant_forward(model, batch)
                y = batch["label"].to(DEVICE, non_blocking=True)
                if criterion is not None:
                    loss_sum += float(criterion(logits, y).item()) * int(y.shape[0])
                    sample_count += int(y.shape[0])
                labels.extend(y.reshape(-1).cpu().numpy())
                probabilities.extend(torch.sigmoid(logits).reshape(-1).cpu().numpy())
        mean_loss = loss_sum / max(sample_count, 1) if criterion is not None else np.nan
        return mean_loss, np.asarray(labels, dtype=np.int32), np.asarray(probabilities, dtype=np.float32)
    
    
    def seed_hparam_loss(logits, labels, loss_type, pos_weight, positive_fraction, row):
        smoothing = float(row.get("label_smoothing", 0.0))
        if smoothing:
            labels = labels * (1.0 - smoothing) + 0.5 * smoothing
        if loss_type == "weighted_bce":
            scaled_weight = pos_weight * float(row.get("pos_weight_multiplier", 1.0))
            return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=scaled_weight)
        if loss_type == "focal":
            per_item = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            probability_correct = torch.exp(-per_item)
            alpha_positive = float(row.get("focal_alpha", 1.0 - positive_fraction))
            alpha = labels * alpha_positive + (1.0 - labels) * (1.0 - alpha_positive)
            return (alpha * (1.0 - probability_correct).pow(float(row.get("focal_gamma", 2.0))) * per_item).mean()
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)
    
    
    def save_seed_outputs(trial_rows, history_rows, test_rows):
        save_progress_csv(trial_rows, SEED_CONFIRM_FILES["trials"])
        save_progress_csv(history_rows, SEED_CONFIRM_FILES["history"])
        save_progress_csv(test_rows, SEED_CONFIRM_FILES["test"])
    
    
    expected_confirmation_keys = {
        (candidate["Target_Config"], candidate["Architecture"], candidate["Variant"], profile, seed)
        for candidate in CONFIRMATION_CANDIDATES
        for profile in candidate["Profiles"]
        for seed in CONFIRMATION_SEEDS
    }
    seed_confirmation_cache_complete = False
    if all(path.exists() for path in SEED_CONFIRM_FILES.values()):
        cached_trials = pd.read_csv(SEED_CONFIRM_FILES["trials"])
        cached_keys = {
            (str(row.Target_Config), str(row.Architecture), str(row.Variant), str(row.Profile), int(row.Seed))
            for row in cached_trials.itertuples(index=False)
        }
        seed_confirmation_cache_complete = expected_confirmation_keys.issubset(cached_keys)
    
    if seed_confirmation_cache_complete:
        print(f"[REUSE] Completed seed confirmation results: {SEED_CONFIRM_DIR}")
        seed_confirmation_trials_df = pd.read_csv(SEED_CONFIRM_FILES["trials"])
        seed_confirmation_history_df = pd.read_csv(SEED_CONFIRM_FILES["history"])
        seed_confirmation_test_df = pd.read_csv(SEED_CONFIRM_FILES["test"])
        seed_confirmation_summary_df = pd.read_csv(SEED_CONFIRM_FILES["summary"])
    else:
        trial_rows = (
            pd.read_csv(SEED_CONFIRM_FILES["trials"]).to_dict("records")
            if SEED_CONFIRM_FILES["trials"].exists() else []
        )
        history_rows = (
            pd.read_csv(SEED_CONFIRM_FILES["history"]).to_dict("records")
            if SEED_CONFIRM_FILES["history"].exists() else []
        )
        test_rows = (
            pd.read_csv(SEED_CONFIRM_FILES["test"]).to_dict("records")
            if SEED_CONFIRM_FILES["test"].exists() else []
        )
        completed_keys = {
            (str(r["Target_Config"]), str(r["Architecture"]), str(r["Variant"]), str(r["Profile"]), int(r["Seed"]))
            for r in trial_rows
        }
        candidate_by_config = {cfg["name"]: cfg for cfg in TARGET_CONFIGS}
        criterion_bce = nn.BCEWithLogitsLoss()
        total_fits = sum(len(c["Profiles"]) for c in CONFIRMATION_CANDIDATES) * (len(CONFIRMATION_SEEDS) - 1)
        finished_new_fits = sum(1 for r in trial_rows if int(r["Seed"]) != 42)
        print(f"[*] Planned new fits: {total_fits}; completed new fits already saved: {finished_new_fits}")
    
        for candidate in CONFIRMATION_CANDIDATES:
            config_name = candidate["Target_Config"]
            target_config = candidate_by_config[config_name]
            architecture = candidate["Architecture"]
            variant = candidate["Variant"]
            for profile_name in candidate["Profiles"]:
                matching_grid = grid_trials_df.loc[
                    grid_trials_df["Target_Config"].eq(config_name)
                    & grid_trials_df["Architecture"].eq(architecture)
                    & grid_trials_df["Variant"].eq(variant)
                    & grid_trials_df["Profile"].eq(profile_name)
                ]
                if matching_grid.empty:
                    raise KeyError(f"Missing grid row for {config_name}/{variant}/{profile_name}")
                grid_row = matching_grid.iloc[0]
                params = {
                    "fusion_dim": int(grid_row["fusion_dim"]),
                    "dropout": float(grid_row["dropout"]),
                    "weight_decay": float(grid_row["weight_decay"]),
                    "learning_rate": float(grid_row["learning_rate"]),
                    "head_scale": float(grid_row["head_scale"]),
                    "dna_embedding_dim": int(grid_row.get("dna_embedding_dim", grid_row["fusion_dim"])),
                    "prot_embedding_dim": int(grid_row.get("prot_embedding_dim", grid_row["fusion_dim"])),
                    "bio_embedding_dim": int(grid_row.get("bio_embedding_dim", 128)),
                    **{key: grid_row[key] for key in (
                        "batch_size", "max_epochs", "early_stop_patience", "optimizer", "grad_clip_norm",
                        "label_smoothing", "pos_weight_multiplier", "focal_gamma", "focal_alpha",
                        "concat_bilinear_rank", "cross_attention_dropout", "transformer_ffn_expansion", "gating_temperature",
                        "cross_attention_heads", "transformer_num_heads", "transformer_depth",
                    ) if key in grid_row.index and pd.notna(grid_row[key])},
                }
                augmentation = {
                    "feature_dropout": float(grid_row["feature_dropout"]),
                    "modality_dropout": float(grid_row["modality_dropout"]),
                    "gaussian_noise": float(grid_row["gaussian_noise"]),
                    **{key: float(grid_row.get(key, 0.0)) for key in (
                        "dna_feature_dropout", "protein_feature_dropout", "bio_feature_dropout",
                        "geom_feature_dropout", "bio_gaussian_noise", "geom_gaussian_noise")},
                }
                loss_type = str(grid_row["Loss_Type"])
    
                for seed in CONFIRMATION_SEEDS:
                    run_key = (config_name, architecture, variant, profile_name, seed)
                    if run_key in completed_keys:
                        print(f"[SKIP] Already complete: {run_key}")
                        continue
                    set_variant_seed(seed)
                    train_loader = pipeline._get_dataloader(
                        "train3", "center", DNA_MODEL["name"], PROTEIN_MODEL["name"],
                        target_config["active_modalities"], True,
                        target_config["geom_feature_names"], "train3", seed,
                    )
                    val_loader = pipeline._get_dataloader(
                        "val", "center", DNA_MODEL["name"], PROTEIN_MODEL["name"],
                        target_config["active_modalities"], False,
                        target_config["geom_feature_names"], "train3", seed,
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
                    labels_train = dataset.labels.float()
                    positive_count = float(labels_train.sum().item())
                    negative_count = float(labels_train.numel() - positive_count)
                    positive_fraction = positive_count / max(positive_count + negative_count, 1.0)
                    pos_weight = torch.tensor(
                        [negative_count / max(positive_count, 1.0)], device=DEVICE
                    )
                    model = FusionArchitectureVariant(
                        dna_dim=dims["dna"], prot_dim=dims["prot"], bio_dim=dims["bio"], geom_dim=dims["geom"],
                        active_modalities=target_config["active_modalities"], variant=variant,
                        fusion_dim=params["fusion_dim"], dropout=params["dropout"], head_scale=params["head_scale"],
                        dna_embedding_dim=params["dna_embedding_dim"],
                        prot_embedding_dim=params["prot_embedding_dim"],
                        bio_embedding_dim=params["bio_embedding_dim"],
                        fusion_hparams={key: params[key] for key in (
                            "concat_bilinear_rank", "cross_attention_dropout", "cross_attention_heads",
                            "transformer_ffn_expansion", "transformer_num_heads", "transformer_depth", "gating_temperature"
                        ) if key in params},
                    ).to(DEVICE)
                    if int(params.get("batch_size", CONFIG["batch_size"])) != int(train_loader.batch_size):
                        from torch.utils.data import DataLoader
                        train_loader = DataLoader(
                            train_loader.dataset, batch_size=int(params["batch_size"]), shuffle=True,
                            num_workers=train_loader.num_workers, pin_memory=getattr(train_loader, "pin_memory", False),
                            drop_last=False, generator=torch.Generator().manual_seed(int(seed)),
                            collate_fn=train_loader.collate_fn,
                        )
                    checkpoint_dir = (
                        SEED_CONFIRM_DIR / config_name / variant / profile_name / f"seed_{seed}"
                    )
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    best_checkpoint = checkpoint_dir / "best_model.pth"
                    fit_seconds = 0.0
                    epoch_history = []
    
                    if seed == 42:
                        # Reuse the seed-42 fit from the preceding 208-run grid.
                        model.load_state_dict(torch.load(grid_row["Checkpoint_Path"], map_location=DEVICE, weights_only=True))
                        best_epoch = int(grid_row["Best_Epoch"])
                        best_train_loss = float(grid_row["Train_Loss_at_Best"])
                        train_eval_loss = float(grid_row["Train_Loss_Eval_Mode"])
                        val_loss, y_val, p_val = seed_eval(model, val_loader, criterion_bce)
                        threshold = float(grid_row["Decision_Threshold"])
                        val_predictions = pipeline._threshold_predictions(p_val, threshold)
                        val_metrics = FusionEvaluatorProfiler.compute_metrics(y_val, p_val, val_predictions)
                        val_mcc = float(val_metrics["MCC"])
                        fit_seconds = float(grid_row["Fit_Seconds"])
                    else:
                        if getattr(train_loader, "generator", None) is not None:
                            train_loader.generator.manual_seed(seed)
                        optimizer_class = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam,
                                           "radam": torch.optim.RAdam}.get(str(params.get("optimizer", "adamw")), torch.optim.AdamW)
                        optimizer = optimizer_class(
                            model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
                        )
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode="max", factor=CONFIG["lr_factor"], patience=CONFIG["lr_patience"]
                        )
                        best_mcc, best_epoch, no_improve = -float("inf"), 0, 0
                        best_train_loss = np.nan
                        started = time.perf_counter()
                        for epoch in range(int(params.get("max_epochs", CONFIG["epochs"]))):
                            model.train()
                            train_sum, train_count = 0.0, 0
                            learning_rate_now = float(optimizer.param_groups[0]["lr"])
                            for batch in train_loader:
                                optimizer.zero_grad(set_to_none=True)
                                logits = variant_forward(model, batch, augmentation, input_stds)
                                labels = batch["label"].to(DEVICE, non_blocking=True)
                                loss = seed_hparam_loss(
                                    logits, labels, loss_type, pos_weight, positive_fraction, params
                                )
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(params.get("grad_clip_norm", 1.0)))
                                optimizer.step()
                                batch_size = int(labels.shape[0])
                                train_sum += float(loss.item()) * batch_size
                                train_count += batch_size
                            train_loss = train_sum / max(train_count, 1)
                            val_loss_epoch, y_val_epoch, p_val_epoch = seed_eval(
                                model, val_loader, criterion_bce
                            )
                            threshold_epoch, mcc_epoch = pipeline._select_validation_threshold(
                                y_val_epoch, p_val_epoch
                            )
                            pred_epoch = pipeline._threshold_predictions(p_val_epoch, threshold_epoch)
                            epoch_metrics = FusionEvaluatorProfiler.compute_metrics(
                                y_val_epoch, p_val_epoch, pred_epoch
                            )
                            scheduler.step(float(mcc_epoch))
                            epoch_history.append({
                                "Target_Config": config_name, "Architecture": architecture,
                                "Variant": variant, "Profile": profile_name, "Seed": seed,
                                "Epoch": epoch + 1, "Learning_Rate": learning_rate_now,
                                "Train_Loss": train_loss, "Val_Loss": val_loss_epoch,
                                "Val_MCC": float(mcc_epoch), "Val_AUROC": float(epoch_metrics["AUROC"]),
                                "Val_AUPRC": float(epoch_metrics["AUPRC"]),
                            })
                            if float(mcc_epoch) > best_mcc:
                                best_mcc = float(mcc_epoch)
                                best_epoch = epoch + 1
                                best_train_loss = train_loss
                                torch.save(model.state_dict(), best_checkpoint)
                                no_improve = 0
                            else:
                                no_improve += 1
                            if no_improve >= int(params.get("early_stop_patience", CONFIG["early_stop_patience"])):
                                break
                        model.load_state_dict(torch.load(best_checkpoint, map_location=DEVICE, weights_only=True))
                        train_eval_loss, _, _ = seed_eval(model, train_loader, criterion_bce)
                        val_loss, y_val, p_val = seed_eval(model, val_loader, criterion_bce)
                        threshold, val_mcc = pipeline._select_validation_threshold(y_val, p_val)
                        val_predictions = pipeline._threshold_predictions(p_val, threshold)
                        val_metrics = FusionEvaluatorProfiler.compute_metrics(y_val, p_val, val_predictions)
                        fit_seconds = time.perf_counter() - started
    
                    trial_rows.append({
                        "Target_Config": config_name, "Ablation": target_config["ablation"],
                        "Architecture": architecture, "Variant": variant, "Profile": profile_name,
                        "Seed": seed, **params, **augmentation, "Loss_Type": loss_type,
                        "Best_Epoch": best_epoch, "Train_Loss_at_Best": best_train_loss,
                        "Train_Loss_Eval_Mode": train_eval_loss, "Val_Loss_Eval_Mode": val_loss,
                        "Val_MCC": float(val_mcc), "Decision_Threshold": float(threshold),
                        **{f"Val_{k}": v for k, v in val_metrics.items()},
                        "Fit_Seconds": round(float(fit_seconds), 3),
                        "Checkpoint_Path": str(grid_row["Checkpoint_Path"] if seed == 42 else best_checkpoint),
                    })
                    history_rows.extend(epoch_history)
    
                    # Evaluate every seed with its own validation-selected threshold; tests do not select a profile.
                    model.eval()
                    for dataset_config in datasets:
                        test_loader = pipeline._get_dataloader(
                            dataset_config["test"], "center", DNA_MODEL["name"], PROTEIN_MODEL["name"],
                            target_config["active_modalities"], False,
                            target_config["geom_feature_names"], "train3", seed,
                        )
                        _, y_test, p_test = seed_eval(model, test_loader)
                        test_predictions = pipeline._threshold_predictions(p_test, threshold)
                        test_metrics = FusionEvaluatorProfiler.compute_metrics(
                            y_test, p_test, test_predictions
                        )
                        test_rows.append({
                            "Target_Config": config_name, "Dataset": dataset_config["name"],
                            "Architecture": architecture, "Variant": variant, "Profile": profile_name,
                            "Seed": seed, "Decision_Threshold": float(threshold), **test_metrics,
                        })
                        del test_loader
                    del model, train_loader, val_loader, dataset
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    completed_keys.add(run_key)
                    save_seed_outputs(trial_rows, history_rows, test_rows)
                    print(
                        f"[{config_name}/{profile_name}/seed={seed}] "
                        f"Val MCC={val_mcc:.4f}, epoch={best_epoch}, test rows={len(datasets)}"
                    )
    
        seed_confirmation_trials_df = pd.DataFrame(trial_rows)
        seed_confirmation_history_df = pd.DataFrame(history_rows)
        seed_confirmation_test_df = pd.DataFrame(test_rows)
        metric_columns = ["Val_MCC", "Val_AUROC", "Val_AUPRC", "Train3Val_Test_MCC",
                          "Train3Val_Test_AUROC", "Train3Val_Test_AUPRC",
                          "Train3Val_ClinVarHQ_MCC", "Train3Val_ClinVarHQ_AUROC", "Train3Val_ClinVarHQ_AUPRC",
                          "Train3Val_ProteinGym_MCC", "Train3Val_ProteinGym_AUROC", "Train3Val_ProteinGym_AUPRC",
                          "Train3Val_UniProt_MCC", "Train3Val_UniProt_AUROC", "Train3Val_UniProt_AUPRC"]
        test_wide = seed_confirmation_test_df.pivot_table(
            index=["Target_Config", "Architecture", "Variant", "Profile", "Seed"],
            columns="Dataset", values=["MCC", "AUROC", "AUPRC"],
        )
        test_wide.columns = [f"{dataset}_{metric}" for metric, dataset in test_wide.columns]
        test_wide = test_wide.reset_index()
        summary_input = seed_confirmation_trials_df.merge(
            test_wide, on=["Target_Config", "Architecture", "Variant", "Profile", "Seed"], how="left"
        )
        summary_metrics = [col for col in metric_columns if col in summary_input.columns]
        summary = summary_input.groupby(
            ["Target_Config", "Architecture", "Variant", "Profile"], as_index=False
        ).agg(
            N_Seeds=("Seed", "nunique"),
            Best_Epoch_Mean=("Best_Epoch", "mean"),
            Best_Epoch_SD=("Best_Epoch", "std"),
            Decision_Threshold_Mean=("Decision_Threshold", "mean"),
            Decision_Threshold_SD=("Decision_Threshold", "std"),
            **{f"{metric}_Mean": (metric, "mean") for metric in summary_metrics},
            **{f"{metric}_SD": (metric, "std") for metric in summary_metrics},
        )
        seed_confirmation_summary_df = summary.sort_values(
            ["Target_Config", "Architecture", "Profile"]
        ).reset_index(drop=True)
        save_seed_outputs(trial_rows, history_rows, test_rows)
        seed_confirmation_summary_df.to_csv(SEED_CONFIRM_FILES["summary"], index=False)
    
    print(f"\n[COMPLETE] Seed-confirmation outputs: {SEED_CONFIRM_DIR}")
    print("Seed rows:", len(seed_confirmation_trials_df), "| expected 126 across 21 seeds")
    display(seed_confirmation_summary_df)
    display(seed_confirmation_trials_df[[
        "Target_Config", "Variant", "Profile", "Seed", "Best_Epoch", "Val_MCC", "Val_AUROC", "Val_AUPRC",
        "Decision_Threshold",
    ]].sort_values(["Target_Config", "Variant", "Profile", "Seed"]))
    
    # ==============================================================================


