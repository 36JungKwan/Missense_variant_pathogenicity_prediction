from .resume_utils import ensure_stage_signature, reject_legacy_stage_artifacts, save_progress_csv, upsert_record
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
    # DL STAGE 3/7: Architecture variants v2
    # ==============================================================================
    import json
    import random
    import time
    from pathlib import Path
    
    import numpy as np
    import torch
    import torch.nn as nn
    
    from core.module05v2_fusion_classifier.evaluator_profiler import FusionEvaluatorProfiler
    
    DL_ARCH_VARIANT_DIR = Path(pipeline.batch_run_dir) / "pure_dl_architecture_variants_v2"
    DL_ARCH_VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    DL_ARCH_VARIANT_OUTPUTS = [
        DL_ARCH_VARIANT_DIR / "architecture_variant_trials.csv",
        DL_ARCH_VARIANT_DIR / "architecture_variant_validation_metrics.csv",
        DL_ARCH_VARIANT_DIR / "architecture_variant_test_metrics.csv",
        DL_ARCH_VARIANT_DIR / "architecture_variant_epoch_history.csv",
        DL_ARCH_VARIANT_DIR / "architecture_variant_validation_comparison.csv",
        DL_ARCH_VARIANT_DIR / "architecture_variant_test_comparison.csv",
    ]
    DL_ARCH_VARIANTS = DL_SEARCH_CONFIG['architecture_variants_v2']
    
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
    
    
    reject_legacy_stage_artifacts(DL_ARCH_VARIANT_DIR, "architecture_variants_v2")
    ensure_stage_signature(DL_ARCH_VARIANT_DIR, "architecture_variants_v2", {
        "variants": DL_ARCH_VARIANTS, "config": CONFIG, "targets": TARGET_CONFIGS, "datasets": datasets,
        "dna_model": DNA_MODEL, "protein_model": PROTEIN_MODEL,
        "checkpoint_selection": "validation_mcc_optimized_threshold_v2",
    })
    if all(path.exists() for path in DL_ARCH_VARIANT_OUTPUTS[:4]):
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
    else:
        class FusionArchitectureVariant(nn.Module):
            """Small structural alternatives sharing the same modality projections/head."""
            def __init__(self, dna_dim, prot_dim, bio_dim, geom_dim, active_modalities,
                         variant, fusion_dim, dropout):
                super().__init__()
                self.variant = variant
                self.active_mods = [str(x).lower() for x in active_modalities]
                self.has_bio = bool({"bio", "bio_core", "spliceai"} & set(self.active_mods))
                self.has_geom = "geom" in self.active_mods
                self.dna_proj = nn.Sequential(
                    nn.Linear(dna_dim, fusion_dim), nn.LayerNorm(fusion_dim),
                    nn.GELU(), nn.Dropout(dropout),
                )
                self.prot_proj = nn.Sequential(
                    nn.Linear(prot_dim, fusion_dim), nn.LayerNorm(fusion_dim),
                    nn.GELU(), nn.Dropout(dropout),
                )
                bio_geom_dim = (bio_dim if self.has_bio else 0) + (geom_dim if self.has_geom else 0)
                self.has_ml_features = bio_geom_dim > 0
                self.bio_as_token = variant == "transformer_bio_token"
                if self.has_ml_features and self.bio_as_token:
                    self.bio_token_proj = nn.Sequential(
                        nn.Linear(bio_geom_dim, fusion_dim), nn.LayerNorm(fusion_dim),
                        nn.GELU(), nn.Dropout(dropout),
                    )
                elif self.has_ml_features:
                    self.bio_geom_upsample = nn.Sequential(
                        nn.Linear(bio_geom_dim, 128), nn.LayerNorm(128),
                        nn.GELU(), nn.Dropout(dropout),
                    )
    
                if variant.startswith("concat_"):
                    extra = int("product" in variant) + int("absdiff" in variant)
                    seq_dim = fusion_dim * (2 + extra)
                elif variant == "cross_gated":
                    self.gate_dna = nn.Linear(fusion_dim, fusion_dim)
                    self.gate_prot = nn.Linear(fusion_dim, fusion_dim)
                    self.cross_dna_ln = nn.LayerNorm(fusion_dim)
                    self.cross_prot_ln = nn.LayerNorm(fusion_dim)
                    seq_dim = fusion_dim * 2
                elif variant == "cross_film":
                    self.film_from_prot = nn.Linear(fusion_dim, fusion_dim * 2)
                    self.film_from_dna = nn.Linear(fusion_dim, fusion_dim * 2)
                    seq_dim = fusion_dim * 2
                elif variant == "cross_token_attention":
                    if fusion_dim % 8:
                        raise ValueError("fusion_dim must be divisible by 8 for token attention")
                    self.token_dim = fusion_dim // 8
                    self.attn_prot_to_dna = nn.MultiheadAttention(
                        self.token_dim, num_heads=8, batch_first=True, dropout=dropout
                    )
                    self.attn_dna_to_prot = nn.MultiheadAttention(
                        self.token_dim, num_heads=8, batch_first=True, dropout=dropout
                    )
                    self.token_dna_ln = nn.LayerNorm(self.token_dim)
                    self.token_prot_ln = nn.LayerNorm(self.token_dim)
                    self.token_out_proj = nn.Sequential(
                        nn.Linear(self.token_dim * 2, fusion_dim * 2),
                        nn.LayerNorm(fusion_dim * 2), nn.GELU(), nn.Dropout(dropout),
                    )
                    seq_dim = fusion_dim * 2
                elif variant in {"transformer_2layer", "transformer_3layer", "transformer_bio_token"}:
                    self.modality_embed = nn.Parameter(torch.randn(1, 3 if self.bio_as_token else 2, fusion_dim) * 0.02)
                    layer = nn.TransformerEncoderLayer(
                        d_model=fusion_dim, nhead=8, batch_first=True, dropout=dropout
                    )
                    depth = 3 if variant == "transformer_3layer" else 2
                    self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
                    seq_dim = fusion_dim * (3 if self.bio_as_token else 2)
                elif variant in {"gating_with_skip", "gating_scalar_skip"}:
                    gate_dim = 1 if variant == "gating_scalar_skip" else fusion_dim
                    self.gate = nn.Linear(fusion_dim * 2, gate_dim)
                    seq_dim = fusion_dim * 3
                elif variant == "gating_interaction":
                    self.gate = nn.Linear(fusion_dim * 2, fusion_dim)
                    seq_dim = fusion_dim * 3
                else:
                    raise ValueError(f"Unknown architecture variant: {variant}")
    
                self.post_fusion_ln = nn.LayerNorm(seq_dim)
                global_dim = seq_dim + (0 if self.bio_as_token else (128 if self.has_ml_features else 0))
                self.classifier_hidden = nn.Sequential(
                    nn.Linear(global_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
                )
                self.shortcut = nn.Linear(global_dim, 128)
                self.output_head = nn.Linear(128, 1)
    
            def _bio_geom_input(self, bio_features, geom_features):
                values = []
                if self.has_bio:
                    values.append(bio_features)
                if self.has_geom:
                    values.append(geom_features)
                return torch.cat(values, dim=-1)
    
            def forward(self, v_dna=None, v_prot=None, bio_features=None, geom_features=None):
                h_dna = self.dna_proj(v_dna)
                h_prot = self.prot_proj(v_prot)
                variant = self.variant
                if variant.startswith("concat_"):
                    parts = [h_dna, h_prot]
                    if "product" in variant:
                        parts.append(h_dna * h_prot)
                    if "absdiff" in variant:
                        parts.append(torch.abs(h_dna - h_prot))
                    seq = torch.cat(parts, dim=-1)
                elif variant == "cross_gated":
                    gate_dna = torch.sigmoid(self.gate_dna(h_prot))
                    gate_prot = torch.sigmoid(self.gate_prot(h_dna))
                    seq = torch.cat([
                        self.cross_dna_ln(h_dna + gate_dna * h_prot),
                        self.cross_prot_ln(h_prot + gate_prot * h_dna),
                    ], dim=-1)
                elif variant == "cross_film":
                    gamma_dna, beta_dna = self.film_from_prot(h_prot).chunk(2, dim=-1)
                    gamma_prot, beta_prot = self.film_from_dna(h_dna).chunk(2, dim=-1)
                    conditioned_dna = h_dna * (1.0 + torch.tanh(gamma_dna)) + beta_dna
                    conditioned_prot = h_prot * (1.0 + torch.tanh(gamma_prot)) + beta_prot
                    seq = torch.cat([conditioned_dna, conditioned_prot], dim=-1)
                elif variant == "cross_token_attention":
                    dna_tokens = h_dna.reshape(h_dna.shape[0], 8, self.token_dim)
                    prot_tokens = h_prot.reshape(h_prot.shape[0], 8, self.token_dim)
                    prot_to_dna, _ = self.attn_prot_to_dna(
                        query=prot_tokens, key=dna_tokens, value=dna_tokens, need_weights=False
                    )
                    dna_to_prot, _ = self.attn_dna_to_prot(
                        query=dna_tokens, key=prot_tokens, value=prot_tokens, need_weights=False
                    )
                    pooled = torch.cat([
                        self.token_prot_ln(prot_tokens + prot_to_dna).mean(dim=1),
                        self.token_dna_ln(dna_tokens + dna_to_prot).mean(dim=1),
                    ], dim=-1)
                    seq = self.token_out_proj(pooled)
                elif variant in {"transformer_2layer", "transformer_3layer", "transformer_bio_token"}:
                    tokens = [h_dna, h_prot]
                    if self.bio_as_token:
                        tokens.append(self.bio_token_proj(self._bio_geom_input(bio_features, geom_features)))
                    token_tensor = torch.stack(tokens, dim=1) + self.modality_embed[:, :len(tokens), :]
                    seq = self.transformer(token_tensor).reshape(token_tensor.shape[0], -1)
                elif variant in {"gating_with_skip", "gating_scalar_skip"}:
                    gate = torch.sigmoid(self.gate(torch.cat([h_dna, h_prot], dim=-1)))
                    mixed = gate * h_dna + (1.0 - gate) * h_prot
                    seq = torch.cat([mixed, h_dna, h_prot], dim=-1)
                else:  # gating_interaction
                    gate = torch.sigmoid(self.gate(torch.cat([h_dna, h_prot], dim=-1)))
                    mixed = gate * h_dna + (1.0 - gate) * h_prot
                    seq = torch.cat([mixed, h_dna * h_prot, torch.abs(h_dna - h_prot)], dim=-1)
    
                features = [self.post_fusion_ln(seq)]
                if self.has_ml_features and not self.bio_as_token:
                    features.append(self.bio_geom_upsample(self._bio_geom_input(bio_features, geom_features)))
                global_features = torch.cat(features, dim=-1)
                return self.output_head(
                    self.classifier_hidden(global_features) + self.shortcut(global_features)
                )
    
        def set_variant_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
        def variant_forward(model, batch):
            def move(key):
                value = batch.get(key)
                return value.to(DEVICE, non_blocking=True) if value is not None else None
            return model(move("v_dna"), move("v_prot"), move("bio_features"), move("geom_features"))
    
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
                            "Val_MCC_at_0_5": np.nan, "Decision_Threshold": float(threshold),
                            "Val_MCC_at_Threshold": float(val_mcc),
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
                    best_mcc_optimized = -float("inf")
                    best_mcc_half = float("nan")
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
                        metrics_half = FusionEvaluatorProfiler.compute_metrics(
                            y_val_epoch, p_val_epoch, (p_val_epoch >= 0.5).astype(np.int32)
                        )
                        val_mcc_half = float(metrics_half["MCC"])
                        epoch_threshold, val_mcc_optimized = pipeline._select_validation_threshold(y_val_epoch, p_val_epoch)
                        scheduler.step(val_mcc_optimized)
                        variant_history_rows.append({
                            "Target_Config": config_name, "Ablation": target_config["ablation"],
                            "Architecture": architecture, "Variant": variant,
                            "Epoch": epoch + 1, "Learning_Rate": lr_now,
                            "Train_Loss_Train_Mode": train_mode_loss,
                            "Val_Loss_Eval_Mode": val_loss,
                            "Val_MCC_at_0_5": val_mcc_half,
                            "Val_MCC_at_Optimized_Threshold": val_mcc_optimized,
                            "Val_Threshold": epoch_threshold,
                        })
                        if val_mcc_optimized > best_mcc_optimized:
                            best_mcc_optimized = val_mcc_optimized
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
                        "Val_MCC_at_0_5": best_mcc_half,
                        "Val_MCC_at_Optimized_Threshold": best_mcc_optimized,
                        "Decision_Threshold": float(threshold),
                        "Val_MCC_at_Threshold": float(val_mcc),
                        **{f"Val_{key}": value for key, value in val_metrics.items()},
                        "Fit_Seconds": round(fit_seconds, 3),
                    }
                    variant_trial_rows = upsert_record(variant_trial_rows, row, ("Target_Config", "Architecture", "Variant"))
                    variant_validation_rows = upsert_record(variant_validation_rows, row.copy(), ("Target_Config", "Architecture", "Variant"))
                    save_progress_csv(variant_trial_rows, DL_ARCH_VARIANT_OUTPUTS[0])
                    save_progress_csv(variant_validation_rows, DL_ARCH_VARIANT_OUTPUTS[1])
                    save_progress_csv(variant_history_rows, DL_ARCH_VARIANT_OUTPUTS[3])
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
        print("Variants:", len(dl_arch_trials_df), "| expected 24")
        display(dl_arch_validation_comparison_df)
        display(dl_arch_test_comparison_df)
    
    # ==============================================================================
