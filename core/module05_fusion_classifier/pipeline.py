import os
import gc
import random
import importlib
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from core.module05_fusion_classifier.dataset import VariantFusionDataset
from core.module05_fusion_classifier.fusion_model import MultiStrategyFusionModel
from core.module05_fusion_classifier.xgboost_model import XGBoostFusionManager
from core.module05_fusion_classifier.evaluator_profiler import FusionEvaluatorProfiler


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _resolve_bio_path(processed_dir: str, split_name: str) -> str:
    candidates = [
        f"{processed_dir}/{split_name}_normalized.parquet",
        f"{processed_dir}/{split_name}_full_seq_final_normalized.parquet",
        f"{processed_dir}/{split_name}_full_seq_after_vep_final_normalized.parquet",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    if os.path.exists(processed_dir):
        for name in sorted(os.listdir(processed_dir)):
            if name.startswith(split_name) and name.endswith("_normalized.parquet"):
                return f"{processed_dir}/{name}"

    raise FileNotFoundError(
        f"Khong tim thay bio parquet normalized cho split={split_name} trong {processed_dir}"
    )


def _resolve_geom_path(geometry_dir: str, split_name: str, model_name: str, pooling: str) -> str:
    candidates = [
        f"{geometry_dir}/{split_name}/{model_name}_{pooling}_geom_norm.parquet",
        f"{geometry_dir}/{split_name}/{model_name}_{pooling}_geom.parquet",
        # Backward-compatible fallback for older geometry naming.
        f"{geometry_dir}/{split_name}/{model_name}_geom_norm.parquet",
        f"{geometry_dir}/{split_name}/{model_name}_geom.parquet",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Khong tim thay geometry parquet cho split={split_name}, model={model_name}, pooling={pooling}"
    )


def _resolve_pt_path(embed_dir: str, split_name: str, model_name: str, pooling: str) -> str:
    p = f"{embed_dir}/{split_name}/{model_name}_{pooling}.pt"
    if not os.path.exists(p):
        raise FileNotFoundError(f"Khong tim thay embedding pt: {p}")
    return p


def _safe_tensor(batch, key, device):
    return batch[key].to(device) if key in batch else None


def _safe_slice_dummy(batch, key, device):
    t = _safe_tensor(batch, key, device)
    return t[:1] if t is not None else None


def _extract_features_for_ml(model, dataloader, device, extract_f_global=True):
    model.eval()
    all_f_global, all_v_dna, all_v_prot, all_bg, all_labels, all_vids = [], [], [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            v_dna = _safe_tensor(batch, "v_dna", device)
            v_prot = _safe_tensor(batch, "v_prot", device)

            bg_list = []
            if "bio_features" in batch:
                bg_list.append(batch["bio_features"])
            if "geom_features" in batch:
                bg_list.append(batch["geom_features"])
            bg = torch.cat(bg_list, dim=-1).to(device) if len(bg_list) > 0 else None

            if extract_f_global:
                f_global = model(
                    v_dna,
                    v_prot,
                    _safe_tensor(batch, "bio_features", device),
                    _safe_tensor(batch, "geom_features", device),
                    return_features=True,
                )
                all_f_global.append(f_global.cpu())

            if v_dna is not None:
                all_v_dna.append(v_dna.cpu())
            if v_prot is not None:
                all_v_prot.append(v_prot.cpu())
            if bg is not None:
                all_bg.append(bg.cpu())

            all_labels.append(batch["label"].cpu())
            all_vids.extend(batch["variant_id"])

    return (
        torch.cat(all_f_global).numpy() if extract_f_global and len(all_f_global) > 0 else None,
        torch.cat(all_v_dna).numpy() if len(all_v_dna) > 0 else None,
        torch.cat(all_v_prot).numpy() if len(all_v_prot) > 0 else None,
        torch.cat(all_bg).numpy() if len(all_bg) > 0 else None,
        torch.cat(all_labels).squeeze(-1).numpy(),
        all_vids,
    )


class FusionBatchPipeline:
    def __init__(
        self,
        base_dir: str = "D:/variant_data",
        config: dict | None = None,
        datasets: list | None = None,
        models_space: list | None = None,
        pooling_strategies: list | None = None,
        experiments: list | None = None,
        explainability: dict | None = None,
        fm_profile_json: str = "D:/variant_data/profiling/fm_profiling.json",
        batch_run_dir: str | None = None,
    ):
        self.base_dir = base_dir
        if config is None or datasets is None or models_space is None or pooling_strategies is None or experiments is None:
            raise ValueError(
                "config, datasets, models_space, pooling_strategies, experiments phai duoc khai bao tai notebook"
            )

        self.config = config
        self.datasets = datasets
        self.models_space = models_space
        self.pooling_strategies = pooling_strategies
        self.experiments = experiments
        self.explainability = explainability or {
            "enable_shap": True,
            "enable_lime": False,
            "max_background_samples": 512,
            "max_explain_samples": 200,
            "lime_num_features": 20,
            "max_lime_samples": 20,
            "random_state": 42,
        }
        self.fm_profile_json = fm_profile_json

        self.bio_dir = f"{self.base_dir}/processed_parquet"
        self.geom_dir = f"{self.base_dir}/geometry"
        self.embed_dir = f"{self.base_dir}/fm_embeddings"

        if batch_run_dir:
            self.batch_run_dir = os.path.abspath(batch_run_dir)
        else:
            now = datetime.now().strftime("%Y%m%d_%H%M")
            self.batch_run_dir = f"{os.path.abspath('../')}/experiments/batch_run_{now}"
        os.makedirs(self.batch_run_dir, exist_ok=True)

        self.metrics_csv_path = f"{self.batch_run_dir}/global_leaderboard_metrics.csv"
        self.profiling_csv_path = f"{self.batch_run_dir}/global_leaderboard_profiling.csv"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim_cache = {}

        self.dna_models = [m for m in self.models_space if m["seq_type"] == "dna"]
        self.prot_models = [m for m in self.models_space if m["seq_type"] == "protein"]

    def _build_run_key(self, dataset_name, pooling, dna_name, prot_name, ablation_str, network_name):
        return f"{dataset_name}_{pooling}_{dna_name}_{prot_name}_{ablation_str}_{network_name}"

    def _load_resume_state(self):
        global_metrics = []
        global_profiling = []
        completed_by_pooling = {}

        if os.path.exists(self.metrics_csv_path):
            df_metrics = pd.read_csv(self.metrics_csv_path)
            req_cols = ["Dataset", "Pooling", "Ablation", "DNA_Model", "Prot_Model", "Network"]
            if all(col in df_metrics.columns for col in req_cols):
                for row in df_metrics.to_dict(orient="records"):
                    global_metrics.append(row)
                    pooling = row["Pooling"]
                    key = self._build_run_key(
                        row["Dataset"],
                        row["Pooling"],
                        row["DNA_Model"],
                        row["Prot_Model"],
                        row["Ablation"],
                        row["Network"],
                    )
                    completed_by_pooling.setdefault(pooling, set()).add(key)
            else:
                print(
                    "[RESUME] global_leaderboard_metrics.csv thieu cot khoa; bo qua resume metrics."
                )

        if os.path.exists(self.profiling_csv_path):
            df_profiling = pd.read_csv(self.profiling_csv_path)
            global_profiling = df_profiling.to_dict(orient="records")

        return global_metrics, global_profiling, completed_by_pooling

    def _persist_global_outputs(self, global_metrics, global_profiling):
        df_metrics = pd.DataFrame(global_metrics)
        if not df_metrics.empty and "Dataset" in df_metrics.columns and "MCC" in df_metrics.columns:
            df_metrics = df_metrics.sort_values(by=["Dataset", "MCC"], ascending=[True, False])
        df_metrics.to_csv(self.metrics_csv_path, index=False)

        df_profiling = pd.DataFrame(global_profiling)
        df_profiling.to_csv(self.profiling_csv_path, index=False)

    def _get_train_cache_ckpt_dir(self, dataset_cfg, pooling, ablation_str, dna_name, prot_name, exp_name):
        train_val_tag = f"{dataset_cfg['train']}__{dataset_cfg['val']}"
        return (
            f"{self.batch_run_dir}/_train_cache/{train_val_tag}/"
            f"{pooling}/{ablation_str}/{dna_name}__{prot_name}/{exp_name}/checkpoints"
        )

    def _feature_names(self, n_features: int, prefix: str):
        return [f"{prefix}_{i:04d}" for i in range(n_features)]

    def _save_shap(self, save_dir: str, xgb_model, x_background: np.ndarray, x_explain: np.ndarray, variant_ids: list, feature_names: list):
        if not self.explainability.get("enable_shap", False):
            return
        try:
            shap = importlib.import_module("shap")
        except ImportError:
            print("[SHAP] Chua cai shap, bo qua explainability SHAP")
            return

        max_bg = int(self.explainability.get("max_background_samples", 512))
        max_ex = int(self.explainability.get("max_explain_samples", 200))
        rnd = np.random.default_rng(int(self.explainability.get("random_state", 42)))

        bg_idx = np.arange(x_background.shape[0])
        if len(bg_idx) > max_bg:
            bg_idx = rnd.choice(bg_idx, size=max_bg, replace=False)

        ex_idx = np.arange(x_explain.shape[0])
        if len(ex_idx) > max_ex:
            ex_idx = rnd.choice(ex_idx, size=max_ex, replace=False)

        x_bg = x_background[bg_idx]
        x_ex = x_explain[ex_idx]
        ex_vids = [variant_ids[i] for i in ex_idx.tolist()]

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(x_ex)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        abs_mean = np.mean(np.abs(shap_values), axis=0)
        df_global = pd.DataFrame({"feature": feature_names, "mean_abs_shap": abs_mean})
        df_global = df_global.sort_values("mean_abs_shap", ascending=False)
        df_global.to_csv(f"{save_dir}/shap_global_importance.csv", index=False)

        top_k = min(30, shap_values.shape[1])
        top_features = df_global.head(top_k)["feature"].tolist()
        top_indices = [feature_names.index(f) for f in top_features]
        rows = []
        for r, vid in enumerate(ex_vids):
            row = {"Variant_ID": vid}
            for feat, c in zip(top_features, top_indices):
                row[f"shap_{feat}"] = float(shap_values[r, c])
            rows.append(row)
        pd.DataFrame(rows).to_csv(f"{save_dir}/shap_local_top_features.csv", index=False)

    def _save_lime(self, save_dir: str, x_train: np.ndarray, x_explain: np.ndarray, variant_ids: list, feature_names: list, predict_proba_fn):
        if not self.explainability.get("enable_lime", False):
            return
        try:
            LimeTabularExplainer = importlib.import_module("lime.lime_tabular").LimeTabularExplainer
        except ImportError:
            print("[LIME] Chua cai lime, bo qua explainability LIME")
            return

        max_ex = int(self.explainability.get("max_lime_samples", 20))
        n_feat = int(self.explainability.get("lime_num_features", 20))
        max_bg = int(self.explainability.get("max_background_samples", 512))
        rnd = np.random.default_rng(int(self.explainability.get("random_state", 42)))

        bg_idx = np.arange(x_train.shape[0])
        if len(bg_idx) > max_bg:
            bg_idx = rnd.choice(bg_idx, size=max_bg, replace=False)
        x_bg = x_train[bg_idx]

        ex_idx = np.arange(x_explain.shape[0])
        if len(ex_idx) > max_ex:
            ex_idx = rnd.choice(ex_idx, size=max_ex, replace=False)

        explainer = LimeTabularExplainer(
            training_data=x_bg,
            feature_names=feature_names,
            class_names=["Benign", "Pathogenic"],
            mode="classification",
            discretize_continuous=True,
        )

        rows = []
        for i in ex_idx.tolist():
            exp = explainer.explain_instance(x_explain[i], predict_proba_fn, num_features=n_feat)
            for rank, (feat_expr, weight) in enumerate(exp.as_list(), start=1):
                rows.append({
                    "Variant_ID": variant_ids[i],
                    "rank": rank,
                    "feature_rule": feat_expr,
                    "weight": float(weight),
                })
        pd.DataFrame(rows).to_csv(f"{save_dir}/lime_local_explanations.csv", index=False)

    def _run_xgb_explainability(self, exp_dir: str, xgb_model, x_train: np.ndarray, x_test: np.ndarray, variant_ids: list, feature_prefix: str):
        if x_train is None or x_test is None or len(variant_ids) == 0:
            return
        explain_dir = f"{exp_dir}/explainability"
        os.makedirs(explain_dir, exist_ok=True)
        feature_names = self._feature_names(x_test.shape[1], feature_prefix)
        self._save_shap(explain_dir, xgb_model, x_train, x_test, variant_ids, feature_names)
        self._save_lime(explain_dir, x_train, x_test, variant_ids, feature_names, xgb_model.predict_proba)

    def _get_cached_dim(self, model_name, split_name, pooling):
        if model_name == "None":
            return 0
        key = f"{model_name}_{pooling}"
        if key not in self.dim_cache:
            pt_path = _resolve_pt_path(self.embed_dir, split_name, model_name, pooling)
            data = torch.load(pt_path, weights_only=False)
            self.dim_cache[key] = int(data["E_ref"][0].shape[0])
            del data
        return self.dim_cache[key]

    def _get_dataloader(self, split_name, pooling, dna_name, prot_name, active_mods, is_train):
        dataset = VariantFusionDataset(
            bio_parquet_path=_resolve_bio_path(self.bio_dir, split_name),
            dna_geom_path=_resolve_geom_path(self.geom_dir, split_name, dna_name, pooling) if dna_name != "None" else None,
            prot_geom_path=_resolve_geom_path(self.geom_dir, split_name, prot_name, pooling) if prot_name != "None" else None,
            dna_pt_path=_resolve_pt_path(self.embed_dir, split_name, dna_name, pooling) if dna_name != "None" else None,
            prot_pt_path=_resolve_pt_path(self.embed_dir, split_name, prot_name, pooling) if prot_name != "None" else None,
            active_modalities=active_mods,
            is_train=is_train,
        )
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=is_train,
            num_workers=self.config["num_workers"],
            pin_memory=(self.device.type == "cuda"),
        )

    def _run_group(self, dna_cfg, prot_cfg, pooling, active_mods, dataset_cfg, completed_keys, g_metrics, g_prof, experiments_subset=None):
        # Keep model names for geometry file resolution even when DNA/Prot are not active modalities.
        dna_name = dna_cfg["name"] if dna_cfg else "None"
        prot_name = prot_cfg["name"] if prot_cfg else "None"
        ablation_str = "_".join(sorted(active_mods))
        num_seq = int("dna" in active_mods) + int("prot" in active_mods)

        exp_candidates = experiments_subset if experiments_subset is not None else self.experiments

        tasks_to_run = []
        for exp in exp_candidates:
            if num_seq < 2 and exp["fusion"] in ["cross_attention", "transformer", "gating"]:
                continue
            run_key = self._build_run_key(
                dataset_cfg["name"],
                pooling,
                dna_name,
                prot_name,
                ablation_str,
                exp["name"],
            )
            if run_key not in completed_keys:
                tasks_to_run.append((exp, run_key))

        if not tasks_to_run:
            return

        dna_dim = self._get_cached_dim(dna_name, dataset_cfg["train"], pooling) if "dna" in active_mods else 0
        prot_dim = self._get_cached_dim(prot_name, dataset_cfg["train"], pooling) if "prot" in active_mods else 0

        train_loader = self._get_dataloader(dataset_cfg["train"], pooling, dna_name, prot_name, active_mods, True)
        val_loader = self._get_dataloader(dataset_cfg["val"], pooling, dna_name, prot_name, active_mods, False)
        test_loader = self._get_dataloader(dataset_cfg["test"], pooling, dna_name, prot_name, active_mods, False)

        for exp, run_key in tasks_to_run:
            completed_keys.add(run_key)
            config_tag = f"{dataset_cfg['train']}__{dataset_cfg['val']}"
            exp_name = f"{config_tag}_{pooling}_{ablation_str}_{dna_name}_{prot_name}_{exp['name']}"
            print(f"[*] Running config={exp_name} | test={dataset_cfg['name']}")

            exp_dir = f"{self.batch_run_dir}/{exp_name}"
            test_out_dir = f"{exp_dir}/tests/{dataset_cfg['name']}"
            os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
            os.makedirs(test_out_dir, exist_ok=True)
            best_model_path = f"{exp_dir}/checkpoints/best_model.pth"
            train_cache_ckpt_dir = self._get_train_cache_ckpt_dir(
                dataset_cfg,
                pooling,
                ablation_str,
                dna_name,
                prot_name,
                exp["name"],
            )
            os.makedirs(train_cache_ckpt_dir, exist_ok=True)
            shared_best_model_path = f"{train_cache_ckpt_dir}/best_model.pth"
            shared_hybrid_feat_path = f"{train_cache_ckpt_dir}/hybrid_feature_extractor.pth"
            shared_xgb_json_path = f"{train_cache_ckpt_dir}/best_model_xgboost.json"

            profiler = FusionEvaluatorProfiler(f"{exp_dir}/tensorboard_logs", self.fm_profile_json, self.device)
            model = MultiStrategyFusionModel(
                dna_in_dim=dna_dim,
                prot_in_dim=prot_dim,
                active_modalities=active_mods,
                fusion_strategy=exp["fusion"],
            ).to(self.device)

            dummy_batch = next(iter(val_loader))
            d_in_safe = (
                _safe_slice_dummy(dummy_batch, "v_dna", self.device),
                _safe_slice_dummy(dummy_batch, "v_prot", self.device),
                _safe_slice_dummy(dummy_batch, "bio_features", self.device),
                _safe_slice_dummy(dummy_batch, "geom_features", self.device),
            )

            test_metrics, e2e_profiling = {}, {}
            vids_t, y_true_t, y_probs_t, y_preds_t = [], [], [], []

            if exp["type"] == "pytorch":
                profiler.profile_pytorch_fusion(model, d_in_safe)
                if os.path.exists(shared_best_model_path):
                    print("[*] Reuse trained PyTorch checkpoint from shared train/val cache")
                    model.load_state_dict(torch.load(shared_best_model_path, weights_only=True))
                else:
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=self.config["lr"],
                        weight_decay=self.config.get("weight_decay", 1e-3),
                    )
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="max",
                        factor=self.config.get("lr_factor", 0.5),
                        patience=self.config.get("lr_patience", 3),
                    )

                    best_val_mcc = -1.0
                    epochs_no_improve = 0

                    for epoch in range(self.config["epochs"]):
                        model.train()
                        total_loss = 0.0
                        train_steps = 0
                        train_pbar = tqdm(
                            train_loader,
                            desc=f"Epoch {epoch + 1:02d}/{self.config['epochs']} [Train]",
                            leave=False,
                        )
                        for batch in train_pbar:
                            optimizer.zero_grad()
                            logits = model(
                                _safe_tensor(batch, "v_dna", self.device),
                                _safe_tensor(batch, "v_prot", self.device),
                                _safe_tensor(batch, "bio_features", self.device),
                                _safe_tensor(batch, "geom_features", self.device),
                            )
                            loss = criterion(logits, batch["label"].to(self.device))
                            loss.backward()
                            optimizer.step()
                            total_loss += float(loss.item())
                            train_steps += 1

                        model.eval()
                        val_total_loss = 0.0
                        val_steps = 0
                        all_probs_v, all_preds_v, all_labels_v = [], [], []
                        with torch.no_grad():
                            for batch in val_loader:
                                logits = model(
                                    _safe_tensor(batch, "v_dna", self.device),
                                    _safe_tensor(batch, "v_prot", self.device),
                                    _safe_tensor(batch, "bio_features", self.device),
                                    _safe_tensor(batch, "geom_features", self.device),
                                )
                                val_loss = criterion(logits, batch["label"].to(self.device))
                                val_total_loss += float(val_loss.item())
                                val_steps += 1
                                probs = torch.sigmoid(logits)
                                all_probs_v.append(probs.cpu())
                                all_preds_v.append((probs > 0.5).float().cpu())
                                all_labels_v.append(batch["label"].cpu())

                        val_metrics = profiler.compute_metrics(
                            torch.cat(all_labels_v).squeeze(-1).numpy(),
                            torch.cat(all_probs_v).squeeze(-1).numpy(),
                            torch.cat(all_preds_v).squeeze(-1).numpy(),
                        )
                        val_mcc = val_metrics["MCC"]
                        train_loss_epoch = total_loss / max(train_steps, 1)
                        val_loss_epoch = val_total_loss / max(val_steps, 1)
                        profiler.log_epoch_scalars(epoch, train_loss_epoch, val_loss_epoch, val_metrics)
                        scheduler.step(val_mcc)

                        print(
                            f"[Epoch {epoch + 1:02d}] "
                            f"train_loss={train_loss_epoch:.4f} | "
                            f"val_loss={val_loss_epoch:.4f} | "
                            f"val_mcc={val_metrics['MCC']:.4f} | "
                            f"val_f1={val_metrics['F1_Score']:.4f} | "
                            f"val_auroc={val_metrics['AUROC']:.4f}"
                        )

                        if val_mcc > best_val_mcc:
                            best_val_mcc = val_mcc
                            torch.save(model.state_dict(), best_model_path)
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1

                        if epochs_no_improve >= self.config.get("early_stop_patience", 6):
                            print(
                                f"[EarlyStop] epoch={epoch + 1} | "
                                f"best_val_mcc={best_val_mcc:.4f}"
                            )
                            break

                    model.load_state_dict(torch.load(best_model_path, weights_only=True))
                    shutil.copy2(best_model_path, shared_best_model_path)

                if not os.path.exists(best_model_path):
                    shutil.copy2(shared_best_model_path, best_model_path)
                model.eval()
                profiler.reset_memory_stats()

                all_probs_t, all_preds_t, all_labels_t = [], [], []
                with torch.no_grad():
                    for batch in test_loader:
                        profiler.tic_inference()
                        logits = model(
                            _safe_tensor(batch, "v_dna", self.device),
                            _safe_tensor(batch, "v_prot", self.device),
                            _safe_tensor(batch, "bio_features", self.device),
                            _safe_tensor(batch, "geom_features", self.device),
                        )
                        profiler.toc_inference()
                        probs = torch.sigmoid(logits)
                        all_probs_t.append(probs.cpu())
                        all_preds_t.append((probs > 0.5).float().cpu())
                        all_labels_t.append(batch["label"].cpu())
                        vids_t.extend(batch["variant_id"])

                profiler.finalize_fusion_inference_profiling(len(test_loader.dataset))
                y_probs_t = torch.cat(all_probs_t).squeeze(-1).numpy()
                y_preds_t = torch.cat(all_preds_t).squeeze(-1).numpy()
                y_true_t = torch.cat(all_labels_t).squeeze(-1).numpy()

                test_metrics = profiler.compute_metrics(y_true_t, y_probs_t, y_preds_t)
                e2e_profiling = profiler.get_e2e_profiling(
                    dna_name,
                    prot_name,
                    active_mods,
                    split_name=dataset_cfg["test"],
                )

            elif exp["type"] == "hybrid":
                profiler.profile_pytorch_fusion(model, d_in_safe)
                xgb_manager = XGBoostFusionManager()
                if os.path.exists(shared_hybrid_feat_path) and os.path.exists(shared_xgb_json_path):
                    print("[*] Reuse trained Hybrid artifacts from shared train/val cache")
                    model.load_state_dict(torch.load(shared_hybrid_feat_path, weights_only=True))
                    xgb_manager.load_model(train_cache_ckpt_dir, prefix="best_model")
                else:
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["lr"])

                    model.train()
                    for _ in range(2):
                        for batch in train_loader:
                            optimizer.zero_grad()
                            logits = model(
                                _safe_tensor(batch, "v_dna", self.device),
                                _safe_tensor(batch, "v_prot", self.device),
                                _safe_tensor(batch, "bio_features", self.device),
                                _safe_tensor(batch, "geom_features", self.device),
                            )
                            loss = criterion(logits, batch["label"].to(self.device))
                            loss.backward()
                            optimizer.step()

                    f_glob_tr, _, _, _, y_tr, _ = _extract_features_for_ml(model, train_loader, self.device, True)
                    f_glob_vl, _, _, _, y_vl, _ = _extract_features_for_ml(model, val_loader, self.device, True)

                    xgb_manager.train_hybrid(f_glob_tr, y_tr, f_glob_vl, y_vl)
                    xgb_manager.save_model(train_cache_ckpt_dir, prefix="best_model")
                    torch.save(model.state_dict(), shared_hybrid_feat_path)

                f_glob_ts, _, _, _, y_ts, vids_t = _extract_features_for_ml(model, test_loader, self.device, True)

                profiler.reset_memory_stats()
                profiler.tic_inference()
                y_probs_t, y_preds_t, _ = xgb_manager.predict_hybrid(f_glob_ts)
                profiler.toc_inference()
                profiler.finalize_fusion_inference_profiling(len(test_loader.dataset))
                y_true_t = y_ts

                if self.explainability.get("enable_shap", False) or self.explainability.get("enable_lime", False):
                    if "f_glob_tr" not in locals() or f_glob_tr is None:
                        f_glob_tr, _, _, _, _, _ = _extract_features_for_ml(model, train_loader, self.device, True)
                else:
                    f_glob_tr = None

                xgb_manager.save_model(f"{exp_dir}/checkpoints", prefix="best_model")
                if not os.path.exists(f"{exp_dir}/checkpoints/hybrid_feature_extractor.pth"):
                    torch.save(model.state_dict(), f"{exp_dir}/checkpoints/hybrid_feature_extractor.pth")

                test_metrics = profiler.compute_metrics(y_true_t, y_probs_t, y_preds_t)
                e2e_profiling = profiler.get_e2e_profiling(
                    dna_name,
                    prot_name,
                    active_mods,
                    split_name=dataset_cfg["test"],
                )

                self._run_xgb_explainability(
                    exp_dir=test_out_dir,
                    xgb_model=xgb_manager.model,
                    x_train=f_glob_tr,
                    x_test=f_glob_ts,
                    variant_ids=vids_t,
                    feature_prefix="f_global",
                )

                del xgb_manager, f_glob_ts

            elif exp["type"] == "xgboost_pure":
                profiler.profile_pytorch_fusion(model, d_in_safe)
                xgb_manager = XGBoostFusionManager()

                if os.path.exists(shared_xgb_json_path):
                    print("[*] Reuse trained XGBoost-Pure artifacts from shared train/val cache")
                    xgb_manager.load_model(train_cache_ckpt_dir, prefix="best_model")
                else:
                    _, dna_tr_fit, prot_tr_fit, bg_tr_fit, y_tr_fit, _ = _extract_features_for_ml(model, train_loader, self.device, False)
                    _, dna_vl_fit, prot_vl_fit, bg_vl_fit, y_vl_fit, _ = _extract_features_for_ml(model, val_loader, self.device, False)
                    xgb_manager.train_pure(
                        dna_tr_fit,
                        prot_tr_fit,
                        bg_tr_fit,
                        y_tr_fit,
                        dna_vl_fit,
                        prot_vl_fit,
                        bg_vl_fit,
                        y_vl_fit,
                    )
                    xgb_manager.save_model(train_cache_ckpt_dir, prefix="best_model")
                    del dna_tr_fit, prot_tr_fit, bg_tr_fit, y_tr_fit, dna_vl_fit, prot_vl_fit, bg_vl_fit, y_vl_fit

                _, dna_ts, prot_ts, bg_ts, y_ts, vids_t = _extract_features_for_ml(model, test_loader, self.device, False)

                profiler.reset_memory_stats()
                profiler.tic_inference()
                y_probs_t, y_preds_t, _ = xgb_manager.predict_pure(dna_ts, prot_ts, bg_ts)
                profiler.toc_inference()
                profiler.finalize_fusion_inference_profiling(len(test_loader.dataset))

                y_true_t = y_ts
                test_metrics = profiler.compute_metrics(y_true_t, y_probs_t, y_preds_t)
                e2e_profiling = profiler.get_e2e_profiling(
                    dna_name,
                    prot_name,
                    active_mods,
                    split_name=dataset_cfg["test"],
                )

                xgb_manager.save_model(f"{exp_dir}/checkpoints", prefix="best_model")

                if self.explainability.get("enable_shap", False) or self.explainability.get("enable_lime", False):
                    _, dna_tr_exp, prot_tr_exp, bg_tr_exp, _, _ = _extract_features_for_ml(model, train_loader, self.device, False)
                    v_dna_pca_tr, v_prot_pca_tr = xgb_manager.transform_pca(dna_tr_exp, prot_tr_exp)
                    x_train_pure = xgb_manager._build_pure_features(v_dna_pca_tr, v_prot_pca_tr, bg_tr_exp)
                    v_dna_pca_ts, v_prot_pca_ts = xgb_manager.transform_pca(dna_ts, prot_ts)
                    x_test_pure = xgb_manager._build_pure_features(v_dna_pca_ts, v_prot_pca_ts, bg_ts)

                    self._run_xgb_explainability(
                        exp_dir=test_out_dir,
                        xgb_model=xgb_manager.model,
                        x_train=x_train_pure,
                        x_test=x_test_pure,
                        variant_ids=vids_t,
                        feature_prefix="xgb_pure",
                    )

                    del dna_tr_exp, prot_tr_exp, bg_tr_exp
                    del v_dna_pca_tr, v_prot_pca_tr, x_train_pure
                    del v_dna_pca_ts, v_prot_pca_ts, x_test_pure

                del xgb_manager, dna_ts, prot_ts, bg_ts

            else:
                raise ValueError(f"Khong ho tro exp type: {exp['type']}")

            pd.DataFrame(
                {
                    "Variant_ID": vids_t,
                    "True_Label": y_true_t,
                    "Predicted_Probability": y_probs_t,
                    "Prediction_Class": y_preds_t,
                }
            ).to_csv(f"{test_out_dir}/test_probabilities.csv", index=False)

            print(
                f"[Test] {exp_name} | test={dataset_cfg['name']} | "
                f"MCC={test_metrics.get('MCC', 0.0):.4f} | "
                f"F1={test_metrics.get('F1_Score', 0.0):.4f} | "
                f"AUROC={test_metrics.get('AUROC', 0.0):.4f} | "
                f"AUPRC={test_metrics.get('AUPRC', 0.0):.4f}"
            )

            profiler.log_hparams(
                hparam_dict={"fusion": exp["fusion"], "pooling": pooling, "ablation": ablation_str},
                final_metrics={"MCC_Test": test_metrics.get("MCC", 0.0)},
            )
            profiler.close()

            g_metrics.append(
                {
                    "Dataset": dataset_cfg["name"],
                    "Pooling": pooling,
                    "Ablation": ablation_str,
                    "DNA_Model": dna_name,
                    "Prot_Model": prot_name,
                    "Network": exp["name"],
                    **test_metrics,
                }
            )
            g_prof.append({"Experiment": f"{exp_name}__{dataset_cfg['name']}", **e2e_profiling})

            # Persist after each finished experiment so resume survives mid-run crashes.
            self._persist_global_outputs(g_metrics, g_prof)

            del model, profiler
            torch.cuda.empty_cache()
            gc.collect()

        del train_loader, val_loader, test_loader
        gc.collect()

    def run(self, resume: bool = False):
        set_seed(42)
        print(f"[*] Device: {self.device}")
        print(f"[*] Batch run dir: {self.batch_run_dir}")

        if resume:
            global_metrics, global_profiling, completed_by_pooling = self._load_resume_state()
            print(
                f"[*] Resume mode: loaded metrics={len(global_metrics)} rows, profiling={len(global_profiling)} rows"
            )
        else:
            global_metrics = []
            global_profiling = []
            completed_by_pooling = {}

        # Create/update resume artifacts immediately at run start.
        self._persist_global_outputs(global_metrics, global_profiling)

        # Group dataset configs by shared train/val so one trained config can be tested on many test splits.
        dataset_groups = {}
        for ds_cfg in self.datasets:
            key = (ds_cfg["train"], ds_cfg["val"])
            dataset_groups.setdefault(key, []).append(ds_cfg)

        for pooling in self.pooling_strategies:
            print(f"\n{'=' * 80}\n[*] POOLING: {pooling.upper()}\n{'=' * 80}")
            completed_keys = set(completed_by_pooling.get(pooling, set()))

            for _, group_ds_cfgs in dataset_groups.items():
                # Stage 1: enumerate DNA/Protein pairs by config first, then evaluate all test datasets.
                for dna_m in self.dna_models:
                    for prot_m in self.prot_models:
                        for exp in self.experiments:
                            for ds_cfg in group_ds_cfgs:
                                self._run_group(
                                    dna_m,
                                    prot_m,
                                    pooling,
                                    ["dna", "prot"],
                                    ds_cfg,
                                    completed_keys,
                                    global_metrics,
                                    global_profiling,
                                    experiments_subset=[exp],
                                )

                group_dataset_names = {d["name"] for d in group_ds_cfgs}
                pair_ds_scores = {}
                for m in global_metrics:
                    if (
                        m["Pooling"] == pooling
                        and m["Ablation"] == "dna_prot"
                        and m["Dataset"] in group_dataset_names
                    ):
                        key = (m["DNA_Model"], m["Prot_Model"], m["Dataset"])
                        pair_ds_scores.setdefault(key, []).append(m["MCC"])

                if pair_ds_scores:
                    pair_ds_max = {k: np.max(v) for k, v in pair_ds_scores.items()}
                    pair_avg_max = {}
                    for (dna, prot, _), max_mcc in pair_ds_max.items():
                        pair_avg_max.setdefault((dna, prot), []).append(max_mcc)

                    final_scores = {k: np.mean(v) for k, v in pair_avg_max.items()}
                    best_pair_key = max(final_scores, key=final_scores.get)
                    best_dna = next(m for m in self.dna_models if m["name"] == best_pair_key[0])
                    best_prot = next(m for m in self.prot_models if m["name"] == best_pair_key[1])
                else:
                    best_dna, best_prot = self.dna_models[0], self.prot_models[0]

                stage2_cfgs = [
                    (best_dna, None, ["dna"]),
                    (None, best_prot, ["prot"]),
                    (best_dna, best_prot, ["dna", "prot", "geom"]),
                    (best_dna, best_prot, ["dna", "prot", "bio"]),
                    (best_dna, best_prot, ["dna", "prot", "bio", "geom"]),
                    (best_dna, best_prot, ["bio", "geom"]),
                ]

                # Stage 2: train one config then test all test splits in this train/val group.
                for dna_cfg, prot_cfg, active_mods in stage2_cfgs:
                    for exp in self.experiments:
                        for ds_cfg in group_ds_cfgs:
                            self._run_group(
                                dna_cfg,
                                prot_cfg,
                                pooling,
                                active_mods,
                                ds_cfg,
                                completed_keys,
                                global_metrics,
                                global_profiling,
                                experiments_subset=[exp],
                            )

        self._persist_global_outputs(global_metrics, global_profiling)

        df_metrics = pd.read_csv(self.metrics_csv_path) if os.path.exists(self.metrics_csv_path) else pd.DataFrame()
        df_profiling = pd.read_csv(self.profiling_csv_path) if os.path.exists(self.profiling_csv_path) else pd.DataFrame()

        return {
            "batch_run_dir": self.batch_run_dir,
            "metrics_path": self.metrics_csv_path,
            "profiling_path": self.profiling_csv_path,
            "num_rows_metrics": int(len(df_metrics)),
            "num_rows_profiling": int(len(df_profiling)),
        }
