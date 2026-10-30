import os
import json
import gc
import random
import importlib
import shutil
import hashlib
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from core.module05_fusion_classifier.dataset import (
    VariantFusionDataset,
    GEOM_FEATURE_COLUMNS,
    GEOM_V2_FEATURE_COLUMNS,
)
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


def _stable_seed(*parts, base_seed: int = 42) -> int:
    """Derive a reproducible per-config seed without relying on Python hash()."""
    payload = "||".join(str(part) for part in (base_seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


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
        f"{geometry_dir}/{split_name}/{model_name}_{pooling}_geom_v2_norm.parquet",
        f"{geometry_dir}/{split_name}/{model_name}_{pooling}_geom_norm.parquet",
        f"{geometry_dir}/{split_name}/{model_name}_{pooling}_geom_v2.parquet",
        f"{geometry_dir}/{split_name}/{model_name}_{pooling}_geom.parquet",
        # Backward-compatible fallback for older geometry naming.
        f"{geometry_dir}/{split_name}/{model_name}_geom_v2_norm.parquet",
        f"{geometry_dir}/{split_name}/{model_name}_geom_norm.parquet",
        f"{geometry_dir}/{split_name}/{model_name}_geom_v2.parquet",
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
        geom_profile_json: str | None = None,
        geometry_dir: str | None = None,
        geom_feature_names: list | None = None,
        batch_run_dir: str | None = None,
        embedding_dir: str | None = None,
    ):
        self.base_dir = base_dir
        if config is None or datasets is None or models_space is None or pooling_strategies is None or experiments is None:
            raise ValueError(
                "config, datasets, models_space, pooling_strategies, experiments phai duoc khai bao tai notebook"
            )

        self.config = config
        self.seed = int(self.config.get("seed", 42))
        self.xgb_n_jobs = int(self.config.get("xgb_n_jobs", 1))
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
        self.geom_profile_json = geom_profile_json or f"{self.base_dir}/profiling/geom_profiling.json"

        self.bio_dir = f"{self.base_dir}/processed_parquet"
        self.geom_dir = geometry_dir or f"{self.base_dir}/geometry"
        self.geom_feature_names = geom_feature_names
        # Allow isolated ablation runs to use a separate embedding namespace
        # without changing the legacy batch-run layout.
        self.embed_dir = embedding_dir or f"{self.base_dir}/fm_embeddings"

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

    @staticmethod
    def _canonical_key_value(value):
        """Normalize CSV null spellings so resume keys stay stable across runs."""
        if value is None:
            return "None"
        try:
            if pd.isna(value):
                return "None"
        except (TypeError, ValueError):
            pass

        text = str(value).strip()
        if not text or text.lower() in {"none", "nan", "null"}:
            return "None"
        return text

    @staticmethod
    def _canonical_geom_features(value):
        if value is None:
            return "ALL"
        if isinstance(value, (list, tuple)):
            return "+".join(str(item) for item in value) if value else "ALL"
        text = str(value).strip()
        return text if text and text.lower() not in {"none", "nan", "null"} else "ALL"

    def _build_run_key(
        self,
        dataset_name,
        pooling,
        dna_name,
        prot_name,
        ablation_str,
        network_name,
        geom_features=None,
    ):
        values = [
            dataset_name,
            pooling,
            dna_name,
            prot_name,
            ablation_str,
            network_name,
            self._canonical_geom_features(geom_features),
        ]
        return "_".join(self._canonical_key_value(value) for value in values)

    @staticmethod
    def _read_csv_or_empty(path: str) -> pd.DataFrame:
        """Read a CSV, treating missing/zero-byte files as an empty result table."""
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _load_resume_state(self):
        global_metrics = []
        global_profiling = []
        completed_by_pooling = {}

        if os.path.exists(self.metrics_csv_path):
            df_metrics = self._read_csv_or_empty(self.metrics_csv_path)
            req_cols = ["Dataset", "Pooling", "Ablation", "DNA_Model", "Prot_Model", "Network"]
            if all(col in df_metrics.columns for col in req_cols):
                for row in df_metrics.to_dict(orient="records"):
                    global_metrics.append(row)
                    pooling = self._canonical_key_value(row["Pooling"])
                    key = self._build_run_key(
                        row["Dataset"],
                        row["Pooling"],
                        row["DNA_Model"],
                        row["Prot_Model"],
                        row["Ablation"],
                        row["Network"],
                        row.get("Geom_Features", "ALL"),
                    )
                    completed_by_pooling.setdefault(pooling, set()).add(key)
            else:
                print(
                    "[RESUME] global_leaderboard_metrics.csv thieu cot khoa; bo qua resume metrics."
                )

        if os.path.exists(self.profiling_csv_path):
            df_profiling = self._read_csv_or_empty(self.profiling_csv_path)
            global_profiling = df_profiling.to_dict(orient="records")

        return global_metrics, global_profiling, completed_by_pooling

    def _persist_global_outputs(self, global_metrics, global_profiling):
        df_metrics = pd.DataFrame(global_metrics)
        if df_metrics.empty and len(df_metrics.columns) == 0:
            metric_columns = [
                "Dataset", "Pooling", "Ablation", "DNA_Model", "Prot_Model", "Network",
                "Decision_Threshold", "Accuracy", "Precision", "Recall", "Specificity",
                "F1_Score", "MCC", "AUROC", "AUPRC", "Val_Accuracy", "Val_Precision",
                "Val_Recall", "Val_Specificity", "Val_F1_Score", "Val_MCC", "Val_AUROC",
                "Val_AUPRC",
            ]
            df_metrics = pd.DataFrame(columns=metric_columns)
        if not df_metrics.empty and "Dataset" in df_metrics.columns and "MCC" in df_metrics.columns:
            df_metrics = df_metrics.sort_values(by=["Dataset", "MCC"], ascending=[True, False])
        df_metrics.to_csv(self.metrics_csv_path, index=False)

        df_profiling = pd.DataFrame(global_profiling)
        if df_profiling.empty and len(df_profiling.columns) == 0:
            df_profiling = pd.DataFrame(columns=["Experiment", "Geom_Features"])
        df_profiling.to_csv(self.profiling_csv_path, index=False)

    def _get_train_cache_ckpt_dir(
        self,
        dataset_cfg,
        pooling,
        ablation_str,
        dna_name,
        prot_name,
        exp_name,
        geom_features=None,
    ):
        train_val_tag = f"{dataset_cfg['train']}__{dataset_cfg['val']}"
        geom_key = self._canonical_geom_features(geom_features)
        geom_suffix = "" if geom_key == "ALL" else f"/geom_{geom_key}"
        return (
            f"{self.batch_run_dir}/_train_cache/{train_val_tag}/"
            f"{pooling}/{ablation_str}/{dna_name}__{prot_name}/{exp_name}{geom_suffix}/checkpoints"
        )

    def _feature_names(self, n_features: int, prefix: str):
        return [f"{prefix}_{i:04d}" for i in range(n_features)]

    def _select_positive_shap_values(self, shap_values, n_samples: int, n_features: int) -> np.ndarray:
        if isinstance(shap_values, list):
            shap_arr = np.asarray(shap_values[1 if len(shap_values) > 1 else 0])
        else:
            shap_arr = np.asarray(shap_values)

        if shap_arr.ndim == 3:
            if shap_arr.shape[0] == n_samples and shap_arr.shape[1] == n_features:
                shap_arr = shap_arr[:, :, 1 if shap_arr.shape[2] > 1 else 0]
            elif shap_arr.shape[1] == n_samples and shap_arr.shape[2] == n_features:
                shap_arr = shap_arr[1 if shap_arr.shape[0] > 1 else 0, :, :]
            elif shap_arr.shape[0] == n_samples and shap_arr.shape[2] == n_features:
                shap_arr = shap_arr[:, 1 if shap_arr.shape[1] > 1 else 0, :]

        if shap_arr.ndim != 2 or shap_arr.shape != (n_samples, n_features):
            raise ValueError(
                "[SHAP] shap_values khong khop shape ky vong: "
                f"got={shap_arr.shape}, expected=({n_samples}, {n_features})"
            )
        return shap_arr

    def _build_pure_feature_names(
        self,
        dna_dim: int,
        prot_dim: int,
        bg_dim: int,
        dataset,
    ) -> list[str]:
        names = []
        names.extend([f"dna_pca_{i:03d}" for i in range(dna_dim or 0)])
        names.extend([f"prot_pca_{i:03d}" for i in range(prot_dim or 0)])

        bg_names = []
        if bg_dim:
            bg_names.extend(getattr(dataset, "bio_cols", []))
            bg_names.extend(getattr(dataset, "geom_cols", []))
            if len(bg_names) != bg_dim:
                bg_names = [f"tabular_{i:03d}" for i in range(bg_dim)]
        names.extend(bg_names)
        return names

    @staticmethod
    def _stratified_sample_indices(
        labels, max_samples: int, rng, balance_classes: bool = False
    ) -> np.ndarray:
        labels = np.asarray(labels).reshape(-1)
        if max_samples <= 0 or len(labels) == 0:
            return np.array([], dtype=np.int64)
        if len(labels) <= max_samples:
            return np.arange(len(labels), dtype=np.int64)

        classes, counts = np.unique(labels, return_counts=True)
        if max_samples < len(classes):
            return np.sort(rng.choice(len(labels), size=max_samples, replace=False))

        if balance_classes:
            target = np.full(len(classes), max_samples // len(classes), dtype=int)
            target[: max_samples % len(classes)] += 1
            quotas = np.minimum(target, counts)
            exact = target.astype(float)
        else:
            exact = counts * (max_samples / len(labels))
            quotas = np.floor(exact).astype(int)
            quotas = np.maximum(quotas, 1)
        while quotas.sum() > max_samples:
            candidates = np.flatnonzero(quotas > 1)
            if len(candidates) == 0:
                break
            remove_idx = candidates[np.argmin(exact[candidates] - quotas[candidates])]
            quotas[remove_idx] -= 1
        while quotas.sum() < max_samples:
            candidates = np.flatnonzero(quotas < counts)
            if len(candidates) == 0:
                break
            add_idx = candidates[np.argmax(exact[candidates] - quotas[candidates])]
            quotas[add_idx] += 1

        selected = []
        for cls, quota in zip(classes, quotas):
            candidates = np.flatnonzero(labels == cls)
            selected.extend(rng.choice(candidates, size=int(quota), replace=False).tolist())
        return np.sort(np.asarray(selected, dtype=np.int64))

    @staticmethod
    def _feature_group(feature_name: str) -> str:
        name = str(feature_name).lower()
        if name.startswith("dna_pca_"):
            return "dna_pca_component"
        if name.startswith("prot_pca_"):
            return "protein_pca_component"
        if name.startswith(("fusion_latent_", "f_global_")):
            return "learned_fusion_latent"
        return "bio_core_or_spliceai"

    def _save_shap(
        self,
        save_dir: str,
        xgb_model,
        x_background: np.ndarray,
        y_background: np.ndarray,
        x_explain: np.ndarray,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        y_preds: np.ndarray,
        variant_ids: list,
        feature_names: list,
        run_metadata: dict | None = None,
    ):
        if not self.explainability.get("enable_shap", False):
            return
        try:
            shap = importlib.import_module("shap")
        except ImportError:
            print("[SHAP] Package 'shap' is missing; skipping SHAP.")
            return

        max_bg = int(self.explainability.get("max_background_samples", 512))
        max_ex = int(self.explainability.get("max_explain_samples", 200))
        seed = int(self.explainability.get("random_state", 42))
        bg_rng = np.random.default_rng(seed)
        ex_rng = np.random.default_rng(seed + 1)
        bg_idx = self._stratified_sample_indices(y_background, max_bg, bg_rng)
        ex_idx = self._stratified_sample_indices(
            y_true, max_ex, ex_rng, balance_classes=True
        )
        if len(bg_idx) == 0 or len(ex_idx) == 0:
            print("[SHAP] Empty stratified background or explanation sample; skipping SHAP.")
            return

        x_bg = x_background[bg_idx]
        x_ex = x_explain[ex_idx]
        y_ex = np.asarray(y_true)[ex_idx].astype(int)
        probs_ex = np.asarray(y_probs)[ex_idx].astype(float)
        preds_ex = np.asarray(y_preds)[ex_idx].astype(int)
        ex_vids = [variant_ids[i] for i in ex_idx.tolist()]
        if len(feature_names) != x_ex.shape[1]:
            print(
                "[SHAP] Feature-name count does not match input columns; skipping: "
                f"names={len(feature_names)}, columns={x_ex.shape[1]}"
            )
            return

        explainer_kwargs = {
            "data": x_bg,
            "feature_perturbation": self.explainability.get("shap_feature_perturbation", "interventional"),
            "model_output": self.explainability.get("shap_model_output", "raw"),
        }
        try:
            explainer = shap.TreeExplainer(xgb_model, **explainer_kwargs)
        except Exception as exc:
            print(f"[SHAP] Interventional explainer failed; using default TreeExplainer: {exc}")
            explainer = shap.TreeExplainer(xgb_model)

        try:
            raw_shap_values = explainer.shap_values(x_ex, check_additivity=False)
        except TypeError:
            raw_shap_values = explainer.shap_values(x_ex)
        try:
            shap_values = self._select_positive_shap_values(
                raw_shap_values, n_samples=x_ex.shape[0], n_features=x_ex.shape[1]
            )
        except ValueError as exc:
            print(str(exc))
            return

        expected = np.asarray(explainer.expected_value).reshape(-1)
        base_value = float(expected[-1]) if len(expected) else float("nan")
        feature_groups = [self._feature_group(name) for name in feature_names]
        global_rows = []
        for idx, name in enumerate(feature_names):
            global_rows.append({
                "feature": name,
                "feature_group": feature_groups[idx],
                "mean_abs_shap": float(np.mean(np.abs(shap_values[:, idx]))),
                "explained_samples": int(len(ex_idx)),
                "sampling": "class_balanced_stratified",
            })
        pd.DataFrame(global_rows).sort_values(
            "mean_abs_shap", ascending=False
        ).to_csv(f"{save_dir}/shap_global_importance.csv", index=False)

        class_global_rows = []
        for cls, class_name in ((0, "Benign"), (1, "Pathogenic")):
            class_mask = y_ex == cls
            if not np.any(class_mask):
                continue
            for idx, name in enumerate(feature_names):
                class_global_rows.append({
                    "true_label": cls,
                    "true_class": class_name,
                    "feature": name,
                    "feature_group": feature_groups[idx],
                    "mean_abs_shap": float(np.mean(np.abs(shap_values[class_mask, idx]))),
                    "explained_samples": int(class_mask.sum()),
                })
        pd.DataFrame(class_global_rows).to_csv(
            f"{save_dir}/shap_global_importance_by_true_class.csv", index=False
        )

        top_k = min(int(self.explainability.get("shap_top_k", 30)), shap_values.shape[1])
        local_rows = []
        for r, vid in enumerate(ex_vids):
            local_idx = np.argsort(np.abs(shap_values[r]))[::-1][:top_k]
            for rank, col in enumerate(local_idx.tolist(), start=1):
                local_rows.append({
                    "Variant_ID": vid,
                    "true_label": int(y_ex[r]),
                    "true_class": "Pathogenic" if y_ex[r] == 1 else "Benign",
                    "predicted_probability_pathogenic": float(probs_ex[r]),
                    "predicted_label": int(preds_ex[r]),
                    "predicted_class": "Pathogenic" if preds_ex[r] == 1 else "Benign",
                    "explained_class": "Pathogenic",
                    "base_value_raw_margin": base_value,
                    "rank": rank,
                    "feature": feature_names[col],
                    "feature_group": feature_groups[col],
                    "shap_value_raw_margin": float(shap_values[r, col]),
                    "abs_shap_raw_margin": float(abs(shap_values[r, col])),
                })
        pd.DataFrame(local_rows).to_csv(f"{save_dir}/shap_local_top_ranked.csv", index=False)

        class_counts = {str(int(cls)): int((y_ex == cls).sum()) for cls in np.unique(y_ex)}
        info = {
            "background_samples": int(len(bg_idx)),
            "background_class_counts": {str(int(cls)): int((np.asarray(y_background)[bg_idx] == cls).sum()) for cls in np.unique(np.asarray(y_background)[bg_idx])},
            "explained_samples": int(len(ex_idx)),
            "explained_true_class_counts": class_counts,
            "input_features": int(x_ex.shape[1]),
            "feature_space": (run_metadata or {}).get("feature_space", "unspecified"),
            "background_sampling": "stratified_by_training_label_prevalence",
            "explanation_sampling": "class_balanced_stratified",
            "random_state": seed,
            "feature_perturbation": explainer_kwargs["feature_perturbation"],
            "model_output": explainer_kwargs["model_output"],
            "explained_class": "Pathogenic (label 1)",
            **(run_metadata or {}),
        }
        with open(f"{save_dir}/shap_run_info.json", "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2, default=str)

    def _save_lime(
        self,
        save_dir: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_explain: np.ndarray,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        y_preds: np.ndarray,
        variant_ids: list,
        feature_names: list,
        predict_proba_fn,
        run_metadata: dict | None = None,
    ):
        if not self.explainability.get("enable_lime", False):
            return
        try:
            LimeTabularExplainer = importlib.import_module("lime.lime_tabular").LimeTabularExplainer
        except ImportError:
            print("[LIME] Package 'lime' is missing; skipping LIME.")
            return

        max_ex = int(self.explainability.get("max_lime_samples", 20))
        n_feat = int(self.explainability.get("lime_num_features", 20))
        max_bg = int(self.explainability.get("max_background_samples", 512))
        num_samples = int(self.explainability.get("lime_num_samples", 2000))
        seed = int(self.explainability.get("random_state", 42))
        bg_rng = np.random.default_rng(seed)
        ex_rng = np.random.default_rng(seed + 2)
        bg_idx = self._stratified_sample_indices(y_train, max_bg, bg_rng)
        ex_idx = self._stratified_sample_indices(
            y_true, max_ex, ex_rng, balance_classes=True
        )
        if len(bg_idx) == 0 or len(ex_idx) == 0:
            print("[LIME] Empty stratified background or explanation sample; skipping LIME.")
            return

        x_bg = x_train[bg_idx]
        explainer = LimeTabularExplainer(
            training_data=x_bg,
            feature_names=feature_names,
            class_names=["Benign", "Pathogenic"],
            mode="classification",
            discretize_continuous=True,
            random_state=seed,
        )
        y_true_arr = np.asarray(y_true).astype(int)
        y_prob_arr = np.asarray(y_probs).astype(float)
        y_pred_arr = np.asarray(y_preds).astype(int)
        rows = []
        for i in ex_idx.tolist():
            exp = explainer.explain_instance(
                x_explain[i], predict_proba_fn, labels=(1,), num_features=n_feat,
                num_samples=num_samples,
            )
            local_pred = getattr(exp, "local_pred", None)
            local_pred_pathogenic = (
                float(np.asarray(local_pred).reshape(-1)[-1])
                if local_pred is not None and np.asarray(local_pred).size
                else float("nan")
            )
            for rank, (feat_expr, weight) in enumerate(exp.as_list(label=1), start=1):
                rows.append({
                    "Variant_ID": variant_ids[i],
                    "true_label": int(y_true_arr[i]),
                    "true_class": "Pathogenic" if y_true_arr[i] == 1 else "Benign",
                    "predicted_probability_pathogenic": float(y_prob_arr[i]),
                    "predicted_label": int(y_pred_arr[i]),
                    "predicted_class": "Pathogenic" if y_pred_arr[i] == 1 else "Benign",
                    "explained_class": "Pathogenic (label 1)",
                    "lime_surrogate_r2": float(exp.score),
                    "lime_local_prediction_pathogenic": local_pred_pathogenic,
                    "lime_num_samples": num_samples,
                    "rank": rank,
                    "feature_rule": feat_expr,
                    "weight_for_pathogenic_probability": float(weight),
                })
        pd.DataFrame(rows).to_csv(f"{save_dir}/lime_local_explanations.csv", index=False)
        info = {
            "background_samples": int(len(bg_idx)),
            "background_class_counts": {str(int(cls)): int((np.asarray(y_train)[bg_idx] == cls).sum()) for cls in np.unique(np.asarray(y_train)[bg_idx])},
            "explained_samples": int(len(ex_idx)),
            "explained_true_class_counts": {str(int(cls)): int((y_true_arr[ex_idx] == cls).sum()) for cls in np.unique(y_true_arr[ex_idx])},
            "feature_space": (run_metadata or {}).get("feature_space", "unspecified"),
            "background_sampling": "stratified_by_training_label_prevalence",
            "explanation_sampling": "class_balanced_stratified",
            "random_state": seed,
            "explained_class": "Pathogenic (label 1)",
            "num_features_per_explanation": n_feat,
            "num_samples_per_explanation": num_samples,
            **(run_metadata or {}),
        }
        with open(f"{save_dir}/lime_run_info.json", "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2, default=str)

    def _run_xgb_explainability(
        self,
        exp_dir: str,
        xgb_model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        y_preds: np.ndarray,
        variant_ids: list,
        feature_prefix: str,
        feature_names: list | None = None,
        run_metadata: dict | None = None,
    ):
        if x_train is None or x_test is None or len(variant_ids) == 0:
            return
        if x_train.shape[1] != x_test.shape[1]:
            print(
                "[Explainability] Train/test feature dimension mismatch; skipping: "
                f"train={x_train.shape[1]}, test={x_test.shape[1]}"
            )
            return
        explain_dir = f"{exp_dir}/explainability"
        os.makedirs(explain_dir, exist_ok=True)
        if feature_names is None:
            feature_names = self._feature_names(x_test.shape[1], feature_prefix)
        if len(feature_names) != x_test.shape[1]:
            print(
                "[Explainability] Feature-name count does not match input columns; skipping: "
                f"names={len(feature_names)}, input_dim={x_test.shape[1]}"
            )
            return
        self._save_shap(
            explain_dir, xgb_model, x_train, y_train, x_test, y_true, y_probs,
            y_preds, variant_ids, feature_names, run_metadata,
        )
        self._save_lime(
            explain_dir, x_train, y_train, x_test, y_true, y_probs, y_preds,
            variant_ids, feature_names, xgb_model.predict_proba, run_metadata,
        )

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

    def _get_dataloader(
        self,
        split_name,
        pooling,
        dna_name,
        prot_name,
        active_mods,
        is_train,
        geom_feature_names=None,
        geometry_context=None,
        loader_seed=None,
    ):
        geometry_active = "geom" in [str(mod).lower() for mod in active_mods]
        selected_geom = set(geom_feature_names or self.geom_feature_names or GEOM_FEATURE_COLUMNS)
        use_dna_geom = geometry_active and any(name.startswith("dna_") for name in selected_geom)
        use_prot_geom = geometry_active and any(name.startswith("prot_") for name in selected_geom)
        geometry_dir = self.geom_dir
        if "{context}" in geometry_dir:
            if geometry_context is None:
                raise ValueError("geometry_dir co {context} nhung chua truyen geometry_context")
            geometry_dir = geometry_dir.replace("{context}", str(geometry_context))
        dataset = VariantFusionDataset(
            bio_parquet_path=_resolve_bio_path(self.bio_dir, split_name),
            dna_geom_path=_resolve_geom_path(geometry_dir, split_name, dna_name, pooling)
            if dna_name != "None" and use_dna_geom else None,
            prot_geom_path=_resolve_geom_path(geometry_dir, split_name, prot_name, pooling)
            if prot_name != "None" and use_prot_geom else None,
            dna_pt_path=_resolve_pt_path(self.embed_dir, split_name, dna_name, pooling) if dna_name != "None" else None,
            prot_pt_path=_resolve_pt_path(self.embed_dir, split_name, prot_name, pooling) if prot_name != "None" else None,
            active_modalities=active_mods,
            geom_feature_names=geom_feature_names or self.geom_feature_names,
            is_train=is_train,
        )
        generator = torch.Generator()
        generator.manual_seed(self.seed if loader_seed is None else int(loader_seed))
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=is_train,
            num_workers=self.config["num_workers"],
            pin_memory=(self.device.type == "cuda"),
            generator=generator,
        )

    @staticmethod
    def _select_validation_threshold(y_true, y_probs) -> tuple[float, float]:
        """Choose validation threshold maximizing MCC in O(n log n); ties prefer 0.5."""
        y_true = np.asarray(y_true, dtype=int).reshape(-1)
        y_probs = np.nan_to_num(
            np.asarray(y_probs, dtype=float).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0
        )
        if len(y_true) != len(y_probs) or len(y_true) == 0:
            raise ValueError("Validation labels/probabilities are empty or have different lengths.")
        y_probs = np.clip(y_probs, 0.0, 1.0)
        order = np.argsort(-y_probs, kind="mergesort")
        sorted_probs = y_probs[order]
        sorted_positive = y_true[order] == 1
        cumulative_tp = np.concatenate(([0], np.cumsum(sorted_positive, dtype=np.int64)))
        total_positive = int(sorted_positive.sum())
        total_negative = len(y_true) - total_positive
        candidates = np.unique(np.concatenate(([0.0, 0.5], sorted_probs, [np.nextafter(sorted_probs[0], np.inf)])))

        best_threshold, best_mcc = 0.5, -float("inf")
        for threshold in candidates:
            predicted_positive = int(np.searchsorted(-sorted_probs, -threshold, side="right"))
            tp = int(cumulative_tp[predicted_positive])
            fp = predicted_positive - tp
            fn = total_positive - tp
            tn = total_negative - fp
            denominator = np.sqrt(
                (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            )
            score = float((tp * tn - fp * fn) / denominator) if denominator else 0.0
            if score > best_mcc or (
                np.isclose(score, best_mcc)
                and abs(threshold - 0.5) < abs(best_threshold - 0.5)
            ):
                best_threshold, best_mcc = float(threshold), score
        return best_threshold, best_mcc

    @staticmethod
    def _threshold_predictions(y_probs, threshold: float) -> np.ndarray:
        probs = np.nan_to_num(
            np.asarray(y_probs, dtype=float).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0
        )
        return (probs >= float(threshold)).astype(int)

    @staticmethod
    def _save_decision_threshold(paths, threshold: float, val_mcc: float):
        payload = {
            "decision_threshold": float(threshold),
            "selection_metric": "MCC",
            "validation_mcc_at_threshold": float(val_mcc),
            "prediction_rule": "positive if probability >= decision_threshold",
        }
        for path in paths:
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)

    def _evaluate_pytorch_loader(self, model, dataloader, profiler):
        """Collect selected-checkpoint probabilities/labels on validation."""
        model.eval()
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                logits = model(
                    _safe_tensor(batch, "v_dna", self.device),
                    _safe_tensor(batch, "v_prot", self.device),
                    _safe_tensor(batch, "bio_features", self.device),
                    _safe_tensor(batch, "geom_features", self.device),
                )
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())
                all_preds.append((probs > 0.5).float().cpu())
                all_labels.append(batch["label"].cpu())

        y_probs = torch.cat(all_probs).squeeze(-1).numpy()
        y_true = torch.cat(all_labels).squeeze(-1).numpy()
        return y_true, y_probs

    def _run_group(
        self,
        dna_cfg,
        prot_cfg,
        pooling,
        active_mods,
        dataset_cfg,
        completed_keys,
        g_metrics,
        g_prof,
        experiments_subset=None,
        geom_feature_names=None,
        ablation_label=None,
        run_explainability=None,
    ):
        # Keep model names for geometry file resolution even when DNA/Prot are not active modalities.
        dna_name = dna_cfg["name"] if dna_cfg else "None"
        prot_name = prot_cfg["name"] if prot_cfg else "None"
        ablation_str = ablation_label or "_".join(sorted(active_mods))
        effective_geom_feature_names = geom_feature_names
        if "geom" in active_mods and effective_geom_feature_names is None:
            effective_geom_feature_names = self.geom_feature_names
        geom_feature_key = self._canonical_geom_features(effective_geom_feature_names)
        num_seq = int("dna" in active_mods) + int("prot" in active_mods)

        config_seed = _stable_seed(
            dataset_cfg["train"],
            dataset_cfg["val"],
            pooling,
            dna_name,
            prot_name,
            ",".join(sorted(str(mod).lower() for mod in active_mods)),
            geom_feature_key,
            base_seed=self.seed,
        )

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
                geom_feature_key,
            )
            if run_key not in completed_keys:
                tasks_to_run.append((exp, run_key))

        if not tasks_to_run:
            return

        explainability_enabled = (
            self.explainability.get("enable_shap", False)
            or self.explainability.get("enable_lime", False)
        ) if run_explainability is None else bool(run_explainability)

        dna_dim = self._get_cached_dim(dna_name, dataset_cfg["train"], pooling) if "dna" in active_mods else 0
        prot_dim = self._get_cached_dim(prot_name, dataset_cfg["train"], pooling) if "prot" in active_mods else 0

        train_loader = self._get_dataloader(
            dataset_cfg["train"], pooling, dna_name, prot_name, active_mods, True,
            effective_geom_feature_names, dataset_cfg["train"]
        )
        val_loader = self._get_dataloader(
            dataset_cfg["val"], pooling, dna_name, prot_name, active_mods, False,
            effective_geom_feature_names, dataset_cfg["train"]
        )
        test_loader = self._get_dataloader(
            dataset_cfg["test"], pooling, dna_name, prot_name, active_mods, False,
            effective_geom_feature_names, dataset_cfg["train"], config_seed
        )
        train_dataset = train_loader.dataset
        bio_dim = len(train_dataset.bio_cols) if train_dataset.has_bio else 0
        geom_dim = len(train_dataset.geom_cols) if train_dataset.has_geom else 0

        for exp, run_key in tasks_to_run:
            completed_keys.add(run_key)
            exp_seed = _stable_seed(exp["name"], base_seed=config_seed)
            set_seed(exp_seed)
            if getattr(train_loader, "generator", None) is not None:
                train_loader.generator.manual_seed(exp_seed)
            config_tag = f"{dataset_cfg['train']}__{dataset_cfg['val']}"
            geom_exp_suffix = "" if geom_feature_key == "ALL" else f"__geom_{geom_feature_key}"
            exp_name = f"{config_tag}_{pooling}_{ablation_str}_{dna_name}_{prot_name}_{exp['name']}{geom_exp_suffix}"
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
                geom_feature_key,
            )
            os.makedirs(train_cache_ckpt_dir, exist_ok=True)
            shared_best_model_path = f"{train_cache_ckpt_dir}/best_model.pth"
            shared_hybrid_feat_path = f"{train_cache_ckpt_dir}/hybrid_feature_extractor.pth"
            shared_xgb_json_path = f"{train_cache_ckpt_dir}/best_model_xgboost.json"

            profiler = FusionEvaluatorProfiler(
                f"{exp_dir}/tensorboard_logs",
                self.fm_profile_json,
                self.device,
                geom_profile_json=self.geom_profile_json,
            )
            model = MultiStrategyFusionModel(
                dna_in_dim=dna_dim,
                prot_in_dim=prot_dim,
                active_modalities=active_mods,
                bio_in_dim=bio_dim,
                geom_in_dim=geom_dim,
                fusion_strategy=exp["fusion"],
            ).to(self.device)

            dummy_batch = next(iter(val_loader))
            d_in_safe = (
                _safe_slice_dummy(dummy_batch, "v_dna", self.device),
                _safe_slice_dummy(dummy_batch, "v_prot", self.device),
                _safe_slice_dummy(dummy_batch, "bio_features", self.device),
                _safe_slice_dummy(dummy_batch, "geom_features", self.device),
            )

            val_metrics, test_metrics, e2e_profiling = {}, {}, {}
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

                # Re-evaluate the selected checkpoint on validation so the
                # leaderboard row contains the selection metric as well as test metrics.
                y_true_v, y_probs_v = self._evaluate_pytorch_loader(model, val_loader, profiler)
                decision_threshold, val_mcc_at_threshold = self._select_validation_threshold(y_true_v, y_probs_v)
                y_preds_v = self._threshold_predictions(y_probs_v, decision_threshold)
                val_metrics = profiler.compute_metrics(y_true_v, y_probs_v, y_preds_v)
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
                y_true_t = torch.cat(all_labels_t).squeeze(-1).numpy()
                y_preds_t = self._threshold_predictions(y_probs_t, decision_threshold)

                test_metrics = profiler.compute_metrics(y_true_t, y_probs_t, y_preds_t)
                e2e_profiling = profiler.get_e2e_profiling(
                    dna_name,
                    prot_name,
                    active_mods,
                    split_name=dataset_cfg["test"],
                    pooling=pooling,
                )

            elif exp["type"] == "hybrid":
                profiler.profile_pytorch_fusion(model, d_in_safe)
                xgb_manager = XGBoostFusionManager(
                    random_state=self.seed,
                    n_jobs=self.xgb_n_jobs,
                )
                f_glob_tr = None
                y_train_explain = None
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
                    y_train_explain = y_tr
                    f_glob_vl, _, _, _, y_vl, _ = _extract_features_for_ml(model, val_loader, self.device, True)

                    xgb_manager.train_hybrid(f_glob_tr, y_tr, f_glob_vl, y_vl)
                    xgb_manager.save_model(train_cache_ckpt_dir, prefix="best_model")
                    torch.save(model.state_dict(), shared_hybrid_feat_path)

                profiler.add_xgboost_model_size(xgb_manager.model)
                f_glob_vl, _, _, _, y_vl, _ = _extract_features_for_ml(
                    model, val_loader, self.device, True
                )
                y_probs_v, y_preds_v, _ = xgb_manager.predict_hybrid(f_glob_vl)
                decision_threshold, val_mcc_at_threshold = self._select_validation_threshold(y_vl, y_probs_v)
                y_preds_v = self._threshold_predictions(y_probs_v, decision_threshold)
                val_metrics = profiler.compute_metrics(y_vl, y_probs_v, y_preds_v)
                del f_glob_vl
                profiler.reset_memory_stats()
                profiler.tic_inference()
                f_glob_ts, _, _, _, y_ts, vids_t = _extract_features_for_ml(model, test_loader, self.device, True)
                y_probs_t, y_preds_t, _ = xgb_manager.predict_hybrid(f_glob_ts)
                profiler.toc_inference()
                profiler.finalize_fusion_inference_profiling(len(test_loader.dataset))
                y_true_t = y_ts
                y_preds_t = self._threshold_predictions(y_probs_t, decision_threshold)

                if explainability_enabled:
                    if f_glob_tr is None:
                        f_glob_tr, _, _, _, y_train_explain, _ = _extract_features_for_ml(model, train_loader, self.device, True)
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
                    pooling=pooling,
                )

                if explainability_enabled:
                    self._run_xgb_explainability(
                        exp_dir=test_out_dir,
                        xgb_model=xgb_manager.model,
                        x_train=f_glob_tr,
                        y_train=y_train_explain,
                        x_test=f_glob_ts,
                        y_true=y_true_t,
                        y_probs=y_probs_t,
                        y_preds=y_preds_t,
                        variant_ids=vids_t,
                        feature_prefix="fusion_latent",
                        run_metadata={
                            "feature_space": "learned_fusion_latent",
                            "dataset": dataset_cfg["name"],
                            "network": exp["name"],
                            "pooling": pooling,
                            "dna_model": dna_name,
                            "protein_model": prot_name,
                        },
                    )

                del xgb_manager, f_glob_ts

            elif exp["type"] == "xgboost_pure":
                profiler.profile_non_pytorch_fusion()
                xgb_manager = XGBoostFusionManager(
                    random_state=self.seed,
                    n_jobs=self.xgb_n_jobs,
                )

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

                _, dna_vl, prot_vl, bg_vl, y_vl, _ = _extract_features_for_ml(
                    model, val_loader, self.device, False
                )
                y_probs_v, y_preds_v, _ = xgb_manager.predict_pure(dna_vl, prot_vl, bg_vl)
                decision_threshold, val_mcc_at_threshold = self._select_validation_threshold(y_vl, y_probs_v)
                y_preds_v = self._threshold_predictions(y_probs_v, decision_threshold)
                val_metrics = profiler.compute_metrics(y_vl, y_probs_v, y_preds_v)
                del dna_vl, prot_vl, bg_vl, y_vl

                _, dna_ts, prot_ts, bg_ts, y_ts, vids_t = _extract_features_for_ml(model, test_loader, self.device, False)

                profiler.add_xgboost_model_size(xgb_manager.model)
                profiler.reset_memory_stats()
                profiler.tic_inference()
                y_probs_t, y_preds_t, _ = xgb_manager.predict_pure(dna_ts, prot_ts, bg_ts)
                profiler.toc_inference()
                profiler.finalize_fusion_inference_profiling(len(test_loader.dataset))

                y_true_t = y_ts
                y_preds_t = self._threshold_predictions(y_probs_t, decision_threshold)
                test_metrics = profiler.compute_metrics(y_true_t, y_probs_t, y_preds_t)
                e2e_profiling = profiler.get_e2e_profiling(
                    dna_name,
                    prot_name,
                    active_mods,
                    split_name=dataset_cfg["test"],
                    pooling=pooling,
                )

                xgb_manager.save_model(f"{exp_dir}/checkpoints", prefix="best_model")

                if explainability_enabled:
                    _, dna_tr_exp, prot_tr_exp, bg_tr_exp, y_train_explain, _ = _extract_features_for_ml(model, train_loader, self.device, False)
                    v_dna_pca_tr, v_prot_pca_tr = xgb_manager.transform_pca(dna_tr_exp, prot_tr_exp)
                    x_train_pure = xgb_manager._build_pure_features(v_dna_pca_tr, v_prot_pca_tr, bg_tr_exp)
                    v_dna_pca_ts, v_prot_pca_ts = xgb_manager.transform_pca(dna_ts, prot_ts)
                    x_test_pure = xgb_manager._build_pure_features(v_dna_pca_ts, v_prot_pca_ts, bg_ts)
                    pure_feature_names = self._build_pure_feature_names(
                        dna_dim=0 if v_dna_pca_ts is None else v_dna_pca_ts.shape[1],
                        prot_dim=0 if v_prot_pca_ts is None else v_prot_pca_ts.shape[1],
                        bg_dim=0 if bg_ts is None else bg_ts.shape[1],
                        dataset=test_loader.dataset,
                    )

                    self._run_xgb_explainability(
                        exp_dir=test_out_dir,
                        xgb_model=xgb_manager.model,
                        x_train=x_train_pure,
                        y_train=y_train_explain,
                        x_test=x_test_pure,
                        y_true=y_true_t,
                        y_probs=y_probs_t,
                        y_preds=y_preds_t,
                        variant_ids=vids_t,
                        feature_prefix="xgb_pure",
                        feature_names=pure_feature_names,
                        run_metadata={
                            "feature_space": "dna_protein_pca_plus_biological_features",
                            "dataset": dataset_cfg["name"],
                            "network": exp["name"],
                            "pooling": pooling,
                            "dna_model": dna_name,
                            "protein_model": prot_name,
                        },
                    )

                    del dna_tr_exp, prot_tr_exp, bg_tr_exp
                    del v_dna_pca_tr, v_prot_pca_tr, x_train_pure
                    del v_dna_pca_ts, v_prot_pca_ts, x_test_pure

                del xgb_manager, dna_ts, prot_ts, bg_ts

            else:
                raise ValueError(f"Khong ho tro exp type: {exp['type']}")

            self._save_decision_threshold(
                [
                    f"{train_cache_ckpt_dir}/decision_threshold.json",
                    f"{exp_dir}/checkpoints/decision_threshold.json",
                ],
                decision_threshold,
                val_mcc_at_threshold,
            )
            pd.DataFrame(
                {
                    "Variant_ID": vids_t,
                    "True_Label": y_true_t,
                    "Predicted_Probability": y_probs_t,
                    "Decision_Threshold": decision_threshold,
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

            metric_row = {
                "Dataset": dataset_cfg["name"],
                "Pooling": pooling,
                "Ablation": ablation_str,
                "DNA_Model": dna_name,
                "Prot_Model": prot_name,
                "Network": exp["name"],
                "Decision_Threshold": float(decision_threshold),
                **test_metrics,
                **{f"Val_{name}": value for name, value in val_metrics.items()},
            }
            if geom_feature_names is not None:
                metric_row["Geom_Features"] = geom_feature_key
            g_metrics.append(metric_row)
            g_prof.append(
                {
                    "Experiment": f"{exp_name}__{dataset_cfg['name']}",
                    "Geom_Features": geom_feature_key,
                    **e2e_profiling,
                }
            )

            # Persist after each finished experiment so resume survives mid-run crashes.
            self._persist_global_outputs(g_metrics, g_prof)

            del model, profiler
            torch.cuda.empty_cache()
            gc.collect()

        del train_loader, val_loader, test_loader
        gc.collect()

    def run_geom_ablation(
        self,
        resume: bool = True,
        dna_model_name: str = "nt_v2_500m",
        prot_model_name: str = "esm1b_650m",
        pooling: str = "mean",
        experiment_name: str = "Pure_XGBoost_Concat",
        run_explainability: bool = False,
    ):
        """Run all non-empty subsets of the configured geometry features.

        This is intentionally isolated from ``run`` because the geometry
        search has a fixed model/pooling/network matrix and should not rerun
        the broader benchmark stages.
        """
        if pooling != "mean":
            raise ValueError("Geom ablation hien tai chi ho tro pooling='mean'")

        dna_cfg = next((m for m in self.dna_models if m["name"] == dna_model_name), None)
        prot_cfg = next((m for m in self.prot_models if m["name"] == prot_model_name), None)
        if dna_cfg is None:
            raise ValueError(f"Khong tim thay DNA model trong MODELS_SPACE: {dna_model_name}")
        if prot_cfg is None:
            raise ValueError(f"Khong tim thay protein model trong MODELS_SPACE: {prot_model_name}")

        experiment = next((e for e in self.experiments if e["name"] == experiment_name), None)
        if experiment is None:
            raise ValueError(f"Khong tim thay experiment trong EXPERIMENTS: {experiment_name}")
        if experiment.get("type") != "xgboost_pure":
            raise ValueError("Geom ablation hien tai phai dung experiment type='xgboost_pure'")

        geom_search_space = list(self.geom_feature_names or GEOM_FEATURE_COLUMNS)
        geom_subsets = [
            list(subset)
            for size in range(1, len(geom_search_space) + 1)
            for subset in combinations(geom_search_space, size)
        ]
        expected_subsets = (2 ** len(geom_search_space)) - 1
        if len(geom_subsets) != expected_subsets:
            raise AssertionError(f"So geom subset khong dung: {len(geom_subsets)}")

        if resume:
            global_metrics, global_profiling, completed_by_pooling = self._load_resume_state()
            print(
                f"[*] Geom ablation resume: loaded metrics={len(global_metrics)} rows, "
                f"profiling={len(global_profiling)} rows"
            )
        else:
            global_metrics, global_profiling, completed_by_pooling = [], [], {}

        self._persist_global_outputs(global_metrics, global_profiling)
        completed_keys = set(completed_by_pooling.get(pooling, set()))
        active_mods = ["dna", "prot", "bio_core", "spliceai", "geom"]
        ablation_label = "bio_core_dna_prot_spliceai_geom"
        dataset_groups = {}
        for ds_cfg in self.datasets:
            dataset_groups.setdefault((ds_cfg["train"], ds_cfg["val"]), []).append(ds_cfg)

        expected_rows = len(geom_subsets) * len(self.datasets)
        print(
            f"[*] Geom ablation: {len(geom_subsets)} subsets | "
            f"{len(self.datasets)} test datasets | expected rows={expected_rows}"
        )

        for index, geom_subset in enumerate(geom_subsets, start=1):
            geom_key = self._canonical_geom_features(geom_subset)
            print(f"\n[Geom {index:03d}/{len(geom_subsets)}] {geom_key}")
            for _, group_ds_cfgs in dataset_groups.items():
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
                        experiments_subset=[experiment],
                        geom_feature_names=geom_subset,
                        ablation_label=ablation_label,
                        run_explainability=run_explainability,
                    )

        self._persist_global_outputs(global_metrics, global_profiling)
        df_metrics = self._read_csv_or_empty(self.metrics_csv_path)
        df_profiling = self._read_csv_or_empty(self.profiling_csv_path)
        geom_rows = (
            int((df_metrics["Ablation"] == ablation_label).sum())
            if "Ablation" in df_metrics.columns
            else 0
        )
        return {
            "batch_run_dir": self.batch_run_dir,
            "metrics_path": self.metrics_csv_path,
            "profiling_path": self.profiling_csv_path,
            "num_rows_metrics": int(len(df_metrics)),
            "num_rows_profiling": int(len(df_profiling)),
            "geom_ablation_rows": geom_rows,
            "geom_subset_count": len(geom_subsets),
            "expected_geom_rows": expected_rows,
        }

    def run_fixed_geometry_ablation(
        self,
        geometry_configs,
        dna_model_name="nt_v2_500m",
        prot_model_name="esm1b_650m",
        pooling="mean",
        experiment_names=None,
        resume=True,
        run_explainability=False,
    ):
        """Run a controlled geometry/architecture matrix.

        Each item in ``geometry_configs`` must contain ``label`` and
        ``features``. The label is part of the resume key, which lets the
        same feature names be evaluated with different LID-k roots.
        """
        if pooling not in self.pooling_strategies:
            raise ValueError(f"Pooling khong co trong pipeline: {pooling}")

        dna_cfg = next((m for m in self.dna_models if m["name"] == dna_model_name), None)
        prot_cfg = next((m for m in self.prot_models if m["name"] == prot_model_name), None)
        if dna_cfg is None or prot_cfg is None:
            raise ValueError("Khong tim thay DNA/Protein model cho geometry ablation")

        selected_experiments = self.experiments
        if experiment_names is not None:
            wanted = set(experiment_names)
            selected_experiments = [e for e in self.experiments if e["name"] in wanted]
            missing = wanted - {e["name"] for e in selected_experiments}
            if missing:
                raise ValueError(f"Khong tim thay architecture: {sorted(missing)}")

        if resume:
            global_metrics, global_profiling, completed_by_pooling = self._load_resume_state()
        else:
            global_metrics, global_profiling, completed_by_pooling = [], [], {}
        completed_keys = set(completed_by_pooling.get(pooling, set()))
        active_mods = ["dna", "prot", "bio_core", "spliceai", "geom"]

        for geom_cfg in geometry_configs:
            label = str(geom_cfg["label"])
            features = list(geom_cfg["features"])
            if not features:
                raise ValueError(f"Geometry config rong: {label}")
            print(
                f"\n[Geometry Matrix] {label} | features={'+'.join(features)} | "
                f"architectures={len(selected_experiments)}"
            )
            for experiment in selected_experiments:
                for dataset_cfg in self.datasets:
                    self._run_group(
                        dna_cfg,
                        prot_cfg,
                        pooling,
                        active_mods,
                        dataset_cfg,
                        completed_keys,
                        global_metrics,
                        global_profiling,
                        experiments_subset=[experiment],
                        geom_feature_names=features,
                        ablation_label=label,
                        run_explainability=run_explainability,
                    )

        self._persist_global_outputs(global_metrics, global_profiling)
        df_metrics = self._read_csv_or_empty(self.metrics_csv_path)
        df_profiling = self._read_csv_or_empty(self.profiling_csv_path)
        return {
            "batch_run_dir": self.batch_run_dir,
            "metrics_path": self.metrics_csv_path,
            "profiling_path": self.profiling_csv_path,
            "num_rows_metrics": int(len(df_metrics)),
            "num_rows_profiling": int(len(df_profiling)),
        }

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
                    (None, None, ["bio"]),
                    (best_dna, best_prot, ["geom"]),
                    (best_dna, None, ["dna"]),
                    (None, best_prot, ["prot"]),
                    (best_dna, None, ["dna", "bio"]),
                    (None, best_prot, ["prot", "bio"]),
                    (best_dna, best_prot, ["dna", "geom"]),
                    (best_dna, best_prot, ["prot", "geom"]),
                    (best_dna, best_prot, ["dna", "bio", "geom"]),
                    (best_dna, best_prot, ["prot", "bio", "geom"]),
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

                # Stage 3: SpliceAI is a single nine-feature group; geometry
                # remains extracted but is deliberately excluded from these runs.
                stage3_cfgs = []
                for dna_cfg in self.dna_models:
                    stage3_cfgs.append((dna_cfg, None, ["dna", "spliceai"]))
                    stage3_cfgs.append((dna_cfg, None, ["dna", "bio_core", "spliceai"]))

                for prot_cfg in self.prot_models:
                    stage3_cfgs.append((None, prot_cfg, ["prot", "spliceai"]))
                    stage3_cfgs.append((None, prot_cfg, ["prot", "bio_core", "spliceai"]))

                for dna_cfg in self.dna_models:
                    for prot_cfg in self.prot_models:
                        stage3_cfgs.append((dna_cfg, prot_cfg, ["dna", "prot", "spliceai"]))
                        stage3_cfgs.append((dna_cfg, prot_cfg, ["dna", "prot", "bio_core", "spliceai"]))

                esm1v_cfg = next((m for m in self.prot_models if m["name"] == "esm1v_650m"), None)
                if esm1v_cfg is None:
                    raise ValueError(
                        "Stage 3 can only run after esm1v_650m is added to MODELS_SPACE."
                    )
                for dna_cfg in self.dna_models:
                    # This no-SpliceAI baseline has already been run for the
                    # older protein models, so it is intentionally ESM-1v-only.
                    stage3_cfgs.append((dna_cfg, esm1v_cfg, ["dna", "prot", "bio_core"]))

                # bio_core + spliceai has no foundation-model or pooling input.
                # Run it once, under the canonical center label, rather than
                # duplicating the same experiment for each pooling iteration.
                if pooling == "center":
                    for exp in self.experiments:
                        for ds_cfg in group_ds_cfgs:
                            self._run_group(
                                None,
                                None,
                                "center",
                                ["bio_core", "spliceai"],
                                ds_cfg,
                                completed_keys,
                                global_metrics,
                                global_profiling,
                                experiments_subset=[exp],
                            )

                for dna_cfg, prot_cfg, active_mods in stage3_cfgs:
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

        df_metrics = self._read_csv_or_empty(self.metrics_csv_path)
        df_profiling = self._read_csv_or_empty(self.profiling_csv_path)

        return {
            "batch_run_dir": self.batch_run_dir,
            "metrics_path": self.metrics_csv_path,
            "profiling_path": self.profiling_csv_path,
            "num_rows_metrics": int(len(df_metrics)),
            "num_rows_profiling": int(len(df_profiling)),
        }

    def run_sequence_ablation(
        self,
        sequence_pairs: list[tuple[str, str]],
        resume: bool = True,
        experiment_name: str = "Pure_XGBoost_Concat",
        pooling: str = "mean",
        active_mods: list[str] | None = None,
    ):
        """Run a fixed-model sequence-length matrix.

        ``sequence_pairs`` contains model names whose embedding files already
        encode the DNA/protein sequence lengths.  This keeps the matrix at
        exactly one experiment per pair instead of accidentally expanding into
        the full model/pooling/architecture search.
        """
        set_seed(42)
        if pooling not in self.pooling_strategies:
            raise ValueError(f"Pooling khong co trong pipeline: {pooling}")

        active_mods = active_mods or ["dna", "prot", "bio_core", "spliceai"]
        experiment = next((e for e in self.experiments if e["name"] == experiment_name), None)
        if experiment is None:
            raise ValueError(f"Khong tim thay experiment: {experiment_name}")
        if "geom" in active_mods:
            raise ValueError("Sequence ablation nay phai chay khong co geometry.")

        model_by_name = {model["name"]: model for model in self.models_space}
        for dna_name, prot_name in sequence_pairs:
            if dna_name not in model_by_name or model_by_name[dna_name]["seq_type"] != "dna":
                raise ValueError(f"DNA sequence model khong hop le: {dna_name}")
            if prot_name not in model_by_name or model_by_name[prot_name]["seq_type"] != "protein":
                raise ValueError(f"Protein sequence model khong hop le: {prot_name}")

        if resume:
            global_metrics, global_profiling, completed_by_pooling = self._load_resume_state()
        else:
            global_metrics, global_profiling, completed_by_pooling = [], [], {}
        self._persist_global_outputs(global_metrics, global_profiling)

        dataset_groups = {}
        for ds_cfg in self.datasets:
            key = (ds_cfg["train"], ds_cfg["val"])
            dataset_groups.setdefault(key, []).append(ds_cfg)

        completed_keys = set(completed_by_pooling.get(pooling, set()))
        dna_lookup = model_by_name
        for dna_name, prot_name in sequence_pairs:
            print(f"\n[SEQUENCE ABLATION] DNA={dna_name} | PROT={prot_name}")
            for _, group_ds_cfgs in dataset_groups.items():
                for ds_cfg in group_ds_cfgs:
                    self._run_group(
                        dna_lookup[dna_name],
                        dna_lookup[prot_name],
                        pooling,
                        active_mods,
                        ds_cfg,
                        completed_keys,
                        global_metrics,
                        global_profiling,
                        experiments_subset=[experiment],
                        geom_feature_names=None,
                        ablation_label="bio_core_dna_prot_spliceai",
                        run_explainability=False,
                    )

        self._persist_global_outputs(global_metrics, global_profiling)
        df_metrics = self._read_csv_or_empty(self.metrics_csv_path)
        df_profiling = self._read_csv_or_empty(self.profiling_csv_path)
        return {
            "batch_run_dir": self.batch_run_dir,
            "metrics_path": self.metrics_csv_path,
            "profiling_path": self.profiling_csv_path,
            "num_rows_metrics": int(len(df_metrics)),
            "num_rows_profiling": int(len(df_profiling)),
        }

    def run_factorial_sequence_ablation(
        self,
        sequence_pairs: list[tuple[str, str]],
        resume: bool = True,
        active_mods: list[str] | None = None,
    ):
        """Run every requested sequence pair, pooling and architecture.

        Embeddings are expected to be precomputed and named with their sequence
        length. Each train/validation configuration is fitted once, then its
        checkpoint is reused for all test datasets sharing that train/val pair.
        """
        set_seed(self.seed)
        active_mods = active_mods or ["dna", "prot", "bio_core", "spliceai"]
        if "geom" in {str(mod).lower() for mod in active_mods}:
            raise ValueError("Sequence-length ablation must not enable geometry.")
        if not {"dna", "prot"}.issubset({str(mod).lower() for mod in active_mods}):
            raise ValueError("Factorial sequence ablation requires both DNA and protein.")
        if not self.pooling_strategies or not self.experiments:
            raise ValueError("At least one pooling strategy and architecture are required.")

        model_by_name = {model["name"]: model for model in self.models_space}
        for dna_name, prot_name in sequence_pairs:
            if dna_name not in model_by_name or model_by_name[dna_name]["seq_type"] != "dna":
                raise ValueError(f"Invalid DNA sequence model: {dna_name}")
            if prot_name not in model_by_name or model_by_name[prot_name]["seq_type"] != "protein":
                raise ValueError(f"Invalid protein sequence model: {prot_name}")

        if resume:
            global_metrics, global_profiling, completed_by_pooling = self._load_resume_state()
        else:
            global_metrics, global_profiling, completed_by_pooling = [], [], {}
        self._persist_global_outputs(global_metrics, global_profiling)

        dataset_groups = {}
        for ds_cfg in self.datasets:
            dataset_groups.setdefault((ds_cfg["train"], ds_cfg["val"]), []).append(ds_cfg)

        for pooling in self.pooling_strategies:
            completed_keys = set(completed_by_pooling.get(pooling, set()))
            print(f"\n{'=' * 80}\n[FACTORIAL] POOLING={pooling} | pairs={len(sequence_pairs)} | architectures={len(self.experiments)}\n{'=' * 80}")
            for dna_name, prot_name in sequence_pairs:
                for group_ds_cfgs in dataset_groups.values():
                    for exp in self.experiments:
                        for ds_cfg in group_ds_cfgs:
                            self._run_group(
                                model_by_name[dna_name],
                                model_by_name[prot_name],
                                pooling,
                                active_mods,
                                ds_cfg,
                                completed_keys,
                                global_metrics,
                                global_profiling,
                                experiments_subset=[exp],
                                geom_feature_names=None,
                                ablation_label="bio_core_dna_prot_spliceai",
                                run_explainability=False,
                            )
            self._persist_global_outputs(global_metrics, global_profiling)

        df_metrics = self._read_csv_or_empty(self.metrics_csv_path)
        df_profiling = self._read_csv_or_empty(self.profiling_csv_path)
        return {
            "batch_run_dir": self.batch_run_dir,
            "metrics_path": self.metrics_csv_path,
            "profiling_path": self.profiling_csv_path,
            "num_rows_metrics": int(len(df_metrics)),
            "num_rows_profiling": int(len(df_profiling)),
        }


    def run_posthoc_explainability(
        self,
        metrics_path: str | None = None,
        top_n: int = 10,
        selection_metric: str = "Val_MCC",
        dataset_names: list[str] | None = None,
        networks: list[str] | None = None,
    ) -> dict:
        """Explain validation-selected trained XGBoost/Hybrid configs without retraining."""
        if not (self.explainability.get("enable_shap", False) or self.explainability.get("enable_lime", False)):
            raise ValueError("Enable SHAP and/or LIME in pipeline.explainability before post-hoc analysis.")
        if top_n <= 0:
            raise ValueError("top_n must be a positive number of unique configurations.")

        metrics_path = metrics_path or self.metrics_csv_path
        metrics = pd.read_csv(metrics_path)
        config_columns = ["Pooling", "Ablation", "DNA_Model", "Prot_Model", "Network"]
        required = set(config_columns + [selection_metric])
        missing = sorted(required.difference(metrics.columns))
        if missing:
            raise ValueError(f"Metrics file missing columns required for selection: {missing}")

        xgb_networks = {
            exp["name"] for exp in self.experiments
            if exp.get("type") in {"hybrid", "xgboost_pure"}
        }
        eligible = metrics.loc[
            metrics["Ablation"].astype(str).eq("bio_core_dna_prot_spliceai")
            & metrics["Network"].astype(str).isin(networks or xgb_networks)
        ].copy()
        eligible[selection_metric] = pd.to_numeric(eligible[selection_metric], errors="coerce")
        eligible = eligible.dropna(subset=[selection_metric])
        if eligible.empty:
            raise ValueError("No trained XGBoost/Hybrid rows found for the requested feature config.")

        ranked = (
            eligible.sort_values(selection_metric, ascending=False)
            .drop_duplicates(config_columns, keep="first")
            .head(top_n)
        )
        selected_configs = ranked[config_columns].to_dict(orient="records")
        selected_datasets = list(dict.fromkeys(dataset_names or [cfg["name"] for cfg in self.datasets]))
        dataset_cfg_by_name = {cfg["name"]: cfg for cfg in self.datasets}
        unknown_datasets = sorted(set(selected_datasets).difference(dataset_cfg_by_name))
        if unknown_datasets:
            raise ValueError(f"Unknown dataset names requested: {unknown_datasets}")

        experiment_by_name = {exp["name"]: exp for exp in self.experiments}
        results = []
        for rank, selected in enumerate(selected_configs, start=1):
            metric_match = eligible
            for column in config_columns:
                metric_match = metric_match.loc[metric_match[column].eq(selected[column])]
            val_score = float(metric_match[selection_metric].iloc[0])
            dna_name, prot_name = selected["DNA_Model"], selected["Prot_Model"]
            pooling, network = selected["Pooling"], selected["Network"]
            if network not in experiment_by_name:
                raise ValueError(f"Network {network} is absent from configured experiments.")
            exp = experiment_by_name[network]
            dna_cfg = next((m for m in self.models_space if m["name"] == dna_name and m["seq_type"] == "dna"), None)
            prot_cfg = next((m for m in self.models_space if m["name"] == prot_name and m["seq_type"] == "protein"), None)
            if dna_cfg is None or prot_cfg is None:
                raise ValueError(f"Models not found in MODELS_SPACE: {dna_name}, {prot_name}")

            for dataset_name in selected_datasets:
                dataset_cfg = dataset_cfg_by_name[dataset_name]
                dna_dim = self._get_cached_dim(dna_name, dataset_cfg["train"], pooling)
                prot_dim = self._get_cached_dim(prot_name, dataset_cfg["train"], pooling)
                active_mods = ["dna", "prot", "bio_core", "spliceai"]
                train_loader = self._get_dataloader(
                    dataset_cfg["train"], pooling, dna_name, prot_name, active_mods,
                    True, None, dataset_cfg["train"],
                )
                val_loader = self._get_dataloader(
                    dataset_cfg["val"], pooling, dna_name, prot_name, active_mods,
                    False, None, dataset_cfg["train"],
                )
                test_loader = self._get_dataloader(
                    dataset_cfg["test"], pooling, dna_name, prot_name, active_mods,
                    False, None, dataset_cfg["train"],
                )
                bio_dim = len(train_loader.dataset.bio_cols) if train_loader.dataset.has_bio else 0
                model = MultiStrategyFusionModel(
                    dna_in_dim=dna_dim,
                    prot_in_dim=prot_dim,
                    active_modalities=active_mods,
                    bio_in_dim=bio_dim,
                    geom_in_dim=0,
                    fusion_strategy=exp["fusion"],
                ).to(self.device)

                cache_dir = self._get_train_cache_ckpt_dir(
                    dataset_cfg, pooling, "bio_core_dna_prot_spliceai",
                    dna_name, prot_name, network, None,
                )
                xgb_path = f"{cache_dir}/best_model_xgboost.json"
                if not os.path.exists(xgb_path):
                    raise FileNotFoundError(
                        f"Trained XGBoost checkpoint is missing for {network}: {xgb_path}"
                    )
                xgb_manager = XGBoostFusionManager(
                    random_state=self.seed, n_jobs=self.xgb_n_jobs,
                )
                xgb_manager.load_model(cache_dir, prefix="best_model")

                feature_space = "dna_protein_pca_plus_biological_features"
                if exp["type"] == "hybrid":
                    hybrid_path = f"{cache_dir}/hybrid_feature_extractor.pth"
                    if not os.path.exists(hybrid_path):
                        raise FileNotFoundError(f"Trained hybrid feature checkpoint is missing: {hybrid_path}")
                    model.load_state_dict(torch.load(hybrid_path, map_location=self.device, weights_only=True))
                    x_train, _, _, _, y_train, _ = _extract_features_for_ml(
                        model, train_loader, self.device, True
                    )
                    x_val, _, _, _, y_val, _ = _extract_features_for_ml(
                        model, val_loader, self.device, True
                    )
                    x_test, _, _, _, y_true, variant_ids = _extract_features_for_ml(
                        model, test_loader, self.device, True
                    )
                    y_val_probs, _, _ = xgb_manager.predict_hybrid(x_val)
                    decision_threshold, val_mcc_at_threshold = self._select_validation_threshold(y_val, y_val_probs)
                    y_probs, _, _ = xgb_manager.predict_hybrid(x_test)
                    y_preds = self._threshold_predictions(y_probs, decision_threshold)
                    del x_val, y_val_probs
                    feature_names = self._feature_names(x_test.shape[1], "fusion_latent")
                    feature_space = "learned_fusion_latent"
                else:
                    _, dna_train, prot_train, bg_train, y_train, _ = _extract_features_for_ml(
                        model, train_loader, self.device, False
                    )
                    _, dna_val, prot_val, bg_val, y_val, _ = _extract_features_for_ml(
                        model, val_loader, self.device, False
                    )
                    _, dna_test, prot_test, bg_test, y_true, variant_ids = _extract_features_for_ml(
                        model, test_loader, self.device, False
                    )
                    val_probs, _, _ = xgb_manager.predict_pure(dna_val, prot_val, bg_val)
                    decision_threshold, val_mcc_at_threshold = self._select_validation_threshold(y_val, val_probs)
                    dna_train_pca, prot_train_pca = xgb_manager.transform_pca(dna_train, prot_train)
                    x_train = xgb_manager._build_pure_features(dna_train_pca, prot_train_pca, bg_train)
                    dna_test_pca, prot_test_pca = xgb_manager.transform_pca(dna_test, prot_test)
                    x_test = xgb_manager._build_pure_features(dna_test_pca, prot_test_pca, bg_test)
                    y_probs, _, _ = xgb_manager.predict_pure(
                        dna_test, prot_test, bg_test, y_true
                    )
                    y_preds = self._threshold_predictions(y_probs, decision_threshold)
                    feature_names = self._build_pure_feature_names(
                        dna_dim=0 if dna_test_pca is None else dna_test_pca.shape[1],
                        prot_dim=0 if prot_test_pca is None else prot_test_pca.shape[1],
                        bg_dim=0 if bg_test is None else bg_test.shape[1],
                        dataset=test_loader.dataset,
                    )
                    del dna_train, prot_train, bg_train, dna_val, prot_val, bg_val, dna_test, prot_test, bg_test
                    del val_probs, dna_train_pca, prot_train_pca, dna_test_pca, prot_test_pca

                exp_name = (
                    f"{dataset_cfg['train']}__{dataset_cfg['val']}_{pooling}_"
                    f"bio_core_dna_prot_spliceai_{dna_name}_{prot_name}_{network}"
                )
                test_out_dir = f"{self.batch_run_dir}/{exp_name}/tests/{dataset_name}"
                run_metadata = {
                    "selection_rank": rank,
                    "selection_metric": selection_metric,
                    "selection_value": val_score,
                    "dataset": dataset_name,
                    "network": network,
                    "pooling": pooling,
                    "dna_model": dna_name,
                    "protein_model": prot_name,
                    "feature_space": feature_space,
                    "selected_by_validation_only": True,
                    "decision_threshold": float(decision_threshold),
                    "threshold_selection_metric": "validation MCC",
                }
                self._run_xgb_explainability(
                    exp_dir=test_out_dir,
                    xgb_model=xgb_manager.model,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_true=y_true,
                    y_probs=y_probs,
                    y_preds=y_preds,
                    variant_ids=variant_ids,
                    feature_prefix="fusion_latent" if exp["type"] == "hybrid" else "xgb_pure",
                    feature_names=feature_names,
                    run_metadata=run_metadata,
                )
                results.append({
                    **selected,
                    "Dataset": dataset_name,
                    "selection_rank": rank,
                    selection_metric: val_score,
                    "Decision_Threshold": float(decision_threshold),
                    "Validation_MCC_at_Threshold": float(val_mcc_at_threshold),
                    "explainability_dir": f"{test_out_dir}/explainability",
                    "explained_test_samples": len(variant_ids),
                })
                del model, xgb_manager, train_loader, val_loader, test_loader, x_train, x_test
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        summary_path = f"{self.batch_run_dir}/posthoc_explainability_summary.csv"
        pd.DataFrame(results).to_csv(summary_path, index=False)
        return {
            "summary_path": summary_path,
            "num_selected_configs": len(selected_configs),
            "num_explained_dataset_runs": len(results),
            "results": results,
        }
