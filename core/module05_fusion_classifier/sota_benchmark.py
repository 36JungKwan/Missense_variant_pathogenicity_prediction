import os
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


SOTA_MODEL_COLUMNS_FULL: dict[str, dict[str, Any]] = {
    "SIFT": {"score": ["SIFT_score"], "rankscore": ["SIFT_converted_rankscore", "SIFT_rankscore"], "pred": ["SIFT_pred"]},
    "SIFT4G": {"score": ["SIFT4G_score"], "rankscore": ["SIFT4G_converted_rankscore", "SIFT4G_rankscore"], "pred": ["SIFT4G_pred"]},
    "Polyphen2_HDIV": {"score": ["Polyphen2_HDIV_score"], "rankscore": ["Polyphen2_HDIV_rankscore"], "pred": ["Polyphen2_HDIV_pred"]},
    "Polyphen2_HVAR": {"score": ["Polyphen2_HVAR_score"], "rankscore": ["Polyphen2_HVAR_rankscore"], "pred": ["Polyphen2_HVAR_pred"]},
    "MutationTaster": {"score": ["MutationTaster_score"], "rankscore": ["MutationTaster_rankscore"], "pred": ["MutationTaster_pred"]},
    "MetaSVM": {"score": ["MetaSVM_score"], "rankscore": ["MetaSVM_rankscore"], "pred": ["MetaSVM_pred"]},
    "MetaLR": {"score": ["MetaLR_score"], "rankscore": ["MetaLR_rankscore"], "pred": ["MetaLR_pred"]},
    "MetaRNN": {"score": ["MetaRNN_score"], "rankscore": ["MetaRNN_rankscore"], "pred": ["MetaRNN_pred"]},
    "M-CAP": {"score": ["M-CAP_score"], "rankscore": ["M-CAP_rankscore"], "pred": ["M-CAP_pred"]},
    "REVEL": {"score": ["REVEL_score"], "rankscore": ["REVEL_rankscore"], "pred": ["REVEL_pred"]},
    "MutPred2": {"score": ["MutPred2_score"], "rankscore": ["MutPred2_rankscore"], "pred": ["MutPred2_pred"]},
    "MVP": {"score": ["MVP_score"], "rankscore": ["MVP_rankscore"], "pred": ["MVP_pred"]},
    "gMVP": {"score": ["gMVP_score"], "rankscore": ["gMVP_rankscore"], "pred": ["gMVP_pred"]},
    "MisFit_D": {"score": ["MisFit_D_score"], "rankscore": ["MisFit_D_rankscore"], "pred": ["MisFit_D_pred_lenient", "MisFit_D_pred"]},
    "MPC": {"score": ["MPC_score"], "rankscore": ["MPC_rankscore"], "pred": ["MPC_pred"]},
    "PrimateAI": {"score": ["PrimateAI_score"], "rankscore": ["PrimateAI_rankscore"], "pred": ["PrimateAI_pred"]},
    "BayesDel_addAF": {"score": ["BayesDel_addAF_score"], "rankscore": ["BayesDel_addAF_rankscore"], "pred": ["BayesDel_addAF_pred"]},
    "BayesDel_noAF": {"score": ["BayesDel_noAF_score"], "rankscore": ["BayesDel_noAF_rankscore"], "pred": ["BayesDel_noAF_pred"]},
    "ClinPred": {"score": ["ClinPred_score"], "rankscore": ["ClinPred_rankscore"], "pred": ["ClinPred_pred"]},
    "LIST-S2": {"score": ["LIST-S2_score"], "rankscore": ["LIST-S2_rankscore"], "pred": ["LIST-S2_pred"]},
    "VARITY_R": {"score": ["VARITY_R_score"], "rankscore": ["VARITY_R_rankscore"], "pred": ["VARITY_R_pred"]},
    "VARITY_ER": {"score": ["VARITY_ER_score"], "rankscore": ["VARITY_ER_rankscore"], "pred": ["VARITY_ER_pred"]},
    "AlphaMissense": {"score": ["AlphaMissense_score"], "rankscore": ["AlphaMissense_rankscore"], "pred": ["AlphaMissense_pred"]},
    "PHACTboost": {"score": ["PHACTboost_score"], "rankscore": ["PHACTboost_rankscore"], "pred": ["PHACTboost_pred"]},
    "MutFormer": {"score": ["MutFormer_score"], "rankscore": ["MutFormer_rankscore"], "pred": ["MutFormer_pred"]},
    "popEVE": {"score": ["popEVE_score"], "rankscore": ["popEVE_converted_rankscore", "popEVE_rankscore"], "pred": ["popEVE_pred"]},
    "CADD": {"score": ["CADD_raw", "CADD_phred"], "rankscore": ["CADD_raw_rankscore"], "pred": ["CADD_pred"]},
    "DANN": {"score": ["DANN_score"], "rankscore": ["DANN_rankscore"], "pred": ["DANN_pred"]},
}


SOTA_REQUIRED_COLUMNS = [
    "SIFT_score", "SIFT_converted_rankscore", "SIFT_pred",
    "SIFT4G_score", "SIFT4G_converted_rankscore", "SIFT4G_pred",
    "Polyphen2_HDIV_score", "Polyphen2_HDIV_rankscore", "Polyphen2_HDIV_pred",
    "Polyphen2_HVAR_score", "Polyphen2_HVAR_rankscore", "Polyphen2_HVAR_pred",
    "MutationTaster_score", "MutationTaster_rankscore", "MutationTaster_pred",
    "MetaSVM_score", "MetaSVM_rankscore", "MetaSVM_pred",
    "MetaLR_score", "MetaLR_rankscore", "MetaLR_pred",
    "MetaRNN_score", "MetaRNN_rankscore", "MetaRNN_pred",
    "M-CAP_score", "M-CAP_rankscore", "M-CAP_pred",
    "REVEL_score", "REVEL_rankscore", "REVEL_pred",
    "MutPred2_score", "MutPred2_rankscore", "MutPred2_pred",
    "MVP_score", "MVP_rankscore", "MVP_pred",
    "gMVP_score", "gMVP_rankscore", "gMVP_pred",
    "MisFit_D_score", "MisFit_D_rankscore", "MisFit_D_pred_lenient",
    "MPC_score", "MPC_rankscore", "MPC_pred",
    "PrimateAI_score", "PrimateAI_rankscore", "PrimateAI_pred",
    "BayesDel_addAF_score", "BayesDel_addAF_rankscore", "BayesDel_addAF_pred",
    "BayesDel_noAF_score", "BayesDel_noAF_rankscore", "BayesDel_noAF_pred",
    "ClinPred_score", "ClinPred_rankscore", "ClinPred_pred",
    "LIST-S2_score", "LIST-S2_rankscore", "LIST-S2_pred",
    "VARITY_R_score", "VARITY_R_rankscore", "VARITY_R_pred",
    "VARITY_ER_score", "VARITY_ER_rankscore", "VARITY_ER_pred",
    "AlphaMissense_score", "AlphaMissense_rankscore", "AlphaMissense_pred",
    "PHACTboost_score", "PHACTboost_rankscore", "PHACTboost_pred",
    "MutFormer_score", "MutFormer_rankscore", "MutFormer_pred",
    "popEVE_score", "popEVE_converted_rankscore", "popEVE_pred",
    "CADD_raw", "CADD_raw_rankscore", "CADD_phred", "CADD_pred",
    "DANN_score", "DANN_rankscore", "DANN_pred",
]


def _to_binary_pred(x: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(x):
        return (x.astype(float).to_numpy() > 0.5).astype(int)

    s = x.astype(str).str.strip().str.lower()
    pos_set = {"1", "true", "pathogenic", "damaging", "deleterious"}
    return s.isin(pos_set).astype(int).to_numpy()


def _safe_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    else:
        auroc = 0.0
        auprc = 0.0

    return {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "Specificity": round(float(spec), 4),
        "F1_Score": round(float(f1), 4),
        "MCC": round(float(mcc), 4),
        "AUROC": round(float(auroc), 4),
        "AUPRC": round(float(auprc), 4),
    }


def _normalize_label(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        arr = series.astype(float).to_numpy()
        return (arr > 0.5).astype(int)
    return _to_binary_pred(series)


def _resolve_label_column(df: pd.DataFrame, label_col: str | None, label_candidates: list[str] | None) -> str:
    if label_col is not None and label_col in df.columns:
        return label_col

    candidates = label_candidates or [
        "Pathogenicity_Label",
        "Label",
        "label",
        "target",
        "Target",
        "class",
        "Class",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        if re.search(r"(label|pathogenic|target|class)", c, re.IGNORECASE):
            return c

    raise ValueError("Khong tim thay cot nhan phu hop")


def _auto_discover_model_cols(columns: list[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    pattern = re.compile(r"(?P<model>.+?)(?:_|\.)(?P<kind>score|(?:converted_)?rankscore|pred(?:_.+)?)$", re.IGNORECASE)

    for c in columns:
        m = pattern.match(c)
        if not m:
            continue
        model = m.group("model")
        kind = m.group("kind").lower()
        if "rankscore" in kind:
            kind = "rankscore"
        elif kind.startswith("pred"):
            kind = "pred"
        out.setdefault(model, {})[kind] = c

    return out


def _first_existing(df: pd.DataFrame, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value in df.columns else None
    if isinstance(value, list):
        for v in value:
            if isinstance(v, str) and v in df.columns:
                return v
    return None


def _choose_prob_pred(df: pd.DataFrame, spec: dict[str, Any], threshold: float) -> tuple[np.ndarray, np.ndarray, str, str]:
    prob_col = _first_existing(df, spec.get("score")) or _first_existing(df, spec.get("rankscore"))
    if prob_col is None:
        raise ValueError("Model khong co cot score/rankscore de tinh AUROC/AUPRC")

    y_prob = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    pred_col = _first_existing(df, spec.get("pred"))
    if pred_col:
        y_pred = _to_binary_pred(df[pred_col])
        pred_source = pred_col
    else:
        y_pred = (y_prob > threshold).astype(int)
        pred_source = f"threshold@{threshold}"

    return y_prob, y_pred, pred_source, prob_col


def audit_required_sota_columns(
    test_files: dict[str, str],
    required_columns: list[str],
    output_dir: str,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    missing_count = 0
    null_issue_count = 0

    for dataset_name, file_path in test_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Khong tim thay test file: {file_path}")

        df = pd.read_parquet(file_path)
        for col in required_columns:
            exists = col in df.columns
            null_count = int(df[col].isnull().sum()) if exists else -1
            missing = not exists
            has_null = exists and null_count > 0

            if missing:
                missing_count += 1
            if has_null:
                null_issue_count += 1

            rows.append({
                "Dataset": dataset_name,
                "file_path": file_path,
                "column": col,
                "exists": bool(exists),
                "null_count": null_count,
            })

    df_audit = pd.DataFrame(rows)
    audit_path = f"{output_dir}/sota_required_columns_audit.csv"
    df_audit.to_csv(audit_path, index=False)

    return {
        "audit_path": audit_path,
        "num_rows": int(len(df_audit)),
        "missing_count": int(missing_count),
        "null_issue_count": int(null_issue_count),
    }


def evaluate_sota_from_test_files(
    test_files: dict[str, str],
    output_dir: str,
    label_col: str | None = "Pathogenicity_Label",
    label_candidates: list[str] | None = None,
    model_columns: dict[str, dict[str, Any]] | None = None,
    threshold: float = 0.5,
    required_columns: list[str] | None = None,
    enforce_required_columns: bool = False,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    metrics_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []

    audit_info = None
    if required_columns is not None:
        audit_info = audit_required_sota_columns(
            test_files=test_files,
            required_columns=required_columns,
            output_dir=output_dir,
        )
        if enforce_required_columns and (audit_info["missing_count"] > 0 or audit_info["null_issue_count"] > 0):
            raise ValueError(
                "SOTA benchmark bi chan do thieu cot hoac co null trong required_columns. "
                f"Xem audit tai: {audit_info['audit_path']}"
            )

    for dataset_name, file_path in test_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Khong tim thay test file: {file_path}")

        df = pd.read_parquet(file_path)
        resolved_label_col = _resolve_label_column(df, label_col, label_candidates)
        y_true = _normalize_label(df[resolved_label_col])
        discovered = _auto_discover_model_cols(list(df.columns))

        specs = model_columns or discovered
        if len(specs) == 0:
            raise ValueError(
                f"[{dataset_name}] Khong tim thay cot theo mau *_score/*_rankscore/*_pred. "
                "Can truyen model_columns thu cong."
            )

        for model_name, spec in specs.items():
            # Với model_columns thủ công, skip model không có cột trong file hiện tại.
            if model_columns is not None:
                any_exists = any(_first_existing(df, v) is not None for v in spec.values())
                if not any_exists:
                    continue

            try:
                y_prob, y_pred, pred_source, prob_col = _choose_prob_pred(df, spec, threshold)
            except Exception:
                continue

            m = _safe_binary_metrics(y_true, y_prob, y_pred)
            metrics_rows.append({
                "Dataset": dataset_name,
                "Network": f"SOTA_{model_name}",
                "Ablation": "sota",
                "Pooling": "N/A",
                "DNA_Model": "N/A",
                "Prot_Model": "N/A",
                **m,
            })
            mapping_rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "file_path": file_path,
                "score_col": spec.get("score"),
                "rankscore_col": spec.get("rankscore"),
                "pred_col": spec.get("pred"),
                "resolved_prob_col": prob_col,
                "pred_source": pred_source,
                "label_col": resolved_label_col,
            })

    df_metrics = pd.DataFrame(metrics_rows)
    df_mapping = pd.DataFrame(mapping_rows)

    metrics_path = f"{output_dir}/sota_metrics.csv"
    mapping_path = f"{output_dir}/sota_column_mapping.csv"
    df_metrics.to_csv(metrics_path, index=False)
    df_mapping.to_csv(mapping_path, index=False)

    return {
        "metrics_path": metrics_path,
        "mapping_path": mapping_path,
        "num_rows": int(len(df_metrics)),
        "num_models": int(df_metrics["Network"].nunique()) if len(df_metrics) > 0 else 0,
        "audit": audit_info,
    }
