"""Chuẩn hoá đặc trưng Bio + Geometry V2 bằng scaler .pkl fit trên tập train (train3).

Toán học giữ nguyên `AdvancedFeatureNormalizer.transform_bio` / `transform_geom_v2` của pipeline huấn
luyện; khác biệt duy nhất là các file .pkl được nạp MỘT lần rồi cache thay vì unpickle ở mỗi request.
"""
from __future__ import annotations

import threading
import warnings
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

GEOM_V2_COLUMNS = ["LLR", "LVD_L2", "LVD_Cosine", "LID", "LVD_Relative"]
GEOM_V2_LOG_COLUMNS = ["LVD_L2", "LVD_Cosine", "LID", "LVD_Relative"]

BIO_FREQ_COLS = ["AF", "gnomADe_AF"]
BIO_PHAST_COLS = ["phastCons100way_vertebrate", "phastCons470way_mammalian", "phastCons17way_primate"]
BIO_PHYLO_GERP_COLS = [
    "phyloP100way_vertebrate", "phyloP470way_mammalian", "phyloP17way_primate",
    "GERP++_RS", "GERP++_NR", "GERP_92_mammals",
]
BIO_SPLICEAI_COLS = [
    "SpliceAI_pred_DS_AG", "SpliceAI_pred_DS_AL", "SpliceAI_pred_DS_DG", "SpliceAI_pred_DS_DL", "SpliceAI_pred_DS_max",
    "SpliceAI_pred_DP_AG", "SpliceAI_pred_DP_AL", "SpliceAI_pred_DP_DG", "SpliceAI_pred_DP_DL",
]
BIO_MEAN_IMPUTE_COLS = BIO_PHAST_COLS + BIO_PHYLO_GERP_COLS

BIO_SCALER_FILES = {
    "mean_imputer": "bio_mean_imputer.pkl",
    "freq_scaler": "bio_freq_scaler.pkl",
    "phast_scaler": "bio_phast_scaler.pkl",
    "phylo_scaler": "bio_phylo_scaler.pkl",
    "spliceai_fill_values": "bio_spliceai_fill_values.pkl",
    "spliceai_scaler": "bio_spliceai_scaler.pkl",
}


def geom_scaler_filename(config_id: str) -> str:
    return f"{config_id}_geom_v2_scalers.pkl"


def _load(path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # InconsistentVersionWarning của sklearn khi unpickle
        return joblib.load(path)


def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


class FeatureNormalizer:
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = float(epsilon)
        self._lock = threading.Lock()
        self._bio: Dict[Path, Dict[str, Any]] = {}
        self._geom: Dict[Path, Dict[str, Any]] = {}

    def bio_artifacts(self, directory: Path) -> Dict[str, Any]:
        with self._lock:
            if directory not in self._bio:
                self._bio[directory] = {key: _load(directory / name) for key, name in BIO_SCALER_FILES.items()}
            return self._bio[directory]

    def geom_artifacts(self, directory: Path, config_id: str) -> Dict[str, Any]:
        path = directory / geom_scaler_filename(config_id)
        with self._lock:
            if path not in self._geom:
                self._geom[path] = _load(path)
            return self._geom[path]

    def transform_bio(self, df: pd.DataFrame, directory: Path) -> pd.DataFrame:
        art = self.bio_artifacts(directory)
        out = _coerce_numeric(df.copy(), BIO_FREQ_COLS + BIO_MEAN_IMPUTE_COLS + BIO_SPLICEAI_COLS)
        out[BIO_FREQ_COLS] = out[BIO_FREQ_COLS].fillna(0.0)
        out[BIO_FREQ_COLS] = art["freq_scaler"].transform(np.log10(out[BIO_FREQ_COLS] + self.epsilon))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[BIO_MEAN_IMPUTE_COLS] = art["mean_imputer"].transform(out[BIO_MEAN_IMPUTE_COLS])
        out[BIO_PHAST_COLS] = art["phast_scaler"].transform(out[BIO_PHAST_COLS])
        out[BIO_PHYLO_GERP_COLS] = art["phylo_scaler"].transform(out[BIO_PHYLO_GERP_COLS])
        out[BIO_SPLICEAI_COLS] = out[BIO_SPLICEAI_COLS].fillna(art["spliceai_fill_values"])
        out[BIO_SPLICEAI_COLS] = art["spliceai_scaler"].transform(out[BIO_SPLICEAI_COLS])
        return out

    def transform_geom_v2(self, df: pd.DataFrame, directory: Path, config_id: str) -> pd.DataFrame:
        art = self.geom_artifacts(directory, config_id)
        out = _coerce_numeric(df.copy(), GEOM_V2_COLUMNS)
        out[GEOM_V2_COLUMNS] = out[GEOM_V2_COLUMNS].fillna(art["fill_values"])
        for column in GEOM_V2_COLUMNS:
            values = out[[column]].to_numpy(dtype=np.float64)
            if column in GEOM_V2_LOG_COLUMNS:
                values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
            out[column] = art["scalers"][column].transform(values).ravel()
        return out
