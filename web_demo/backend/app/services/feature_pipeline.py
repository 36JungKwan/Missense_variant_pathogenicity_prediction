"""Trích xuất đặc trưng 3 nhánh cho một biến thể: embedding DNA/Protein (center pooling),
11 bio core + 9 SpliceAI, 8 geometry V2; và chuẩn hoá bằng scaler .pkl của nhóm."""
from __future__ import annotations

import contextlib
import hashlib
import io
import math
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from app.core.config import settings
from app.core.constants import BIO_CORE_COLUMNS, GEOM_BRANCH_COLUMNS, SPLICEAI_COLUMNS, VariantKey
from app.core.errors import FeatureArtifactMissingError, InvalidVariantError
from app.ml_core.geometry_v2 import compute_online_geometry
from app.ml_core.normalizer import GEOM_V2_COLUMNS, AdvancedFeatureNormalizer
from app.ml_core.spliceai_lookup import IndexedSpliceAILookup
from app.services import sequence_service
from app.services.checkpoint_registry import resolve_device
from app.services.vep_runner import VepRecord

BIO_SCALER_FILES = [
    "bio_mean_imputer.pkl", "bio_freq_scaler.pkl", "bio_phast_scaler.pkl",
    "bio_phylo_scaler.pkl", "bio_spliceai_fill_values.pkl", "bio_spliceai_scaler.pkl",
]
BRANCHES = {"dna": settings.DNA_CONFIG_ID, "prot": settings.PROT_CONFIG_ID}
# Đổi model/cửa sổ/ký tự đệm thì tag đổi -> đặc trưng và dự đoán đã lưu trong DB tự bị coi là cũ.
EXTRACTOR_TAG = "fx-" + hashlib.sha1(
    "|".join([
        settings.DNA_MODEL_ID, settings.PROT_MODEL_ID,
        str(settings.DNA_WINDOW), str(settings.PROT_WINDOW),
        settings.DNA_PAD_CHAR, settings.PROT_PAD_CHAR,
    ]).encode()
).hexdigest()[:12]


@dataclass
class VariantFeatures:
    variant_id: str
    dna_ref: np.ndarray
    dna_alt: np.ndarray
    prot_ref: np.ndarray
    prot_alt: np.ndarray
    raw: Dict[str, Optional[float]]  # 20 bio thô + {dna,prot}_{LLR,LVD_L2,LVD_Cosine,LID,LVD_Relative} thô
    warnings: List[str] = field(default_factory=list)


class FoundationModelHub:
    """Singleton nạp lười NT-v1 500M và ESM2 650M; giữ trên GPU giữa các request."""

    _SPECS = {
        "dna": (settings.DNA_MODEL_ID, "dna"),
        "prot": (settings.PROT_MODEL_ID, "protein"),
    }

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._extractors: Dict[str, Any] = {}

    @property
    def loaded(self) -> List[str]:
        return sorted(self._extractors)

    def _extractor(self, kind: str):
        with self._load_lock:
            if kind not in self._extractors:
                # Import lười: transformers chỉ được nạp khi thật sự cần suy luận.
                from app.ml_core.fm_extractor import FeatureExtractor
                from app.ml_core.models import BioModelManager

                device = resolve_device()
                model_id = self._SPECS[kind][0]
                try:
                    manager = BioModelManager(model_id, model_type="mlm", device=str(device))
                except Exception as exc:
                    setting = "DNA_MODEL_ID" if kind == "dna" else "PROT_MODEL_ID"
                    raise FeatureArtifactMissingError(
                        f"Cannot load foundation model '{model_id}' ({setting} in backend/.env): "
                        f"{str(exc).splitlines()[0]}",
                        status="foundation_model_unavailable",
                        model_id=model_id,
                    ) from exc
                if device.type == "cpu":
                    manager.model.float()  # half precision không chạy ổn định trên CPU
                self._extractors[kind] = FeatureExtractor(manager)
            return self._extractors[kind]

    def embed(self, kind: str, ref_seq: str, alt_seq: str, center: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Trả (E_ref, E_alt, LLR) với center pooling tại token chứa vị trí đột biến."""
        extractor = self._extractor(kind)
        batch = {"name": ["query"], "ref_seq": [ref_seq], "alt_seq": [alt_seq], "variant_char_pos": [center]}
        with self._run_lock, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = extractor._process_batch_with_oom_protection(batch, self._SPECS[kind][1])
        return (
            out["center_ref"][0].astype(np.float32),
            out["center_alt"][0].astype(np.float32),
            float(out["llr"][0]),
        )


fm_hub = FoundationModelHub()

_faiss_lock = threading.Lock()
_faiss_indexes: Dict[str, Any] = {}


def _faiss_index(config_id: str):
    with _faiss_lock:
        if config_id not in _faiss_indexes:
            path = settings.resolve_path(settings.FAISS_INDEX_DIR) / f"{config_id}.index"
            index = None
            if path.is_file():
                import faiss

                index = faiss.read_index(str(path))
            _faiss_indexes[config_id] = index
        return _faiss_indexes[config_id]


def _nan_to_none(value: float) -> Optional[float]:
    return None if value is None or not math.isfinite(value) else float(value)


def lookup_spliceai(variants: List[VariantKey]) -> Dict[str, Dict[str, float]]:
    """Tra 9 cột SpliceAI (4 DS, 4 DP, DS_max) từ VCF tabix; biến thể không có trong VCF -> NaN."""
    if not variants:
        return {}
    try:
        # Cửa sổ nhỏ: mỗi truy vấn chỉ giải nén các block BGZF quanh đúng vị trí cần tra.
        lookup = IndexedSpliceAILookup(settings.SPLICEAI_VCF_PATH, window_size=1024)
    except FileNotFoundError as exc:
        raise FeatureArtifactMissingError(str(exc)) from exc
    frame = pd.DataFrame(
        [{"CHROM": v.chrom, "POS": v.pos, "REF": v.ref, "ALT": v.alt} for v in variants]
    )
    enriched, _ = lookup.enrich(frame)
    return {
        v.variant_id: {c: float(enriched.iloc[i][c]) for c in SPLICEAI_COLUMNS}
        for i, v in enumerate(variants)
    }


def build_features(
    variant: VariantKey,
    vep: VepRecord,
    spliceai: Optional[Dict[str, float]] = None,
) -> VariantFeatures:
    """Chạy toàn bộ khâu trích xuất cho một biến thể missense đã được VEP chú thích."""
    if not (vep.ensp and vep.protein_position and vep.ref_aa and vep.alt_aa):
        raise InvalidVariantError(
            f"{variant.variant_id}: VEP did not return a single-residue protein change "
            f"(ENSP={vep.ensp}, position={vep.protein_position}); cannot build the protein window."
        )
    feature_warnings: List[str] = []

    ref_dna, alt_dna, dna_center = sequence_service.dna_windows(variant)
    ref_prot, alt_prot, prot_center = sequence_service.protein_windows(
        vep.ensp, vep.protein_position, vep.ref_aa, vep.alt_aa
    )

    dna_ref, dna_alt, dna_llr = fm_hub.embed("dna", ref_dna, alt_dna, dna_center)
    prot_ref, prot_alt, prot_llr = fm_hub.embed("prot", ref_prot, alt_prot, prot_center)

    raw: Dict[str, Optional[float]] = {c: _nan_to_none(vep.bio_core.get(c, float("nan"))) for c in BIO_CORE_COLUMNS}
    if spliceai is None:
        spliceai = lookup_spliceai([variant])[variant.variant_id]
    raw.update({c: _nan_to_none(spliceai.get(c, float("nan"))) for c in SPLICEAI_COLUMNS})
    if raw["SpliceAI_pred_DS_max"] is None:
        feature_warnings.append("spliceai_not_found: variant absent from the SpliceAI MANE VCF; train-median imputation applies")

    for prefix, e_ref, e_alt, llr in (("dna", dna_ref, dna_alt, dna_llr), ("prot", prot_ref, prot_alt, prot_llr)):
        index = _faiss_index(BRANCHES[prefix])
        if index is None:
            feature_warnings.append(
                f"lid_imputed_{prefix}: no FAISS index '{BRANCHES[prefix]}.index' in FAISS_INDEX_DIR; "
                "LID uses the train-median fill value"
            )
        geometry = compute_online_geometry(e_ref, e_alt, llr, index=index, k_neighbors=settings.GEOM_K_NEIGHBORS)
        raw.update({f"{prefix}_{name}": _nan_to_none(value) for name, value in geometry.items()})

    return VariantFeatures(
        variant_id=variant.variant_id,
        dna_ref=dna_ref, dna_alt=dna_alt, prot_ref=prot_ref, prot_alt=prot_alt,
        raw=raw, warnings=feature_warnings,
    )


def refresh_imputed_geometry(features: VariantFeatures) -> bool:
    """Tính lại geometry từ embedding đã lưu cho nhánh nào trước đây phải điền LID (thiếu FAISS index)
    mà nay index đã có. Không chạy lại VEP / Foundation Models. Trả True nếu có thay đổi."""
    changed = False
    embeddings = {"dna": (features.dna_ref, features.dna_alt), "prot": (features.prot_ref, features.prot_alt)}
    for prefix, (e_ref, e_alt) in embeddings.items():
        if features.raw.get(f"{prefix}_LID") is not None:
            continue
        index = _faiss_index(BRANCHES[prefix])
        if index is None:
            continue
        llr = features.raw.get(f"{prefix}_LLR")
        geometry = compute_online_geometry(
            e_ref, e_alt, float("nan") if llr is None else llr,
            index=index, k_neighbors=settings.GEOM_K_NEIGHBORS,
        )
        features.raw.update({f"{prefix}_{name}": _nan_to_none(value) for name, value in geometry.items()})
        features.raw[f"{prefix}_LLR"] = llr  # LLR đến từ Foundation Model, giữ nguyên giá trị đã lưu
        features.warnings = [w for w in features.warnings if not w.startswith(f"lid_imputed_{prefix}:")]
        changed = True
    return changed


def missing_bio_scalers() -> List[str]:
    directory = settings.resolve_path(settings.BIO_SCALER_DIR)
    return [name for name in BIO_SCALER_FILES if not (directory / name).is_file()]


def missing_geom_scalers() -> List[str]:
    directory = settings.resolve_path(settings.MODEL_ARTIFACTS_DIR)
    names = [f"{config_id}_geom_v2_scalers.pkl" for config_id in BRANCHES.values()]
    return [name for name in names if not (directory / name).is_file()]


def _value(raw: Dict[str, Optional[float]], key: str) -> float:
    value = raw.get(key)
    return float("nan") if value is None else float(value)


def normalize_tabular(
    raw: Dict[str, Optional[float]],
    bio_columns: List[str],
    geom_columns: List[str],
) -> np.ndarray:
    """Áp đúng transform của `AdvancedFeatureNormalizer` rồi trả vector theo thứ tự bio_columns + geom_columns."""
    normalizer = AdvancedFeatureNormalizer()
    values: Dict[str, float] = {}

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")  # InconsistentVersionWarning của sklearn khi unpickle

        if bio_columns:
            missing = missing_bio_scalers()
            if missing:
                directory = settings.resolve_path(settings.BIO_SCALER_DIR)
                raise FeatureArtifactMissingError(
                    "Bio features cannot be normalized: the scaler files fitted on the training set are missing "
                    f"in '{directory}': {', '.join(missing)}. The two *_geom_v2_scalers.pkl files only cover the "
                    "geometry columns. Copy the bio_*.pkl files written by normalizer.fit_transform_bio "
                    "(normalization artifacts of train3) into that folder.",
                    status="bio_scalers_missing",
                    missing_files=missing,
                    expected_dir=str(directory),
                )
            all_bio = BIO_CORE_COLUMNS + SPLICEAI_COLUMNS
            frame = pd.DataFrame([{c: _value(raw, c) for c in all_bio}])
            scaled = normalizer.transform_bio(frame, str(settings.resolve_path(settings.BIO_SCALER_DIR)))
            values.update({c: float(scaled.iloc[0][c]) for c in all_bio})

        if geom_columns:
            missing = missing_geom_scalers()
            if missing:
                raise FeatureArtifactMissingError(
                    f"Geometry scalers missing in '{settings.resolve_path(settings.MODEL_ARTIFACTS_DIR)}': "
                    f"{', '.join(missing)}",
                    status="geom_scalers_missing",
                    missing_files=missing,
                )
            artifacts_dir = str(settings.resolve_path(settings.MODEL_ARTIFACTS_DIR))
            for prefix, config_id in BRANCHES.items():
                frame = pd.DataFrame([{c: _value(raw, f"{prefix}_{c}") for c in GEOM_V2_COLUMNS}])
                scaled = normalizer.transform_geom_v2(frame, artifacts_dir, config_id)
                values.update({f"{prefix}_{c}": float(scaled.iloc[0][c]) for c in GEOM_BRANCH_COLUMNS})

    return np.asarray([values[c] for c in bio_columns + geom_columns], dtype=np.float32)


def artifact_status() -> Dict[str, Any]:
    faiss_dir = settings.resolve_path(settings.FAISS_INDEX_DIR)
    return {
        "bio_scalers_missing": missing_bio_scalers(),
        "geom_scalers_missing": missing_geom_scalers(),
        "faiss_indexes": {
            config_id: (faiss_dir / f"{config_id}.index").is_file() for config_id in BRANCHES.values()
        },
        "foundation_models_loaded": fm_hub.loaded,
        "torch_cuda_available": torch.cuda.is_available(),
    }
