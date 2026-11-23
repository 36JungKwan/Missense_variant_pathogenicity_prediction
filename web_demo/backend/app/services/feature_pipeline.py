"""Trích xuất đặc trưng 3 nhánh: embedding DNA/Protein (center pooling), 11 bio core + 9 SpliceAI,
8 geometry V2; và chuẩn hoá bằng scaler .pkl của nhóm.

Các bước được tách nhỏ để:
  * luồng đơn lẻ chạy NT-DNA + SpliceAI song song với tiến trình VEP (chỉ cần toạ độ),
  * luồng Batch VCF gộp nhiều biến thể vào một forward pass của Foundation Model.
"""
from __future__ import annotations

import hashlib
import logging
import math
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from app.core.config import settings
from app.core.constants import BIO_CORE_COLUMNS, GEOM_BRANCH_COLUMNS, SPLICEAI_COLUMNS, VariantKey
from app.core.errors import FeatureArtifactMissingError, FoundationModelUnavailableError, InvalidVariantError, PipelineError
from app.core.log import timed_stage
from app.ml_core.fm_extractor import EmbeddingBatch
from app.ml_core.geometry_v2 import compute_online_geometry
from app.ml_core.normalizer import BIO_SCALER_FILES, GEOM_V2_COLUMNS, FeatureNormalizer, geom_scaler_filename
from app.ml_core.spliceai_lookup import IndexedSpliceAILookup
from app.services import sequence_service
from app.services.checkpoint_registry import resolve_device
from app.services.vep_runner import VepRecord

logger = logging.getLogger("app.services.feature_pipeline")

BRANCHES = {"dna": settings.DNA_CONFIG_ID, "prot": settings.PROT_CONFIG_ID}
FM_DEVICE = resolve_device()
if settings.FM_CUDA_DTYPE not in ("fp16", "fp32"):
    raise ValueError(f"FM_CUDA_DTYPE must be 'fp16' or 'fp32', got {settings.FM_CUDA_DTYPE!r}")
FM_DTYPE = settings.FM_CUDA_DTYPE if FM_DEVICE.type == "cuda" else "fp32"
# Đổi model/cửa sổ/ký tự đệm/thiết bị/độ chính xác thì tag đổi -> đặc trưng và dự đoán cũ trong DB tự bị coi là cũ.
EXTRACTOR_TAG = "fx-" + hashlib.sha1(
    "|".join([
        settings.DNA_MODEL_ID, settings.PROT_MODEL_ID,
        str(settings.DNA_WINDOW), str(settings.PROT_WINDOW),
        settings.DNA_PAD_CHAR, settings.PROT_PAD_CHAR,
        FM_DEVICE.type, FM_DTYPE,
    ]).encode()
).hexdigest()[:12]

Embedding = Tuple[np.ndarray, np.ndarray, float]  # (E_ref, E_alt, LLR)
SeqInput = Tuple[str, str, int]  # (ref_seq, alt_seq, center_index)


@dataclass
class VariantFeatures:
    variant_id: str
    dna_ref: np.ndarray
    dna_alt: np.ndarray
    prot_ref: np.ndarray
    prot_alt: np.ndarray
    raw: Dict[str, Optional[float]]  # 20 bio thô + {dna,prot}_{LLR,LVD_L2,LVD_Cosine,LID,LVD_Relative} thô
    warnings: List[str] = field(default_factory=list)


# ============================================================================================
# Foundation Models
# ============================================================================================
class FoundationModelHub:
    """Singleton giữ NT-v1 500M và ESM2 650M trên GPU (fp16) / CPU (fp32) giữa các request."""

    _SPECS = {"dna": (settings.DNA_MODEL_ID, "dna"), "prot": (settings.PROT_MODEL_ID, "protein")}

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._run_locks = {kind: threading.Lock() for kind in self._SPECS}
        self._extractors: Dict[str, Any] = {}

    @property
    def loaded(self) -> List[str]:
        return sorted(self._extractors)

    def _extractor(self, kind: str):
        with self._load_lock:
            if kind not in self._extractors:
                from app.ml_core.fm_extractor import FeatureExtractor  # import lười: transformers rất nặng
                from app.ml_core.models import BioModelManager

                model_id = self._SPECS[kind][0]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        manager = BioModelManager(model_id, device=str(FM_DEVICE), cuda_dtype=FM_DTYPE)
                except Exception as exc:
                    setting = "DNA_MODEL_ID" if kind == "dna" else "PROT_MODEL_ID"
                    raise FoundationModelUnavailableError(
                        f"Cannot load foundation model '{model_id}' ({setting} in backend/.env): "
                        f"{str(exc).splitlines()[0] if str(exc) else type(exc).__name__}",
                        model_id=model_id,
                    ) from exc
                self._extractors[kind] = FeatureExtractor(manager)
            return self._extractors[kind]

    def embed_many(self, kind: str, inputs: Sequence[SeqInput]) -> List[Embedding]:
        """Embedding cho nhiều biến thể; gộp tối đa FM_BATCH_VARIANTS biến thể (x3 chuỗi) mỗi forward."""
        if not inputs:
            return []
        extractor = self._extractor(kind)
        step = max(1, settings.FM_BATCH_VARIANTS)
        parts: List[EmbeddingBatch] = []
        with timed_stage(f"fm_{kind}"), self._run_locks[kind], warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for start in range(0, len(inputs), step):
                chunk = inputs[start:start + step]
                parts.append(extractor.embed(
                    [c[0] for c in chunk], [c[1] for c in chunk], [c[2] for c in chunk], self._SPECS[kind][1]
                ))
        batch = EmbeddingBatch.concat(parts)
        return [(batch.center_ref[i], batch.center_alt[i], float(batch.llr[i])) for i in range(len(inputs))]

    def embed(self, kind: str, ref_seq: str, alt_seq: str, center: int) -> Embedding:
        return self.embed_many(kind, [(ref_seq, alt_seq, center)])[0]

    def load_all(self, warm_up: bool = True) -> None:
        for kind in self._SPECS:
            self._extractor(kind)
        if warm_up:  # khởi tạo cuBLAS/cuDNN + cấp phát bộ nhớ trước request đầu
            dna = ("ACGT" * 251)[: settings.DNA_WINDOW]
            prot = ("M" + "ACDEFGHIKLMNPQRSTVWY" * 26)[: settings.PROT_WINDOW]
            for kind, seq, alt_char in (("dna", dna, "T"), ("prot", prot, "W")):
                center = len(seq) // 2
                alt_char = alt_char if seq[center] != alt_char else "G"
                self.embed(kind, seq, seq[:center] + alt_char + seq[center + 1:], center)


fm_hub = FoundationModelHub()

# ============================================================================================
# Tài nguyên dùng chung (nạp một lần)
# ============================================================================================
_resource_lock = threading.Lock()
_faiss_indexes: Dict[str, Any] = {}
_spliceai: Optional[IndexedSpliceAILookup] = None
normalizer = FeatureNormalizer()


def _faiss_index(config_id: str):
    with _resource_lock:
        if config_id not in _faiss_indexes:
            path = settings.resolve_path(settings.FAISS_INDEX_DIR) / f"{config_id}.index"
            index = None
            if path.is_file():
                import faiss

                index = faiss.read_index(str(path))
            _faiss_indexes[config_id] = index
        return _faiss_indexes[config_id]


def _spliceai_lookup() -> IndexedSpliceAILookup:
    global _spliceai
    with _resource_lock:
        if _spliceai is None:
            try:
                _spliceai = IndexedSpliceAILookup(settings.SPLICEAI_VCF_PATH, window_size=settings.SPLICEAI_WINDOW_SIZE)
            except FileNotFoundError as exc:
                raise FeatureArtifactMissingError(str(exc)) from exc
        return _spliceai


def _nan_to_none(value: Optional[float]) -> Optional[float]:
    return None if value is None or not math.isfinite(value) else float(value)


def lookup_spliceai(variants: List[VariantKey]) -> Dict[str, Dict[str, float]]:
    """9 cột SpliceAI (4 DS, 4 DP, DS_max) từ VCF tabix; biến thể không có trong VCF -> NaN."""
    if not variants:
        return {}
    lookup = _spliceai_lookup()
    with timed_stage("spliceai"):
        frame = pd.DataFrame([{"CHROM": v.chrom, "POS": v.pos, "REF": v.ref, "ALT": v.alt} for v in variants])
        enriched, _ = lookup.enrich(frame)
        values = enriched[SPLICEAI_COLUMNS].to_numpy(dtype=float)
    return {v.variant_id: dict(zip(SPLICEAI_COLUMNS, map(float, values[i]))) for i, v in enumerate(variants)}


# ============================================================================================
# Các bước trích xuất
# ============================================================================================
def dna_inputs(variant: VariantKey) -> SeqInput:
    return sequence_service.dna_windows(variant)


def protein_inputs(variant: VariantKey, vep: VepRecord) -> SeqInput:
    if not (vep.ensp and vep.protein_position and vep.ref_aa and vep.alt_aa):
        raise InvalidVariantError(
            f"{variant.variant_id}: VEP did not return a single-residue protein change "
            f"(ENSP={vep.ensp}, position={vep.protein_position}); cannot build the protein window."
        )
    return sequence_service.protein_windows(vep.ensp, vep.protein_position, vep.ref_aa, vep.alt_aa)


def _geometry(prefix: str, e_ref: np.ndarray, e_alt: np.ndarray, llr: float) -> Tuple[Dict[str, Optional[float]], Optional[str]]:
    index = _faiss_index(BRANCHES[prefix])
    warning = None
    if index is None:
        warning = (f"lid_imputed_{prefix}: no FAISS index '{BRANCHES[prefix]}.index' in FAISS_INDEX_DIR; "
                   "LID uses the train-median fill value")
    with timed_stage("geom"):
        geometry = compute_online_geometry(e_ref, e_alt, llr, index=index, k_neighbors=settings.GEOM_K_NEIGHBORS)
    return {f"{prefix}_{name}": _nan_to_none(value) for name, value in geometry.items()}, warning


def assemble_features(
    variant: VariantKey,
    vep: VepRecord,
    spliceai: Dict[str, float],
    dna: Embedding,
    prot: Embedding,
) -> VariantFeatures:
    raw: Dict[str, Optional[float]] = {c: _nan_to_none(vep.bio_core.get(c, float("nan"))) for c in BIO_CORE_COLUMNS}
    raw.update({c: _nan_to_none(spliceai.get(c, float("nan"))) for c in SPLICEAI_COLUMNS})
    feature_warnings: List[str] = []
    if raw["SpliceAI_pred_DS_max"] is None:
        feature_warnings.append(
            "spliceai_not_found: variant absent from the SpliceAI MANE VCF; train-median imputation applies")
    for prefix, (e_ref, e_alt, llr) in (("dna", dna), ("prot", prot)):
        values, warning = _geometry(prefix, e_ref, e_alt, llr)
        raw.update(values)
        if warning:
            feature_warnings.append(warning)
    return VariantFeatures(
        variant_id=variant.variant_id,
        dna_ref=dna[0], dna_alt=dna[1], prot_ref=prot[0], prot_alt=prot[1],
        raw=raw, warnings=feature_warnings,
    )


def build_features(
    variant: VariantKey,
    vep: VepRecord,
    spliceai: Optional[Dict[str, float]] = None,
    dna: Optional[Embedding] = None,
) -> VariantFeatures:
    """Toàn bộ khâu trích xuất cho một biến thể missense. `dna`/`spliceai` có thể được tính trước
    (song song với VEP) để không phải tính lại."""
    prot_input = protein_inputs(variant, vep)
    if dna is None:
        dna = fm_hub.embed("dna", *dna_inputs(variant))
    prot = fm_hub.embed("prot", *prot_input)
    if spliceai is None:
        spliceai = lookup_spliceai([variant])[variant.variant_id]
    return assemble_features(variant, vep, spliceai, dna, prot)


def build_features_batch(
    items: Sequence[Tuple[VariantKey, VepRecord, Dict[str, float]]],
) -> Dict[str, Union[VariantFeatures, PipelineError]]:
    """Trích xuất cho nhiều biến thể; lỗi riêng của từng biến thể (REF sai, thiếu ENSP...) được trả về
    thay vì ném ra, còn lỗi hạ tầng (thiếu model/scaler) vẫn ném ra để dừng cả lô."""
    results: Dict[str, Union[VariantFeatures, PipelineError]] = {}
    valid: List[Tuple[VariantKey, VepRecord, Dict[str, float], SeqInput, SeqInput]] = []
    for key, vep, spliceai in items:
        try:
            valid.append((key, vep, spliceai, dna_inputs(key), protein_inputs(key, vep)))
        except InvalidVariantError as exc:
            results[key.variant_id] = exc
    dna = fm_hub.embed_many("dna", [v[3] for v in valid])
    prot = fm_hub.embed_many("prot", [v[4] for v in valid])
    for i, (key, vep, spliceai, _, _) in enumerate(valid):
        results[key.variant_id] = assemble_features(key, vep, spliceai, dna[i], prot[i])
    return results


def refresh_imputed_geometry(features: VariantFeatures) -> bool:
    """Tính lại geometry từ embedding đã lưu cho nhánh nào trước đây phải điền LID (thiếu FAISS index)
    mà nay index đã có. Không chạy lại VEP / Foundation Models. Trả True nếu có thay đổi."""
    changed = False
    embeddings = {"dna": (features.dna_ref, features.dna_alt), "prot": (features.prot_ref, features.prot_alt)}
    for prefix, (e_ref, e_alt) in embeddings.items():
        if features.raw.get(f"{prefix}_LID") is not None or _faiss_index(BRANCHES[prefix]) is None:
            continue
        llr = features.raw.get(f"{prefix}_LLR")
        values, _ = _geometry(prefix, e_ref, e_alt, float("nan") if llr is None else llr)
        features.raw.update(values)
        features.raw[f"{prefix}_LLR"] = llr  # LLR đến từ Foundation Model, giữ nguyên giá trị đã lưu
        features.warnings = [w for w in features.warnings if not w.startswith(f"lid_imputed_{prefix}:")]
        changed = True
    return changed


# ============================================================================================
# Chuẩn hoá
# ============================================================================================
def missing_bio_scalers() -> List[str]:
    directory = settings.resolve_path(settings.BIO_SCALER_DIR)
    return [name for name in BIO_SCALER_FILES.values() if not (directory / name).is_file()]


def missing_geom_scalers() -> List[str]:
    directory = settings.resolve_path(settings.MODEL_ARTIFACTS_DIR)
    names = [geom_scaler_filename(config_id) for config_id in BRANCHES.values()]
    return [name for name in names if not (directory / name).is_file()]


def _value(raw: Dict[str, Optional[float]], key: str) -> float:
    value = raw.get(key)
    return float("nan") if value is None else float(value)


def normalize_tabular(raw: Dict[str, Optional[float]], bio_columns: List[str], geom_columns: List[str]) -> np.ndarray:
    """Áp đúng transform của pipeline huấn luyện rồi trả vector theo thứ tự bio_columns + geom_columns."""
    values: Dict[str, float] = {}
    with timed_stage("norm"):
        if bio_columns:
            missing = missing_bio_scalers()
            if missing:
                directory = settings.resolve_path(settings.BIO_SCALER_DIR)
                raise FeatureArtifactMissingError(
                    "Bio features cannot be normalized: the scaler files fitted on the training set are missing "
                    f"in '{directory}': {', '.join(missing)}. Copy the bio_*.pkl files written by "
                    "normalizer.fit_transform_bio (normalization artifacts of train3) into that folder.",
                    status="bio_scalers_missing", missing_files=missing, expected_dir=str(directory),
                )
            all_bio = BIO_CORE_COLUMNS + SPLICEAI_COLUMNS
            frame = pd.DataFrame([{c: _value(raw, c) for c in all_bio}])
            scaled = normalizer.transform_bio(frame, settings.resolve_path(settings.BIO_SCALER_DIR))
            values.update({c: float(scaled.iloc[0][c]) for c in all_bio})

        if geom_columns:
            missing = missing_geom_scalers()
            if missing:
                raise FeatureArtifactMissingError(
                    f"Geometry scalers missing in '{settings.resolve_path(settings.MODEL_ARTIFACTS_DIR)}': "
                    f"{', '.join(missing)}",
                    status="geom_scalers_missing", missing_files=missing,
                )
            artifacts_dir = settings.resolve_path(settings.MODEL_ARTIFACTS_DIR)
            for prefix, config_id in BRANCHES.items():
                frame = pd.DataFrame([{c: _value(raw, f"{prefix}_{c}") for c in GEOM_V2_COLUMNS}])
                scaled = normalizer.transform_geom_v2(frame, artifacts_dir, config_id)
                values.update({f"{prefix}_{c}": float(scaled.iloc[0][c]) for c in GEOM_BRANCH_COLUMNS})

    return np.asarray([values[c] for c in bio_columns + geom_columns], dtype=np.float32)


# ============================================================================================
# Preload & trạng thái
# ============================================================================================
def preload() -> None:
    """Nạp mọi tài nguyên của pipeline đặc trưng; gọi trong lifespan của FastAPI."""
    sequence_service.preload()
    for config_id in BRANCHES.values():
        _faiss_index(config_id)
    _spliceai_lookup().warm_up()
    if not missing_bio_scalers():
        normalizer.bio_artifacts(settings.resolve_path(settings.BIO_SCALER_DIR))
    if not missing_geom_scalers():
        for config_id in BRANCHES.values():
            normalizer.geom_artifacts(settings.resolve_path(settings.MODEL_ARTIFACTS_DIR), config_id)
    fm_hub.load_all()


def artifact_status() -> Dict[str, Any]:
    faiss_dir = settings.resolve_path(settings.FAISS_INDEX_DIR)
    return {
        "bio_scalers_missing": missing_bio_scalers(),
        "geom_scalers_missing": missing_geom_scalers(),
        "faiss_indexes": {config_id: (faiss_dir / f"{config_id}.index").is_file() for config_id in BRANCHES.values()},
        "foundation_models_loaded": fm_hub.loaded,
        "fm_device": FM_DEVICE.type,
        "fm_dtype": FM_DTYPE,
        "extractor_tag": EXTRACTOR_TAG,
        "torch_cuda_available": torch.cuda.is_available(),
    }
