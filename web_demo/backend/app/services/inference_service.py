"""Điều phối suy luận một biến thể: VEP -> gatekeeper -> đặc trưng -> checkpoint -> SHAP -> PostgreSQL."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.constants import (
    VariantKey,
    compute_ablation_mask,
    mask_to_modalities,
    normalize_clinvar_label,
    parse_variant_id,
)
from app.models.variant import InferenceFeature, Prediction, Variant
from app.schemas.common import ModelResults, RBNetScore, ShapItem, SotaScores
from app.schemas.single import InferenceMeta, SingleVariantRequest, SingleVariantResponse
from app.services import ml_engine
from app.services.checkpoint_registry import LoadedModel, checkpoint_registry
from app.services.feature_pipeline import (
    EXTRACTOR_TAG,
    VariantFeatures,
    build_features,
    lookup_spliceai,
    refresh_imputed_geometry,
)
from app.services.vep_filter import ensure_missense
from app.services.vep_runner import VepRecord, annotate_variants

SOTA_FALLBACK = 0.5  # Frontend yêu cầu số; dùng khi dbNSFP không có điểm (kèm cảnh báo trong meta)


@dataclass
class Prefetched:
    """Kết quả chú thích theo lô (Batch VCF) để không gọi VEP/SpliceAI lặp lại cho từng biến thể."""
    vep: Dict[str, VepRecord]
    spliceai: Dict[str, Dict[str, float]]


def resolve_request_mask(request: SingleVariantRequest) -> int:
    dna, prot = request.dna_enabled, request.protein_enabled
    bio, geo = request.bio_enabled, request.geo_enabled
    if request.use_dna is not None:
        dna = request.use_dna
    if request.use_prot is not None:
        prot = request.use_prot
    if request.use_biogeom is not None:
        bio = geo = request.use_biogeom
    return compute_ablation_mask(dna, prot, bio, geo)


def resolve_model(mask: int) -> LoadedModel:
    """400 nếu tắt hết / tổ hợp không hợp lệ, 409 `checkpoint_missing` nếu chưa có file .pth."""
    return checkpoint_registry.get(mask_to_modalities(mask))


def _prediction_tag(loaded: LoadedModel) -> str:
    """Khoá cache của bảng predictions: checkpoint + cấu hình trích xuất đặc trưng."""
    return f"{loaded.tag}|{EXTRACTOR_TAG}"[:128]


def _stored_features(db: Session, variant_id: str) -> Optional[InferenceFeature]:
    row = db.get(InferenceFeature, variant_id)
    return row if row is not None and row.extractor_tag == EXTRACTOR_TAG else None


def _refresh_stale_geometry(db: Session, row: InferenceFeature) -> None:
    """Tính lại LID từ embedding đã lưu khi dòng được lưu lúc chưa có FAISS index (LID = null) hoặc với
    k láng giềng khác GEOM_K_NEIGHBORS hiện tại; xoá các dự đoán cũ của biến thể (dựa trên LID cũ)."""
    features = _to_features(row)
    annotation = row.annotation or {}
    if annotation.get("lid_k") != settings.GEOM_K_NEIGHBORS:
        for prefix in ("dna", "prot"):
            features.raw[f"{prefix}_LID"] = None
    if not refresh_imputed_geometry(features):
        return
    row.raw_features = features.raw
    row.annotation = {**annotation, "warnings": features.warnings, "lid_k": settings.GEOM_K_NEIGHBORS}
    db.query(Prediction).filter(Prediction.variant_id == row.variant_id).delete(synchronize_session=False)
    db.commit()


def prefetch_annotations(db: Session, keys: Sequence[VariantKey]) -> Prefetched:
    """Chạy VEP một lần cho các biến thể chưa có đặc trưng trong DB, rồi tra SpliceAI cho các missense."""
    pending = [k for k in keys if _stored_features(db, k.variant_id) is None]
    vep = annotate_variants(pending)
    missense = [k for k in pending if vep[k.variant_id].is_missense and len(k.ref) == 1 and len(k.alt) == 1]
    return Prefetched(vep=vep, spliceai=lookup_spliceai(missense))


def _display_consequence(variant: Variant) -> str:
    if variant.protein_change and variant.protein_change not in variant.consequence:
        return f"{variant.consequence} ({variant.protein_change})"
    return variant.consequence


def _extract_and_store(
    db: Session, key: VariantKey, prefetched: Optional[Prefetched]
) -> Tuple[Variant, InferenceFeature]:
    variant_id = key.variant_id
    record = (prefetched.vep.get(variant_id) if prefetched else None) or annotate_variants([key])[variant_id]
    ensure_missense(record)
    spliceai = prefetched.spliceai.get(variant_id) if prefetched else None
    features = build_features(key, record, spliceai=spliceai)

    variant = db.get(Variant, variant_id) or Variant(variant_id=variant_id)
    variant.gene = record.gene[:32]
    variant.consequence = record.consequence[:128]
    variant.protein_change = (record.hgvsp or f"p.{record.ref_aa}{record.protein_position}{record.alt_aa}")[:64]
    variant.clinvar_ground_truth = normalize_clinvar_label(record.clinvar_raw)
    db.add(variant)
    db.flush()

    row = db.get(InferenceFeature, variant_id) or InferenceFeature(variant_id=variant_id)
    row.extractor_tag = EXTRACTOR_TAG
    row.emb_dna_ref, row.emb_dna_alt = features.dna_ref, features.dna_alt
    row.emb_pro_ref, row.emb_pro_alt = features.prot_ref, features.prot_alt
    row.raw_features = features.raw
    row.annotation = {
        "transcript": record.transcript,
        "ensp": record.ensp,
        "protein_position": record.protein_position,
        "hgvsp": record.hgvsp,
        "clinvar_raw": record.clinvar_raw,
        "alphamissense": record.alphamissense,
        "revel": record.revel,
        "warnings": features.warnings,
        "lid_k": settings.GEOM_K_NEIGHBORS,
    }
    db.add(row)
    # Commit ngay: nếu bước suy luận phía sau lỗi (thiếu scaler/checkpoint) thì không phải chạy lại VEP + FM.
    db.commit()
    return variant, row


def _to_features(row: InferenceFeature) -> VariantFeatures:
    return VariantFeatures(
        variant_id=row.variant_id,
        dna_ref=np.asarray(row.emb_dna_ref, dtype=np.float32),
        dna_alt=np.asarray(row.emb_dna_alt, dtype=np.float32),
        prot_ref=np.asarray(row.emb_pro_ref, dtype=np.float32),
        prot_alt=np.asarray(row.emb_pro_alt, dtype=np.float32),
        raw=dict(row.raw_features),
        warnings=list((row.annotation or {}).get("warnings", [])),
    )


def _response(
    variant: Variant, prediction: Prediction, loaded: LoadedModel, mask: int,
    is_cache_hit: bool, started: float, warnings: List[str],
) -> SingleVariantResponse:
    return SingleVariantResponse(
        variant_id=variant.variant_id,
        gene=variant.gene or "UNKNOWN",
        consequence=_display_consequence(variant),
        clinvar_ground_truth=normalize_clinvar_label(variant.clinvar_ground_truth),
        results=ModelResults(
            rb_net=RBNetScore(probability=prediction.rb_net_prob, label=prediction.classification)
        ),
        sota=SotaScores(
            alphamissense=SOTA_FALLBACK if prediction.sota_alphamissense is None else prediction.sota_alphamissense,
            revel=SOTA_FALLBACK if prediction.sota_revel is None else prediction.sota_revel,
        ),
        shap_analysis=[ShapItem(**item) for item in prediction.shap_values],
        meta=InferenceMeta(
            is_cache_hit=is_cache_hit,
            computation_time_ms=max(1, int((time.perf_counter() - started) * 1000)),
            ablation_mask=mask,
            active_modalities=list(loaded.modalities),
            checkpoint=loaded.path.name,
            architecture=loaded.architecture,
            decision_threshold=loaded.threshold,
            warnings=warnings,
        ),
    )


def run_single_variant_inference(
    db: Session,
    request: SingleVariantRequest,
    prefetched: Optional[Prefetched] = None,
) -> SingleVariantResponse:
    started = time.perf_counter()
    key = parse_variant_id(request.variant_id)
    variant_id = key.variant_id
    mask = resolve_request_mask(request)
    loaded = resolve_model(mask)

    feature_row = _stored_features(db, variant_id)
    if feature_row is not None:
        _refresh_stale_geometry(db, feature_row)
    variant = db.get(Variant, variant_id) if feature_row is not None else None
    warnings: List[str] = []

    def _sota_warnings(annotation: dict) -> List[str]:
        return [f"{name}_unavailable: dbNSFP has no score, {SOTA_FALLBACK} shown as placeholder"
                for name in ("alphamissense", "revel") if annotation.get(name) is None]

    # 1) Cache: chỉ nhận dòng do đúng checkpoint hiện tại sinh ra (bỏ qua dữ liệu seed/mock cũ).
    if request.use_cache and variant is not None:
        cached = (
            db.query(Prediction)
            .filter(
                Prediction.variant_id == variant_id,
                Prediction.ablation_mask == mask,
                Prediction.model_tag == _prediction_tag(loaded),
            )
            .first()
        )
        if cached is not None:
            annotation = feature_row.annotation or {}
            warnings = list(annotation.get("warnings", [])) + _sota_warnings(annotation)
            return _response(variant, cached, loaded, mask, True, started, warnings)

    # 2) Đặc trưng: đọc lại từ PostgreSQL nếu đã trích xuất, nếu không chạy VEP + Foundation Models.
    if feature_row is None or variant is None:
        variant, feature_row = _extract_and_store(db, key, prefetched)
    features = _to_features(feature_row)
    annotation = feature_row.annotation or {}
    warnings = list(features.warnings) + _sota_warnings(annotation)

    # 3) Suy luận Pure DL + Shapley.
    output = ml_engine.predict(loaded, features)

    # 4) Upsert bảng predictions.
    prediction = (
        db.query(Prediction)
        .filter(Prediction.variant_id == variant_id, Prediction.ablation_mask == mask)
        .first()
    ) or Prediction(variant_id=variant_id, ablation_mask=mask)
    prediction.rb_net_prob = output.probability
    prediction.classification = output.label
    prediction.sota_alphamissense = annotation.get("alphamissense")
    prediction.sota_revel = annotation.get("revel")
    prediction.shap_values = output.shap_items
    prediction.model_tag = _prediction_tag(loaded)
    db.add(prediction)
    db.commit()

    return _response(variant, prediction, loaded, mask, False, started, warnings)
