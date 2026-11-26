"""Điều phối suy luận: VEP -> gatekeeper -> đặc trưng -> checkpoint -> Integrated Gradients -> PostgreSQL.
Attention map chỉ tính khi mở trang XAI (không lưu)."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.constants import (
    VariantKey,
    compute_ablation_mask,
    mask_to_modalities,
    normalize_clinvar_label,
    parse_variant_id,
)
from app.core.errors import InvalidVariantError, NonMissenseVariantError, PipelineError
from app.core.log import format_timings, run_in_context, stage_scope, timed_stage
from app.models.variant import InferenceFeature, Prediction, Variant
from app.schemas.common import AttentionMaps, IgAnalysis, ModelResults, SotaScores, TriMVPPScore
from app.schemas.single import InferenceMeta, SingleVariantRequest, SingleVariantResponse, XaiResponse
from app.ml_core.fm_extractor import TOKEN_DIM, WINDOW_TOKENS, TokenWindow
from app.services import ml_engine
from app.services.checkpoint_registry import LoadedModel, checkpoint_registry
from app.services.feature_pipeline import (
    EXTRACTOR_TAG,
    VariantFeatures,
    build_features,
    build_features_batch,
    dna_inputs,
    fm_hub,
    lookup_spliceai,
    refresh_imputed_geometry,
)
from app.services.vep_runner import VepRecord, annotate_variants, ensure_missense

logger = logging.getLogger("app.services.inference")

# VEP là tiến trình con (docker exec) nên chạy được song song với NT-DNA + SpliceAI trên luồng chính.
_vep_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vep")


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
    with timed_stage("db"):
        stored = {row.variant_id for row in db.query(InferenceFeature.variant_id).filter(
            InferenceFeature.variant_id.in_([k.variant_id for k in keys]),
            InferenceFeature.extractor_tag == EXTRACTOR_TAG,
        )}
    pending = [k for k in keys if k.variant_id not in stored]
    vep = annotate_variants(pending)
    missense = [k for k in pending if vep[k.variant_id].is_missense and len(k.ref) == 1 and len(k.alt) == 1]
    return Prefetched(vep=vep, spliceai=lookup_spliceai(missense))


def _display_consequence(variant: Variant) -> str:
    if variant.protein_change and variant.protein_change not in variant.consequence:
        return f"{variant.consequence} ({variant.protein_change})"
    return variant.consequence


def _upsert(db: Session, model, values: Dict, conflict_columns: Sequence[str]) -> None:
    """INSERT ... ON CONFLICT DO UPDATE nguyên tử của PostgreSQL: chạy lại (Live / use_cache=false) hay hai
    request đồng thời cho cùng biến thể đều ghi đè đúng một dòng, không bao giờ chèn bản ghi trùng."""
    stmt = pg_insert(model.__table__).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=list(conflict_columns),
        set_={name: stmt.excluded[name] for name in values if name not in conflict_columns},
    )
    db.execute(stmt)


def _store_features(db: Session, key: VariantKey, record: VepRecord, features: VariantFeatures) -> None:
    """Upsert variants (khoá variant_id) + inference_features (khoá variant_id); chưa commit."""
    variant_id = key.variant_id
    _upsert(db, Variant, {
        "variant_id": variant_id,
        "gene": record.gene[:32],
        "consequence": record.consequence[:128],
        "protein_change": (record.hgvsp or f"p.{record.ref_aa}{record.protein_position}{record.alt_aa}")[:64],
        "clinvar_ground_truth": normalize_clinvar_label(record.clinvar_raw),
    }, ["variant_id"])
    _upsert(db, InferenceFeature, {
        "variant_id": variant_id,
        "extractor_tag": EXTRACTOR_TAG,
        "emb_dna_ref": features.dna.center_ref.astype(np.float32),
        "emb_dna_alt": features.dna.center_alt.astype(np.float32),
        "emb_pro_ref": features.prot.center_ref.astype(np.float32),
        "emb_pro_alt": features.prot.center_alt.astype(np.float32),
        "tok_dna": features.dna.hidden.astype("<f2").tobytes(),
        "tok_pro": features.prot.hidden.astype("<f2").tobytes(),
        "tok_mask": np.stack([features.dna.mask, features.prot.mask]).astype(np.uint8).tobytes(),
        "raw_features": features.raw,
        "annotation": {
            "transcript": record.transcript,
            "ensp": record.ensp,
            "protein_position": record.protein_position,
            "hgvsp": record.hgvsp,
            "clinvar_raw": record.clinvar_raw,
            "alphamissense": record.alphamissense,
            "revel": record.revel,
            "warnings": features.warnings,
            "lid_k": settings.GEOM_K_NEIGHBORS,
        },
    }, ["variant_id"])


def _load_stored(db: Session, variant_id: str) -> Tuple[Variant, InferenceFeature]:
    # populate_existing: bỏ qua bản cũ trong identity map của session sau khi upsert bằng Core.
    variant = db.get(Variant, variant_id, populate_existing=True)
    row = db.get(InferenceFeature, variant_id, populate_existing=True)
    return variant, row


def _annotate_with_overlap(key: VariantKey) -> Tuple[VepRecord, Optional[TokenWindow], Optional[Dict[str, float]]]:
    """VEP (tiến trình con) chạy song song với NT-DNA + SpliceAI, vốn chỉ cần toạ độ biến thể."""
    vep_future = _vep_executor.submit(run_in_context(annotate_variants), [key])
    dna: Optional[TokenWindow] = None
    spliceai: Optional[Dict[str, float]] = None
    dna_error: Optional[InvalidVariantError] = None
    try:
        try:
            dna = fm_hub.embed("dna", *dna_inputs(key))
            spliceai = lookup_spliceai([key])[key.variant_id]
        except InvalidVariantError as exc:  # báo sau gatekeeper để giữ thứ tự lỗi như trước
            dna_error = exc
        record = vep_future.result()[key.variant_id]
    except BaseException:
        vep_future.cancel()
        raise
    ensure_missense(record)
    if dna_error is not None:
        raise dna_error
    return record, dna, spliceai


def _extract_and_store(db: Session, key: VariantKey, prefetched: Optional[Prefetched]) -> Tuple[Variant, InferenceFeature]:
    record = prefetched.vep.get(key.variant_id) if prefetched else None
    if record is not None:
        ensure_missense(record)
        features = build_features(key, record, spliceai=prefetched.spliceai.get(key.variant_id))
    else:
        record, dna, spliceai = _annotate_with_overlap(key)
        features = build_features(key, record, spliceai=spliceai, dna=dna)
    with timed_stage("db"):
        _store_features(db, key, record, features)
        # Commit ngay: nếu bước suy luận phía sau lỗi thì không phải chạy lại VEP + FM.
        db.commit()
        return _load_stored(db, key.variant_id)


def prepare_features_batch(db: Session, keys: Sequence[VariantKey], prefetched: Prefetched) -> Dict[str, PipelineError]:
    """Batch VCF: trích xuất đặc trưng cho cả nhóm biến thể với forward pass FM gộp, lưu vào DB.

    Trả {variant_id: lỗi riêng của biến thể} (non-missense, REF sai...) để luồng SSE báo `variant_skipped`;
    lỗi hạ tầng vẫn được ném ra."""
    started = time.perf_counter()
    failures: Dict[str, PipelineError] = {}
    items = []
    with stage_scope() as timings:
        with timed_stage("db"):
            stored = {row.variant_id for row in db.query(InferenceFeature.variant_id).filter(
                InferenceFeature.variant_id.in_([k.variant_id for k in keys]),
                InferenceFeature.extractor_tag == EXTRACTOR_TAG,
            )}
        for key in keys:
            if key.variant_id in stored:
                continue
            record = prefetched.vep.get(key.variant_id) or annotate_variants([key])[key.variant_id]
            try:
                ensure_missense(record)
            except NonMissenseVariantError as exc:
                failures[key.variant_id] = exc
                continue
            items.append((key, record, prefetched.spliceai.get(key.variant_id, {})))

        results = build_features_batch(items)
        with timed_stage("db"):
            for key, record, _ in items:
                result = results[key.variant_id]
                if isinstance(result, PipelineError):
                    failures[key.variant_id] = result
                else:
                    _store_features(db, key, record, result)
            db.commit()
    if items:
        logger.info("batch extraction variants=%d extracted=%d total=%dms %s", len(keys),
                    len(items) - sum(1 for k, _, _ in items if k.variant_id in failures),
                    (time.perf_counter() - started) * 1000, format_timings(timings))
    return failures


def _to_features(row: InferenceFeature) -> VariantFeatures:
    masks = np.frombuffer(row.tok_mask, dtype=np.uint8).reshape(2, 2, WINDOW_TOKENS).astype(bool)

    def window(blob: bytes, mask: np.ndarray) -> TokenWindow:
        return TokenWindow(np.frombuffer(blob, dtype="<f2").reshape(2, WINDOW_TOKENS, TOKEN_DIM), mask)

    return VariantFeatures(
        variant_id=row.variant_id,
        dna=window(row.tok_dna, masks[0]),
        prot=window(row.tok_pro, masks[1]),
        raw=dict(row.raw_features),
        warnings=list((row.annotation or {}).get("warnings", [])),
    )


def _sota_warnings(annotation: dict) -> List[str]:
    return [f"{name}_unavailable: dbNSFP has no score for this variant"
            for name in ("alphamissense", "revel") if annotation.get(name) is None]


def _response(
    variant: Variant, prediction: Prediction, loaded: LoadedModel, mask: int,
    is_cache_hit: bool, started: float, warnings: List[str], timings: Dict[str, int],
) -> SingleVariantResponse:
    return SingleVariantResponse(
        variant_id=variant.variant_id,
        gene=variant.gene or "UNKNOWN",
        consequence=_display_consequence(variant),
        clinvar_ground_truth=normalize_clinvar_label(variant.clinvar_ground_truth),
        results=ModelResults(trimvpp=TriMVPPScore(probability=prediction.trimvpp_prob, label=prediction.classification)),
        sota=SotaScores(alphamissense=prediction.sota_alphamissense, revel=prediction.sota_revel),
        ig_analysis=IgAnalysis(**prediction.ig_values),
        meta=InferenceMeta(
            is_cache_hit=is_cache_hit,
            computation_time_ms=max(1, int((time.perf_counter() - started) * 1000)),
            ablation_mask=mask,
            active_modalities=list(loaded.modalities),
            checkpoint=loaded.path.name,
            architecture=loaded.architecture,
            decision_threshold=loaded.threshold,
            test_mcc=loaded.test_mcc,
            warnings=warnings,
            timings_ms=dict(timings),
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
    cache_state = "miss"

    with stage_scope() as timings:
        try:
            with timed_stage("db"):
                feature_row = _stored_features(db, variant_id)
                if feature_row is not None:
                    _refresh_stale_geometry(db, feature_row)
                variant = db.get(Variant, variant_id) if feature_row is not None else None

            # 1) Cache: chỉ nhận dòng do đúng checkpoint + extractor hiện tại sinh ra.
            if request.use_cache and variant is not None:
                with timed_stage("db"):
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
                    cache_state = "hit"
                    annotation = feature_row.annotation or {}
                    warnings = list(annotation.get("warnings", [])) + _sota_warnings(annotation)
                    return _response(variant, cached, loaded, mask, True, started, warnings, timings)

            # 2) Đặc trưng: đọc lại từ PostgreSQL nếu đã trích xuất, nếu không chạy VEP + Foundation Models.
            if feature_row is None or variant is None:
                cache_state = "miss"
                variant, feature_row = _extract_and_store(db, key, prefetched)
            else:
                cache_state = "features"
            features = _to_features(feature_row)
            annotation = feature_row.annotation or {}
            warnings = list(features.warnings) + _sota_warnings(annotation)

            # 3) Suy luận TriMVPP + Integrated Gradients.
            output = ml_engine.predict(loaded, features)

            # 4) Upsert bảng predictions theo unique index ix_variant_mask (variant_id, ablation_mask):
            # mỗi tổ hợp chỉ giữ đúng một dòng, model_tag được ghi đè bằng checkpoint/extractor hiện tại.
            values = {
                "variant_id": variant_id,
                "ablation_mask": mask,
                "trimvpp_prob": output.probability,
                "classification": output.label,
                "sota_alphamissense": annotation.get("alphamissense"),
                "sota_revel": annotation.get("revel"),
                "ig_values": output.ig,
                "model_tag": _prediction_tag(loaded),
            }
            with timed_stage("db"):
                _upsert(db, Prediction, values, ["variant_id", "ablation_mask"])
                db.commit()

            return _response(variant, Prediction(**values), loaded, mask, False, started, warnings, timings)
        except Exception as exc:
            cache_state = f"error({getattr(exc, 'error_code', type(exc).__name__)})"
            raise
        finally:
            logger.info("inference variant=%s mask=%d cache=%s total=%dms %s", variant_id, mask, cache_state,
                        (time.perf_counter() - started) * 1000, format_timings(timings))


def run_xai(db: Session, request: SingleVariantRequest) -> XaiResponse:
    """Dự đoán + IG (đọc cache predictions hoặc tính mới), rồi tính attention map từ đặc trưng đã lưu."""
    result = run_single_variant_inference(db=db, request=request)
    loaded = resolve_model(result.meta.ablation_mask)
    attention = None
    if len(loaded.model.cross_layers) > 0:
        with stage_scope() as timings:
            with timed_stage("db"):
                row = _stored_features(db, result.variant_id)
            if row is None:
                raise PipelineError(f"Stored features for {result.variant_id} disappeared during the XAI request.")
            maps = ml_engine.attention(loaded, _to_features(row))
        attention = AttentionMaps(**maps)
        result.meta.timings_ms.update({k: v for k, v in timings.items() if k == "attn"})
    return XaiResponse(**result.model_dump(), attention=attention)
