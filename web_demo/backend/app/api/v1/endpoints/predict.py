import asyncio
import csv
import io
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.core.constants import VARIANT_ID_PATTERN, compute_ablation_mask, parse_variant_id
from app.core.errors import InvalidVariantError, NonMissenseVariantError, PipelineError
from app.db.session import get_db, SessionLocal
from app.schemas.single import SingleVariantRequest, SingleVariantResponse
from app.schemas.batch import BatchVariant, BatchStreamEvent
from app.services.inference_service import (
    prefetch_annotations,
    resolve_model,
    run_single_variant_inference,
)

router = APIRouter()


@router.post("/single", response_model=SingleVariantResponse)
def predict_single_variant(
    request: SingleVariantRequest,
    db: Session = Depends(get_db),
):
    # PipelineError (400/409/422/503) được handler toàn cục trong app.main chuyển thành JSON detail.
    return run_single_variant_inference(db=db, request=request)


def _parse_uploaded_variants(raw_bytes: bytes, filename: str = "") -> List[str]:
    """
    Hỗ trợ parse linh hoạt và khử trùng lặp (giữ nguyên thứ tự):
    - File .vcf chuẩn (#CHROM  POS  ID  REF  ALT ...)
    - File .csv hoặc .txt chứa danh sách chrN:POS:REF>ALT hoặc các cột CHROM, POS, REF, ALT
    """
    seen = set()
    variant_ids: List[str] = []

    def _add_unique(raw_candidate: str) -> None:
        if VARIANT_ID_PATTERN.match(raw_candidate.strip()):
            norm = parse_variant_id(raw_candidate).variant_id
            if norm not in seen:
                seen.add(norm)
                variant_ids.append(norm)

    text_content = raw_bytes.decode("utf-8", errors="ignore")
    lower_name = filename.lower()

    # Trường hợp file CSV
    if lower_name.endswith(".csv"):
        reader = csv.reader(io.StringIO(text_content))
        for row in reader:
            if not row:
                continue
            matched_in_cell = False
            for cell in row:
                val = cell.strip()
                if VARIANT_ID_PATTERN.match(val):
                    _add_unique(val)
                    matched_in_cell = True
                    break
            if not matched_in_cell and len(row) >= 4 and row[1].strip().isdigit():
                chrom = row[0].strip()
                pos = row[1].strip()
                ref = row[2].strip().upper()
                alt = row[3].strip().split(",")[0].strip().upper()
                _add_unique(f"{chrom}:{pos}:{ref}>{alt}")
        return variant_ids

    # Trường hợp file VCF hoặc TXT
    for line in text_content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        cols = stripped.split("\t") if "\t" in stripped else stripped.split()
        if len(cols) >= 5 and cols[1].isdigit():
            chrom = cols[0].strip()
            pos = cols[1].strip()
            ref = cols[3].strip().upper()
            alt = cols[4].split(",")[0].strip().upper()
            if alt != ".":
                _add_unique(f"{chrom}:{pos}:{ref}>{alt}")
            continue

        if VARIANT_ID_PATTERN.match(stripped):
            _add_unique(stripped)

    return variant_ids


def _get_demo_cohort_variants() -> List[str]:
    """Cohort mẫu (toạ độ GRCh38). Hậu quả thật do VEP quyết định, không còn tra bảng cứng."""
    return [
        "chr7:140753336:A>T",   # BRAF p.Val600Glu
        "chr17:41245466:G>A",   # toạ độ hg19 cũ của BRCA1 -> minh hoạ kiểm tra REF/gatekeeper
        "chr13:32906729:A>C",   # toạ độ hg19 cũ của BRCA2
        "chr17:7674220:C>T",    # TP53 p.Arg248Gln
        "chr17:7673802:C>T",    # TP53 p.Arg273His
        "chr12:25245350:C>T",   # KRAS p.Gly12Asp
        "chr17:7675088:C>T",    # TP53
        "chr3:178936091:G>A",   # toạ độ hg19 cũ của PIK3CA
        "chr10:89692904:C>T",   # toạ độ hg19 cũ của PTEN
        "chr1:115256529:T>C",   # toạ độ hg19 cũ của NRAS
    ]


def _create_batch_sse_response(
    variant_queue: List[str],
    stream_delay_ms: int,
    dna_enabled: bool,
    protein_enabled: bool,
    bio_enabled: bool,
    geo_enabled: bool,
) -> EventSourceResponse:
    # Kiểm tra tổ hợp + checkpoint TRƯỚC khi mở stream: tắt hết -> 400, thiếu .pth -> 409 checkpoint_missing.
    resolve_model(compute_ablation_mask(dna_enabled, protein_enabled, bio_enabled, geo_enabled))
    total_count = len(variant_queue)

    def _event(**kwargs) -> dict:
        return {"data": BatchStreamEvent(**kwargs).model_dump_json()}

    async def event_generator():
        db = SessionLocal()
        accepted_count = 0
        rejected_count = 0

        def _summary() -> dict:
            return {
                "total_processed": accepted_count + rejected_count,
                "missense_accepted": accepted_count,
                "non_missense_rejected": rejected_count,
            }

        try:
            if total_count == 0:
                yield _event(type="stream_completed", progress_percent=100, summary=_summary())
                return

            # Một lần chạy VEP + tra SpliceAI cho toàn bộ file (chạy trong thread để không chặn event loop).
            try:
                keys = [parse_variant_id(vid) for vid in variant_queue]
                prefetched = await asyncio.to_thread(prefetch_annotations, db, keys)
            except PipelineError as exc:
                db.rollback()
                yield _event(type="stream_error", progress_percent=0, reason=exc.message, error_code=exc.error_code)
                return

            for idx, vid in enumerate(variant_queue, start=1):
                progress = int((idx / total_count) * 100)
                req = SingleVariantRequest(
                    variant_id=vid,
                    dna_enabled=dna_enabled,
                    protein_enabled=protein_enabled,
                    bio_enabled=bio_enabled,
                    geo_enabled=geo_enabled,
                    use_cache=True,
                )
                try:
                    res = await asyncio.to_thread(run_single_variant_inference, db, req, prefetched)
                    accepted_count += 1
                    yield _event(
                        type="variant_stream",
                        progress_percent=progress,
                        variant=BatchVariant(
                            id=accepted_count,
                            variant_id=res.variant_id,
                            gene=res.gene,
                            consequence=res.consequence,
                            rb_net_probability=res.results.rb_net.probability,
                            classification=res.results.rb_net.label,
                            clinvar_ground_truth=res.clinvar_ground_truth,
                            sota_alphamissense=res.sota.alphamissense,
                            danger_rank=accepted_count,
                        ),
                    )
                except NonMissenseVariantError as exc:
                    db.rollback()
                    rejected_count += 1
                    yield _event(
                        type="variant_skipped",
                        progress_percent=progress,
                        variant_id=exc.variant_id,
                        detected_consequence=exc.detected_consequence,
                        reason=f"Rejected by gatekeeper: Non-missense variant ({exc.detected_consequence}).",
                        error_code=exc.error_code,
                    )
                except InvalidVariantError as exc:
                    # Lỗi riêng của biến thể (REF sai build, không phải SNV...) -> bỏ qua, chạy tiếp.
                    db.rollback()
                    rejected_count += 1
                    yield _event(
                        type="variant_skipped",
                        progress_percent=progress,
                        variant_id=vid,
                        detected_consequence="invalid_variant",
                        reason=exc.message,
                        error_code=exc.error_code,
                    )
                except PipelineError as exc:
                    # Lỗi hạ tầng (thiếu scaler, VEP/Docker tắt...) ảnh hưởng mọi biến thể -> dừng stream.
                    db.rollback()
                    yield _event(type="stream_error", progress_percent=progress, variant_id=vid,
                                 reason=exc.message, error_code=exc.error_code)
                    return
                except Exception as exc:  # noqa: BLE001 - báo lỗi về UI thay vì cắt stream im lặng
                    db.rollback()
                    yield _event(type="stream_error", progress_percent=progress, variant_id=vid,
                                 reason=f"Unexpected inference error: {exc}", error_code="INTERNAL_ERROR")
                    return

                await asyncio.sleep(max(0, stream_delay_ms) / 1000.0)

            yield _event(type="stream_completed", progress_percent=100, summary=_summary())
        finally:
            db.close()

    return EventSourceResponse(event_generator())


@router.get("/batch-stream")
async def predict_batch_stream(
    stream_delay_ms: int = Query(120),
    dna_enabled: bool = Query(True),
    protein_enabled: bool = Query(True),
    bio_enabled: bool = Query(True),
    geo_enabled: bool = Query(True),
):
    """Stream SSE cho tập biến thể mẫu khi người dùng không upload file."""
    return _create_batch_sse_response(
        variant_queue=_get_demo_cohort_variants(),
        stream_delay_ms=stream_delay_ms,
        dna_enabled=dna_enabled,
        protein_enabled=protein_enabled,
        bio_enabled=bio_enabled,
        geo_enabled=geo_enabled,
    )


@router.post("/batch-upload")
async def predict_batch_upload(
    file: UploadFile = File(...),
    stream_delay_ms: int = Form(120),
    dna_enabled: bool = Form(True),
    protein_enabled: bool = Form(True),
    bio_enabled: bool = Form(True),
    geo_enabled: bool = Form(True),
):
    """
    Nhận file .vcf / .csv / .txt từ Frontend, parse tọa độ biến thể, chạy VEP thật,
    lọc qua Missense Gatekeeper và trả về luồng SSE trực tiếp.
    """
    raw_bytes = await file.read()
    variant_queue = _parse_uploaded_variants(raw_bytes, filename=file.filename or "")

    if not variant_queue:
        raise HTTPException(
            status_code=400,
            detail=f"No valid genomic variants found in '{file.filename}'. Expected VCF rows or chrN:POS:REF>ALT coordinates.",
        )

    return _create_batch_sse_response(
        variant_queue=variant_queue,
        stream_delay_ms=stream_delay_ms,
        dna_enabled=dna_enabled,
        protein_enabled=protein_enabled,
        bio_enabled=bio_enabled,
        geo_enabled=geo_enabled,
    )
