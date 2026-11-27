import asyncio
import logging
import threading
import time
from typing import List

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.core.constants import compute_ablation_mask, parse_variant_id
from app.core.config import settings
from app.core.errors import InvalidUploadError, InvalidVariantError, NonMissenseVariantError, PipelineError
from app.db.session import get_db, SessionLocal
from app.schemas.single import SingleVariantRequest, SingleVariantResponse
from app.schemas.batch import BatchVariant, BatchStreamEvent
from app.services.upload_parser import parse_uploaded_variants
from app.services.inference_service import (
    prefetch_annotations,
    prepare_features_batch,
    resolve_model,
    run_single_variant_inference,
)

logger = logging.getLogger("app.api.predict")
router = APIRouter()


@router.post("/single", response_model=SingleVariantResponse)
def predict_single_variant(
    request: SingleVariantRequest,
    db: Session = Depends(get_db),
):
    # PipelineError (400/409/422/503) được handler toàn cục trong app.main chuyển thành JSON detail.
    return run_single_variant_inference(db=db, request=request)


def _get_demo_cohort_variants() -> List[str]:
    """Cohort mẫu: các đột biến missense kinh điển, toạ độ GRCh38 (REF đã đối chiếu với hg38 FASTA)."""
    return [
        "chr7:140753336:A>T",   # BRAF p.Val600Glu
        "chr17:7674220:C>T",    # TP53 p.Arg248Gln
        "chr17:7673802:C>T",    # TP53 p.Arg273His
        "chr17:7675088:C>T",    # TP53 p.Arg175His
        "chr12:25245350:C>T",   # KRAS p.Gly12Asp
        "chr3:179234297:A>G",   # PIK3CA p.His1047Arg
        "chr3:179218303:G>A",   # PIK3CA p.Glu545Lys
        "chr7:55191822:T>G",    # EGFR p.Leu858Arg
        "chr2:208248388:C>T",   # IDH1 p.Arg132His
        "chr1:114713908:T>C",   # NRAS p.Gln61Arg
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
        pending_prefetch = None
        # Set khi stream kết thúc (xong, lỗi, hoặc trình duyệt ngắt kết nối vì người dùng bấm Stop / rời trang):
        # VEP đang chạy (kể cả khối chạy trước) bị kill, các khối còn lại không được xử lý.
        cancel_event = threading.Event()
        finished = False
        accepted_count = 0
        rejected_count = 0
        started = time.perf_counter()

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

            # File được chia thành các khối BATCH_CHUNK_VARIANTS biến thể. Mỗi khối: VEP + SpliceAI -> FM (micro-batch
            # FM_BATCH_VARIANTS) -> dự đoán -> gửi kết quả ngay. VEP + SpliceAI của khối kế tiếp chạy song song (thread
            # + session riêng) trong lúc GPU xử lý khối hiện tại, nên tổng thời gian gần bằng thời gian VEP.
            keys = [parse_variant_id(vid) for vid in variant_queue]
            block = max(1, settings.BATCH_CHUNK_VARIANTS)
            blocks = [keys[i:i + block] for i in range(0, total_count, block)]
            n_blocks = len(blocks)

            def _prefetch_block(block_keys):
                session = SessionLocal()
                try:
                    return prefetch_annotations(session, block_keys, cancel_event=cancel_event)
                finally:
                    session.close()

            def _start_prefetch(index: int) -> asyncio.Future:
                task = asyncio.ensure_future(asyncio.to_thread(_prefetch_block, blocks[index]))
                # Tránh cảnh báo "exception was never retrieved" khi stream bị huỷ giữa chừng.
                task.add_done_callback(lambda f: f.cancelled() or f.exception())
                return task

            def _status(index: int, stage: str, message: str) -> dict:
                return _event(type="stream_status", progress_percent=int(done / total_count * 100), stage=stage,
                              chunk=index + 1, total_chunks=n_blocks, total_variants=total_count, status=message)

            done = 0
            chunk_size = max(1, settings.FM_BATCH_VARIANTS)
            pending_prefetch = _start_prefetch(0)
            for block_index, block_keys in enumerate(blocks):
                if not pending_prefetch.done():
                    yield _status(block_index, "annotating",
                                  f"Chunk {block_index + 1}/{n_blocks}: annotating {len(block_keys)} variants "
                                  "with VEP + SpliceAI")
                try:
                    prefetched = await pending_prefetch
                except PipelineError as exc:
                    db.rollback()
                    yield _event(type="stream_error", progress_percent=int(done / total_count * 100),
                                 reason=exc.message, error_code=exc.error_code)
                    return
                except OperationalError:
                    logger.error("Database unavailable during batch prefetch")
                    yield _event(type="stream_error", progress_percent=int(done / total_count * 100),
                                 error_code="DATABASE_UNAVAILABLE",
                                 reason="PostgreSQL is not reachable. Start Docker Desktop and the rbnet_postgres container.")
                    return
                except Exception:  # noqa: BLE001
                    logger.exception("Batch prefetch failed")
                    yield _event(type="stream_error", progress_percent=int(done / total_count * 100),
                                 error_code="INTERNAL_ERROR", reason="Unexpected inference error (see server log).")
                    return
                pending_prefetch = _start_prefetch(block_index + 1) if block_index + 1 < n_blocks else None
                yield _status(block_index, "classifying",
                              f"Chunk {block_index + 1}/{n_blocks}: extracting features and classifying "
                              f"{len(block_keys)} variants")

                for chunk_start in range(0, len(block_keys), chunk_size):
                    chunk_keys = block_keys[chunk_start:chunk_start + chunk_size]
                    # Foundation Models chạy gộp cho cả nhóm; đặc trưng được lưu vào DB trước khi suy luận.
                    try:
                        failures = await asyncio.to_thread(prepare_features_batch, db, chunk_keys, prefetched)
                    except PipelineError as exc:
                        db.rollback()
                        yield _event(type="stream_error", progress_percent=int(done / total_count * 100),
                                     reason=exc.message, error_code=exc.error_code)
                        return
                    except Exception:  # noqa: BLE001 - báo lỗi về UI thay vì cắt stream im lặng
                        logger.exception("Batch feature extraction failed")
                        db.rollback()
                        yield _event(type="stream_error", progress_percent=int(done / total_count * 100),
                                     reason="Unexpected inference error (see server log).", error_code="INTERNAL_ERROR")
                        return

                    for key in chunk_keys:
                        vid = key.variant_id
                        done += 1
                        progress = int(done / total_count * 100)
                        req = SingleVariantRequest(
                            variant_id=vid,
                            dna_enabled=dna_enabled,
                            protein_enabled=protein_enabled,
                            bio_enabled=bio_enabled,
                            geo_enabled=geo_enabled,
                            use_cache=True,
                        )
                        try:
                            if vid in failures:
                                raise failures[vid]
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
                                    trimvpp_probability=res.results.trimvpp.probability,
                                    classification=res.results.trimvpp.label,
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
                        except Exception:  # noqa: BLE001
                            logger.exception("Batch inference failed for %s", vid)
                            db.rollback()
                            yield _event(type="stream_error", progress_percent=progress, variant_id=vid,
                                         reason="Unexpected inference error (see server log).", error_code="INTERNAL_ERROR")
                            return

                        if stream_delay_ms > 0:
                            await asyncio.sleep(stream_delay_ms / 1000.0)

            logger.info("batch completed variants=%d accepted=%d rejected=%d total=%dms", total_count,
                        accepted_count, rejected_count, (time.perf_counter() - started) * 1000)
            finished = True
            yield _event(type="stream_completed", progress_percent=100, summary=_summary())
        finally:
            cancel_event.set()
            if pending_prefetch is not None and not pending_prefetch.done():
                pending_prefetch.cancel()
            if not finished:
                logger.info("batch stopped early (client disconnected or error) after %d/%d variants; "
                            "background VEP/SpliceAI work cancelled", accepted_count + rejected_count, total_count)
            db.close()

    return EventSourceResponse(event_generator())


@router.get("/batch-stream")
async def predict_batch_stream(
    stream_delay_ms: int = Query(0, ge=0),
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
    stream_delay_ms: int = Form(0, ge=0),
    dna_enabled: bool = Form(True),
    protein_enabled: bool = Form(True),
    bio_enabled: bool = Form(True),
    geo_enabled: bool = Form(True),
):
    """
    Nhận file .vcf / .csv / .txt / .parquet / .xlsx từ Frontend, parse tọa độ biến thể, chạy VEP thật,
    lọc qua Missense Gatekeeper và trả về luồng SSE trực tiếp.
    """
    raw_bytes = await file.read()
    variant_queue = parse_uploaded_variants(raw_bytes, filename=file.filename or "")

    if not variant_queue:
        raise InvalidUploadError(
            f"No valid genomic variants found in '{file.filename}'. Expected VCF rows or chrN:POS:REF>ALT coordinates."
        )

    return _create_batch_sse_response(
        variant_queue=variant_queue,
        stream_delay_ms=stream_delay_ms,
        dna_enabled=dna_enabled,
        protein_enabled=protein_enabled,
        bio_enabled=bio_enabled,
        geo_enabled=geo_enabled,
    )
