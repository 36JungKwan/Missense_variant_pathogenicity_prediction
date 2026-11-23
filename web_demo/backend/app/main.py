import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.log import request_id_var, setup_logging, stage_scope, format_timings
from app.core.errors import PipelineError, error_body

setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger("app.main")

from app.api.v1.router import api_router  # noqa: E402  (logging phải được cấu hình trước)


def _preload() -> None:
    """Nạp FM (GPU fp16), FASTA, FAISS, scaler, SpliceAI .tbi và checkpoint 111 trước request đầu tiên.
    Lỗi chỉ được ghi log: request sau đó vẫn tự nạp lại và trả lỗi nghiệp vụ rõ ràng (503/409)."""
    from app.services import feature_pipeline
    from app.services.checkpoint_registry import checkpoint_registry

    with stage_scope() as timings:
        steps = (
            ("features", feature_pipeline.preload),
            ("checkpoint_111", lambda: checkpoint_registry.get(("dna", "prot", "bio", "geom"))),
        )
        for name, step in steps:
            started = time.perf_counter()
            try:
                step()
                logger.info("preload %s done in %d ms", name, (time.perf_counter() - started) * 1000)
            except PipelineError as exc:
                logger.error("preload %s failed: [%s] %s", name, exc.error_code, exc.message)
            except Exception:  # noqa: BLE001
                logger.exception("preload %s failed", name)
    logger.info("preload finished device=%s extractor_tag=%s %s", feature_pipeline.FM_DEVICE,
                feature_pipeline.EXTRACTOR_TAG, format_timings(timings))


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.PRELOAD_ON_STARTUP:
        await asyncio.to_thread(_preload)
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="RB-Net Hybrid Architecture - API Gateway",
    version="1.1.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Gán request_id, đo tổng thời gian, và biến mọi lỗi không lường trước thành JSON 500 thống nhất."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:8]
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            response = JSONResponse(
                status_code=500,
                content=error_body("INTERNAL_ERROR", "Unexpected server error. See server log for details.", request_id),
            )
        response.headers["X-Request-ID"] = request_id
        logger.info("%s %s -> %d (%d ms)", request.method, request.url.path, response.status_code,
                    (time.perf_counter() - started) * 1000)
        return response
    finally:
        request_id_var.reset(token)


# Thêm sau middleware request_context để CORS là lớp ngoài cùng (cả response lỗi 500 cũng có header CORS).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.exception_handler(PipelineError)
async def pipeline_error_handler(_: Request, exc: PipelineError) -> JSONResponse:
    level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
    logger.log(level, "%s: %s", exc.error_code, exc.message)
    detail = {**exc.to_detail(), "request_id": request_id_var.get()}
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(OperationalError)
async def database_error_handler(_: Request, exc: OperationalError) -> JSONResponse:
    logger.error("Database unavailable: %s", str(exc.orig).strip().splitlines()[0] if exc.orig else exc)
    return JSONResponse(status_code=503, content=error_body(
        "DATABASE_UNAVAILABLE",
        "PostgreSQL is not reachable. Start Docker Desktop and the rbnet_postgres container (start_system.bat).",
        request_id_var.get(),
    ))


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(status_code=exc.status_code, headers=getattr(exc, "headers", None),
                        content=error_body(f"HTTP_{exc.status_code}", message, request_id_var.get()))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = jsonable_encoder(exc.errors())
    first = errors[0] if errors else {}
    location = ".".join(str(part) for part in first.get("loc", []))
    message = f"Invalid request: {location} - {first.get('msg', 'validation error')}" if first else "Invalid request"
    return JSONResponse(status_code=422,
                        content=error_body("VALIDATION_ERROR", message, request_id_var.get(), errors=errors))


app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "status": "Online",
        "docs": "/docs",
        "api_prefix": settings.API_V1_STR,
    }
