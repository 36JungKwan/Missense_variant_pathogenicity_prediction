import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.core.errors import PipelineError
from app.api.v1.router import api_router

# Log tiếng Việt của ml_core không được làm sập request khi stdout không phải UTF-8 (cp1252 trên Windows).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="RB-Net Hybrid Architecture - API Gateway",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
)

# Cấu hình CORS để Frontend (React/Vite cổng 5173) gọi được xuống Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(PipelineError)
async def pipeline_error_handler(_: Request, exc: PipelineError) -> JSONResponse:
    """Lỗi nghiệp vụ của pipeline -> {"detail": {"error_code", "message", ...}}, đúng dạng Frontend đang đọc."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.to_detail()})


# Nhúng toàn bộ các endpoint (Predict, XAI, Health) vào /api/v1
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "status": "Online",
        "docs": "/docs",
        "api_prefix": settings.API_V1_STR,
    }
