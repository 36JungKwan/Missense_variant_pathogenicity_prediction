"""Logging UTF-8 có request_id và đo thời gian từng chặng của pipeline.

    with stage_scope() as timings:        # mở một phạm vi đo cho một biến thể
        with timed_stage("vep"):
            ...
    timings -> {"vep": 5123, ...}  (ms)
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import logging.config
import sys
import time
from typing import Callable, Dict, Iterator, Optional, TypeVar

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_timings_var: contextvars.ContextVar[Optional[Dict[str, int]]] = contextvars.ContextVar("stage_timings", default=None)

# Thứ tự cố định khi in dòng tổng kết.
STAGE_ORDER = ("vep", "spliceai", "fm_dna", "fm_prot", "geom", "norm", "shap", "db")

T = TypeVar("T")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    # Windows mặc định cp1252: ép UTF-8 để log tiếng Việt không làm sập request.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"request_id": {"()": _RequestIdFilter}},
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)-7s [req=%(request_id)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
                "filters": ["request_id"],
            }
        },
        "loggers": {
            "app": {"level": level.upper(), "handlers": ["console"], "propagate": False},
            "uvicorn.error": {"level": "INFO"},
        },
    })
    # Thư viện bên ngoài quá ồn ở mức INFO/WARNING.
    for noisy in ("transformers", "huggingface_hub", "sklearn"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


@contextlib.contextmanager
def stage_scope() -> Iterator[Dict[str, int]]:
    """Mở phạm vi đo mới (một biến thể); các `timed_stage` bên trong ghi vào dict trả về."""
    timings: Dict[str, int] = {}
    token = _timings_var.set(timings)
    try:
        yield timings
    finally:
        _timings_var.reset(token)


@contextlib.contextmanager
def timed_stage(name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        record_stage(name, (time.perf_counter() - started) * 1000)


def record_stage(name: str, elapsed_ms: float) -> None:
    timings = _timings_var.get()
    if timings is not None:
        timings[name] = timings.get(name, 0) + int(round(elapsed_ms))


def format_timings(timings: Dict[str, int]) -> str:
    keys = [k for k in STAGE_ORDER if k in timings] + sorted(k for k in timings if k not in STAGE_ORDER)
    return " ".join(f"{k}={timings[k]}" for k in keys)


def run_in_context(fn: Callable[..., T]) -> Callable[..., T]:
    """Bọc hàm để chạy trong ThreadPoolExecutor mà vẫn giữ request_id + phạm vi đo hiện tại."""
    ctx = contextvars.copy_context()
    return lambda *args, **kwargs: ctx.run(fn, *args, **kwargs)
