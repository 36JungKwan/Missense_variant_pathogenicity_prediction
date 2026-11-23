from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.checkpoint_registry import checkpoint_registry, resolve_device
from app.services.feature_pipeline import artifact_status

router = APIRouter()


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    db_status = "disconnected"
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "error"

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
        "model_mode": "pure_dl_inference",
        "device": str(resolve_device()),
        "checkpoints": checkpoint_registry.status(),
        "artifacts": artifact_status(),
    }
