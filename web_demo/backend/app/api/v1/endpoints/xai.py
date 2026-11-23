from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.constants import mask_to_flags
from app.db.session import get_db
from app.schemas.single import SingleVariantRequest
from app.services.inference_service import run_single_variant_inference

router = APIRouter()


@router.get("/{variant_id}")
def get_variant_xai(
    variant_id: str,
    ablation_mask: int = Query(15, ge=0, le=15),
    db: Session = Depends(get_db),
):
    """SHAP cho một biến thể dưới một ablation mask; đọc cache PostgreSQL hoặc tính mới bằng pipeline thật."""
    res = run_single_variant_inference(
        db=db,
        request=SingleVariantRequest(variant_id=variant_id, use_cache=True, **mask_to_flags(ablation_mask)),
    )
    return res.model_dump()
