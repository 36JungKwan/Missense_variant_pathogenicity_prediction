from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.constants import mask_to_flags
from app.db.session import get_db
from app.schemas.single import SingleVariantRequest, XaiResponse
from app.services.inference_service import run_xai

router = APIRouter()


@router.get("/{variant_id}", response_model=XaiResponse)
def get_variant_xai(
    variant_id: str,
    ablation_mask: int = Query(15, ge=0, le=15),
    db: Session = Depends(get_db),
):
    """Integrated Gradients (lưu cùng dự đoán trong PostgreSQL) + attention map 3 lớp x 8 head (tính khi gọi).
    attention = null với tổ hợp chỉ một nhánh."""
    return run_xai(
        db=db,
        request=SingleVariantRequest(variant_id=variant_id, use_cache=True, **mask_to_flags(ablation_mask)),
    )
