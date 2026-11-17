from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
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
        request=SingleVariantRequest(
            variant_id=variant_id,
            dna_enabled=bool((ablation_mask >> 3) & 1),
            protein_enabled=bool((ablation_mask >> 2) & 1),
            bio_enabled=bool((ablation_mask >> 1) & 1),
            geo_enabled=bool(ablation_mask & 1),
            use_cache=True,
        ),
    )
    return {
        "variant_id": res.variant_id,
        "gene": res.gene,
        "consequence": res.consequence,
        "clinvar_ground_truth": res.clinvar_ground_truth,
        "results": res.results.model_dump(),
        "sota": res.sota.model_dump(),
        "shap_analysis": [item.model_dump() for item in res.shap_analysis],
        "meta": res.meta.model_dump(),
    }
