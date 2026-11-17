from typing import List, Optional
from pydantic import BaseModel, Field
from app.core.constants import ClassificationLabel
from app.schemas.common import ModelResults, SotaScores, ShapItem


class SingleVariantRequest(BaseModel):
    variant_id: str = Field(..., min_length=3, examples=["chr7:140753336:A>T"])
    dna_enabled: bool = True
    protein_enabled: bool = True
    bio_enabled: bool = True
    geo_enabled: bool = True
    # 3 nút bật/tắt gộp (Bio + Geom đi cùng nhau). Nếu gửi, chúng ghi đè 4 cờ *_enabled ở trên.
    use_dna: Optional[bool] = None
    use_prot: Optional[bool] = None
    use_biogeom: Optional[bool] = None
    use_cache: bool = True  # False buộc chạy lại suy luận thay vì đọc bảng predictions


class InferenceMeta(BaseModel):
    is_cache_hit: bool
    computation_time_ms: int
    ablation_mask: int
    active_modalities: List[str] = []
    checkpoint: Optional[str] = None
    architecture: Optional[str] = None
    decision_threshold: Optional[float] = None
    warnings: List[str] = []


class SingleVariantResponse(BaseModel):
    variant_id: str
    gene: str
    consequence: str
    clinvar_ground_truth: ClassificationLabel
    results: ModelResults
    sota: SotaScores
    shap_analysis: List[ShapItem]
    meta: InferenceMeta
