from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from app.core.constants import ClassificationLabel
from app.schemas.common import AttentionMaps, IgAnalysis, ModelResults, SotaScores


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
    test_mcc: Optional[float] = None  # MCC của tổ hợp này trên tập Test nội bộ
    warnings: List[str] = []
    # Thời gian từng chặng (ms): vep, spliceai, fm_dna, fm_prot, geom, norm, ig, attn, db
    timings_ms: Dict[str, int] = {}


class SingleVariantResponse(BaseModel):
    variant_id: str
    gene: str
    consequence: str
    clinvar_ground_truth: ClassificationLabel
    results: ModelResults
    sota: SotaScores
    # None: chưa tính IG (chỉ tính khi mở trang XAI); /xai luôn trả giá trị.
    ig_analysis: Optional[IgAnalysis] = None
    meta: InferenceMeta


class XaiResponse(SingleVariantResponse):
    # None: tổ hợp chỉ một nhánh (không có cross-attention).
    attention: Optional[AttentionMaps] = None
