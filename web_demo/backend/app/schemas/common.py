from typing import List, Optional
from pydantic import BaseModel, Field
from app.core.constants import PredictionLabel


class TriMVPPScore(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    label: PredictionLabel


class ModelResults(BaseModel):
    trimvpp: TriMVPPScore


class SotaScores(BaseModel):
    # None = dbNSFP không có điểm cho biến thể này (Frontend hiển thị "N/A").
    alphamissense: Optional[float] = Field(None, ge=0.0, le=1.0)
    revel: Optional[float] = Field(None, ge=0.0, le=1.0)


class IgModality(BaseModel):
    modality: str  # DNA | Protein | Bio_SpliceAI | Geometry
    signed: float
    absolute: float


class IgToken(BaseModel):
    modality: str  # DNA | Protein
    allele: str  # Ref | Alt
    position: int  # -16..+16 quanh biến thể
    signed: float
    absolute: float


class IgFeature(BaseModel):
    feature: str
    name: str
    group: str  # Bio_SpliceAI | Geometry
    description: Optional[str] = None
    raw_value: Optional[float] = None  # giá trị thô trước chuẩn hoá (None = thiếu, đã điền)
    raw_note: Optional[str] = None
    input_value: float  # giá trị đã chuẩn hoá đưa vào model
    signed: float


class IgAnalysis(BaseModel):
    """Integrated Gradients giải thích logit Pathogenic (trước sigmoid) so với baseline train3:
    tổng mọi đóng góp + completeness_residual = logit - baseline_logit."""
    method: str = "integrated_gradients"
    mode: str = "full_prediction"
    baseline: str
    steps: int
    logit: float
    baseline_logit: float
    completeness_residual: float
    relative_residual: float
    residual_over_5pct: bool
    modalities: List[IgModality]
    tokens: List[IgToken]
    features: List[IgFeature]


class AttentionSegment(BaseModel):
    name: str
    labels: List[str]


class AttentionMap(BaseModel):
    query: str  # dna | protein | bio_geometry
    query_labels: List[str]
    context_segments: List[AttentionSegment]
    key_labels: List[str]
    weights: List[List[List[float]]]  # (heads, query, key); mỗi hàng query cộng lại = 1


class AttentionLayer(BaseModel):
    layer: int
    maps: List[AttentionMap]


class AttentionMaps(BaseModel):
    num_layers: int
    num_heads: int
    layers: List[AttentionLayer]