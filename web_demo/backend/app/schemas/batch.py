from typing import Optional, Literal, Dict
from pydantic import BaseModel
from app.core.constants import ClassificationLabel, PredictionLabel


class BatchVariant(BaseModel):
    id: int
    variant_id: str
    gene: str
    consequence: str
    trimvpp_probability: float
    classification: PredictionLabel
    clinvar_ground_truth: ClassificationLabel
    sota_alphamissense: Optional[float] = None
    danger_rank: int


class BatchStreamEvent(BaseModel):
    type: Literal["variant_stream", "variant_skipped", "stream_status", "stream_completed", "stream_error"]
    progress_percent: int
    variant: Optional[BatchVariant] = None
    variant_id: Optional[str] = None
    detected_consequence: Optional[str] = None
    reason: Optional[str] = None
    error_code: Optional[str] = None
    summary: Optional[Dict[str, int]] = None
    # stream_status: tiến độ theo khối (annotating = VEP + SpliceAI, classifying = FM + dự đoán)
    stage: Optional[Literal["annotating", "classifying"]] = None
    status: Optional[str] = None
    chunk: Optional[int] = None
    total_chunks: Optional[int] = None
    total_variants: Optional[int] = None