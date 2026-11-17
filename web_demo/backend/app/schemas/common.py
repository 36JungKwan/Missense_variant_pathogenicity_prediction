from typing import Optional
from pydantic import BaseModel, Field
from app.core.constants import PredictionLabel


class RBNetScore(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    label: PredictionLabel


class ModelResults(BaseModel):
    rb_net: RBNetScore


class SotaScores(BaseModel):
    alphamissense: float = Field(..., ge=0.0, le=1.0)
    revel: float = Field(..., ge=0.0, le=1.0)


class ShapItem(BaseModel):
    feature: str
    name: str
    contribution_value: float
    description: Optional[str] = None