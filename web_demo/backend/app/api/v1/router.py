from fastapi import APIRouter
from app.api.v1.endpoints import health, predict, xai

api_router = APIRouter()
api_router.include_router(health.router, tags=["System Health"])
api_router.include_router(predict.router, prefix="/predict", tags=["Pathogenicity Inference"])
api_router.include_router(xai.router, prefix="/xai", tags=["Explainable AI (SHAP)"])