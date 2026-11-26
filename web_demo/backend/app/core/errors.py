"""Lỗi nghiệp vụ của pipeline suy luận, ánh xạ thẳng sang HTTP status + error_code.

Mọi lỗi API đều có dạng: {"detail": {"error_code": str, "message": str, "request_id": str, ...extra}}.
"""
from typing import Any, Dict, Optional


class PipelineError(Exception):
    status_code: int = 500
    error_code: str = "PIPELINE_ERROR"

    def __init__(self, message: str, **extra: Any):
        super().__init__(message)
        self.message = message
        self.extra: Dict[str, Any] = extra

    def to_detail(self) -> Dict[str, Any]:
        return {"error_code": self.error_code, "message": self.message, **self.extra}


class InvalidVariantError(PipelineError):
    """Sai định dạng, không phải SNV, hoặc REF không khớp hệ gen hg38."""
    status_code = 400
    error_code = "INVALID_VARIANT"


class InvalidUploadError(PipelineError):
    """File upload không chứa biến thể hợp lệ nào."""
    status_code = 400
    error_code = "INVALID_UPLOAD"


class NoModalitySelectedError(PipelineError):
    status_code = 400
    error_code = "NO_MODALITY_SELECTED"


class UnsupportedCombinationError(PipelineError):
    status_code = 400
    error_code = "UNSUPPORTED_MODALITY_COMBINATION"


class CheckpointMissingError(PipelineError):
    """Tổ hợp hợp lệ nhưng file .pth chưa được copy vào đường dẫn đã cấu hình."""
    status_code = 409
    error_code = "CHECKPOINT_MISSING"


class CheckpointIncompatibleError(PipelineError):
    status_code = 500
    error_code = "CHECKPOINT_INCOMPATIBLE"


class FeatureArtifactMissingError(PipelineError):
    """Thiếu scaler / FASTA / tài nguyên cần để tạo đặc trưng đúng như lúc huấn luyện."""
    status_code = 503
    error_code = "FEATURE_ARTIFACT_MISSING"


class FoundationModelUnavailableError(PipelineError):
    """Không nạp được NT-v1 / ESM-1b (sai model id, thiếu trọng số trong HF_HOME, hết VRAM...)."""
    status_code = 503
    error_code = "FOUNDATION_MODEL_UNAVAILABLE"


class VepUnavailableError(PipelineError):
    status_code = 503
    error_code = "VEP_UNAVAILABLE"


class NonMissenseVariantError(PipelineError):
    """Hậu quả VEP không phải missense."""
    status_code = 422
    error_code = "NON_MISSENSE_VARIANT"

    def __init__(self, variant_id: str, detected_consequence: str, message: Optional[str] = None):
        self.variant_id = variant_id
        self.detected_consequence = detected_consequence
        super().__init__(
            message
            or (
                f"Invalid variant consequence: '{detected_consequence}'. "
                "TriMVPP exclusively supports missense variants."
            ),
            variant_id=variant_id,
            detected_consequence=detected_consequence,
        )


def error_body(error_code: str, message: str, request_id: str, **extra: Any) -> Dict[str, Any]:
    return {"detail": {"error_code": error_code, "message": message, "request_id": request_id, **extra}}
