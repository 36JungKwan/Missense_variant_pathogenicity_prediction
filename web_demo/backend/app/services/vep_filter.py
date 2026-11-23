"""Missense Gatekeeper.

Chú thích hậu quả giờ đến từ Ensembl VEP thật (xem `app.services.vep_runner`); module này chỉ còn
giữ điểm import cũ của `NonMissenseVariantError` và hàm kiểm tra gatekeeper.
"""
from app.core.errors import NonMissenseVariantError
from app.services.vep_runner import VepRecord

__all__ = ["NonMissenseVariantError", "ensure_missense"]


def ensure_missense(record: VepRecord) -> VepRecord:
    """Chặn mọi biến thể mà VEP (--pick, MANE Select/Canonical) không gán hậu quả missense_variant."""
    if not record.is_missense:
        raise NonMissenseVariantError(record.variant_id, record.consequence)
    return record
