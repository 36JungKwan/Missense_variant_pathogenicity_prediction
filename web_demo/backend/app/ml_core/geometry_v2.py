"""Đặc trưng hình học V2 (LVD + LID) cho một cặp embedding, dùng khi suy luận online.

Phần dựng FAISS index / fit scaler theo lô thuộc repo huấn luyện, không nằm trong backend.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

GEOM_V2_MODEL_COLUMNS = ["LVD_L2", "LVD_Cosine", "LID", "LVD_Relative"]


def compute_online_geometry(e_ref, e_alt, llr, index=None, k_neighbors: int = 32, epsilon: float = 1e-8) -> Dict[str, float]:
    """LVD_L2 / LVD_Cosine / LVD_Relative chỉ phụ thuộc cặp embedding. LID cần k láng giềng của vector
    delta trong FAISS index của context huấn luyện; thiếu index thì trả NaN để normalizer điền trung vị train."""
    ref = np.asarray(e_ref, dtype=np.float32).reshape(-1)
    alt = np.asarray(e_alt, dtype=np.float32).reshape(-1)
    delta = alt - ref
    ref_norm = float(np.linalg.norm(ref))
    alt_norm = float(np.linalg.norm(alt))
    l2 = float(np.linalg.norm(delta))
    cosine = 1.0 - float(np.dot(ref, alt)) / (ref_norm * alt_norm + epsilon)

    lid = float("nan")
    if index is not None:
        k = int(min(k_neighbors, index.ntotal))
        if k >= 2:
            sq_dist, _ = index.search(delta.reshape(1, -1), k)
            dist = np.sqrt(np.clip(sq_dist[0], 0.0, None)) + epsilon
            # Ước lượng MLE (Levina-Bickel) của Local Intrinsic Dimensionality.
            lid = float(-1.0 / (np.mean(np.log(dist / dist[-1])) - epsilon))

    return {
        "LLR": float(llr),
        "LVD_L2": l2,
        "LVD_Cosine": cosine,
        "LID": lid,
        "LVD_Relative": l2 / (ref_norm + 1e-8),
    }
