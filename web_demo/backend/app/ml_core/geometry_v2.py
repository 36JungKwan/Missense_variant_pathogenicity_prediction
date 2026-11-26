"""Đặc trưng hình học V2 (LVD + LID) cho một cặp embedding, khớp module03 của repo huấn luyện.

Các bước làm tròn float16 (delta, LVD_L2, LVD_Cosine) là chủ ý: pipeline huấn luyện lưu embedding fp16 trên GPU
và trả LVD ở fp16 trước khi fit scaler. LLR = 0 vì bộ geometry step2_matched không dùng LLR.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

GEOM_V2_MODEL_COLUMNS = ["LVD_L2", "LVD_Cosine", "LID", "LVD_Relative"]


def _lid(delta: np.ndarray, index, k: int, epsilon: float) -> float:
    if k > index.ntotal:
        raise ValueError(f"FAISS index has {index.ntotal} vectors, cannot query k={k}")
    sq_dist, _ = index.search(np.ascontiguousarray(delta.astype(np.float32).reshape(1, -1)), k)
    r = np.sqrt(np.maximum(sq_dist, 0))
    log_ratio = np.log(np.clip(r / (r[:, -1:] + epsilon), epsilon, 1.0))
    sum_log = np.minimum(np.sum(log_ratio, axis=1), -epsilon)
    return float(-(k / sum_log[0]))


def compute_online_geometry(e_ref, e_alt, index=None, k_neighbors: int = 64, epsilon: float = 1e-8) -> Dict[str, float]:
    """e_ref / e_alt: vector token tâm (1280) ở float16. Thiếu FAISS index thì LID = NaN (điền trung vị train)."""
    ref16 = np.asarray(e_ref).astype(np.float16).reshape(-1)
    alt16 = np.asarray(e_alt).astype(np.float16).reshape(-1)
    delta = alt16 - ref16  # float16
    ref = torch.from_numpy(ref16.astype(np.float32))[None]
    alt = torch.from_numpy(alt16.astype(np.float32))[None]

    l2 = torch.norm(alt - ref, p=2, dim=1).to(torch.float16).item()
    cosine = (1.0 - F.cosine_similarity(ref, alt, dim=1)).to(torch.float16).item()
    ref_norm = torch.linalg.vector_norm(ref, ord=2, dim=1).item()
    return {
        "LLR": 0.0,
        "LVD_L2": l2,
        "LVD_Cosine": cosine,
        "LID": float("nan") if index is None else _lid(delta, index, int(k_neighbors), epsilon),
        "LVD_Relative": l2 / (ref_norm + 1e-8),
    }
