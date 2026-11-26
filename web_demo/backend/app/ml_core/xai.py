"""Giải thích dự đoán TriMVPP bằng Integrated Gradients (IG) và attention map của cross-attention.

Giữ nguyên phương pháp của notebook 19 (module06: cross_attention_explainability.integrated_gradients,
cross_attention_interpretability_v2):
  * IG chế độ full_prediction: baseline = token trung bình (theo mask) + trung vị đặc trưng của 256 biến thể
    train3 (seed 2026), tích phân Riemann-trapezoid 64 bước; chạy lại 128 bước nếu
    |residual| / |logit - logit_baseline| > 5%. Đầu ra được giải thích: logit Pathogenic trước sigmoid.
  * Attention: trọng số của từng head (không lấy trung bình trong mạng), trục key là các nhánh còn lại.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .trimvpp_model import TriMVPP

BRANCH_TO_INPUT = {"dna": "token_dna", "protein": "token_protein"}
QUERY_NAME = {"dna": "dna", "protein": "protein", "bio": "bio_geometry"}
MODALITY_LABEL = {"token_dna": "DNA", "token_protein": "Protein", "bio": "Bio_SpliceAI", "geom": "Geometry"}


def token_labels(prefix: str, k_radius: int) -> List[str]:
    return [f"{prefix}:{position:+d}" for position in range(-k_radius, k_radius + 1)]


def load_baseline(path) -> Dict[str, torch.Tensor]:
    """File của nhóm có khoá token_dna, token_protein, bio_features, geom_features; đổi sang tên đầu vào của TriMVPP."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    rename = {"token_dna": "token_dna", "token_protein": "token_protein", "bio_features": "bio", "geom_features": "geom"}
    baseline = {rename[key]: value.to(torch.float32) for key, value in raw.items()}
    expected = {"token_dna": 4, "token_protein": 4, "bio": 2, "geom": 2}
    for key, ndim in expected.items():
        if key not in baseline or baseline[key].ndim != ndim or baseline[key].shape[0] != 1:
            raise ValueError(f"IG baseline file {path} has an unexpected '{key}' tensor")
    return baseline


def _explained_keys(model: TriMVPP) -> List[str]:
    keys = [BRANCH_TO_INPUT[b] for b in ("dna", "protein") if b in model.active_branches]
    if "bio" in model.active_branches:
        keys += ["bio", "geom"]
    return keys


def integrated_gradients(
    model: TriMVPP,
    inputs: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    baseline: Dict[str, torch.Tensor],
    n_steps: int,
    internal_batch_size: int,
) -> Dict:
    """IG cho MỘT biến thể (batch = 1). inputs/baseline: các tensor float32 cùng shape; masks cố định."""
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    keys = list(inputs)
    device = next(model.parameters()).device

    def score(values: Dict[str, torch.Tensor], count: int) -> torch.Tensor:
        fixed = {name: mask.expand((count,) + tuple(mask.shape[1:])) for name, mask in masks.items()}
        return model(**values, **fixed).reshape(-1)

    with torch.no_grad():
        actual_logit = float(score(inputs, 1)[0])
        baseline_logit = float(score(baseline, 1)[0])

    gradient_sums = {key: torch.zeros_like(inputs[key]) for key in keys}
    alphas = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=device)
    chunk_size = max(1, int(internal_batch_size))
    for start in range(0, len(alphas), chunk_size):
        chunk = alphas[start:start + chunk_size]
        interpolated = {}
        for key in keys:
            alpha = chunk.view((len(chunk),) + (1,) * (inputs[key].ndim - 1))
            values = baseline[key] + alpha * (inputs[key] - baseline[key])
            interpolated[key] = values.detach().requires_grad_(True)
        logits = score(interpolated, len(chunk))
        gradients = torch.autograd.grad(logits.sum(), tuple(interpolated[key] for key in keys))
        # Quy tắc hình thang: hai đầu mút có trọng số 0.5.
        weights = torch.ones(len(chunk), device=device)
        global_indices = torch.arange(start, start + len(chunk), device=device)
        weights[(global_indices == 0) | (global_indices == len(alphas) - 1)] = 0.5
        for key, gradient in zip(keys, gradients):
            view = (len(chunk),) + (1,) * (gradient.ndim - 1)
            gradient_sums[key] += (gradient.detach() * weights.view(view)).sum(dim=0, keepdim=True)

    attributions = {
        key: ((inputs[key] - baseline[key]) * gradient_sums[key] / float(n_steps)).detach().cpu()
        for key in keys
    }
    residual = actual_logit - baseline_logit - sum(float(value.sum()) for value in attributions.values())
    return {
        "attributions": attributions,
        "actual_logit": actual_logit,
        "baseline_logit": baseline_logit,
        "completeness_residual": residual,
    }


def _relative_residual(result: Dict) -> float:
    denominator = abs(result["actual_logit"] - result["baseline_logit"])
    residual = abs(result["completeness_residual"])
    if denominator <= 1e-8:
        return 0.0 if residual <= 1e-8 else float("inf")
    return residual / denominator


def explain_ig(
    model: TriMVPP,
    inputs: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    baseline: Dict[str, torch.Tensor],
    *,
    bio_columns: Sequence[str],
    geom_columns: Sequence[str],
    n_steps: int = 64,
    retry_steps: int = 128,
    retry_threshold: float = 0.05,
    internal_batch_size: int = 8,
) -> Dict:
    """IG full_prediction, kèm bảng theo nhánh / vị trí token / đặc trưng (cùng định nghĩa với notebook 19)."""
    keys = _explained_keys(model)
    device = next(model.parameters()).device
    inputs = {key: inputs[key].to(device=device, dtype=torch.float32) for key in keys}
    used_baseline = {key: baseline[key].to(device=device, dtype=torch.float32) for key in keys}

    with torch.enable_grad():
        result = integrated_gradients(model, inputs, masks, used_baseline, n_steps, internal_batch_size)
        steps_used = n_steps
        ratio = _relative_residual(result)
        if ratio > retry_threshold and retry_steps > n_steps:
            result = integrated_gradients(model, inputs, masks, used_baseline, retry_steps, internal_batch_size)
            steps_used = retry_steps
            ratio = _relative_residual(result)

    k_radius = model.k_radius
    modalities, tokens, features = [], [], []
    for key in keys:
        attribution = result["attributions"][key][0]
        modalities.append({
            "modality": MODALITY_LABEL[key],
            "signed": float(attribution.sum()),
            "absolute": float(attribution.abs().sum()),
        })
        if key.startswith("token_"):
            values = attribution.numpy()  # (2, 33, D)
            for allele_index, allele in enumerate(("Ref", "Alt")):
                for position in range(values.shape[1]):
                    tokens.append({
                        "modality": MODALITY_LABEL[key],
                        "allele": allele,
                        "position": position - k_radius,
                        "signed": float(values[allele_index, position].sum()),
                        "absolute": float(np.abs(values[allele_index, position]).sum()),
                    })
        else:
            names = bio_columns if key == "bio" else geom_columns
            input_values = inputs[key][0].detach().cpu().numpy().reshape(-1)
            for index, name in enumerate(names):
                features.append({
                    "feature": str(name),
                    "group": MODALITY_LABEL[key],
                    "input_value": float(input_values[index]),
                    "signed": float(attribution[index]),
                })
    return {
        "method": "integrated_gradients",
        "mode": "full_prediction",
        "baseline": "train3_background_256_seed2026",
        "steps": int(steps_used),
        "logit": result["actual_logit"],
        "baseline_logit": result["baseline_logit"],
        "completeness_residual": result["completeness_residual"],
        "relative_residual": float(ratio),
        "residual_over_5pct": bool(ratio > retry_threshold),
        "modalities": modalities,
        "tokens": tokens,
        "features": features,
    }


@torch.inference_mode()
def attention_maps(
    model: TriMVPP,
    inputs: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    *,
    bio_columns: Sequence[str],
    geom_columns: Sequence[str],
) -> Optional[Dict]:
    """Attention của từng lớp/head. None khi model chỉ có một nhánh (không có cross-attention)."""
    if len(model.cross_layers) == 0:
        return None
    device = next(model.parameters()).device
    keys = _explained_keys(model)
    values = {key: inputs[key].to(device=device, dtype=torch.float32) for key in keys}
    _, maps = model(**values, **masks, return_attention=True)

    k_radius = model.k_radius
    labels = {
        "dna": token_labels("DNA", k_radius),
        "protein": token_labels("Protein", k_radius),
        "bio": [*map(str, bio_columns), *map(str, geom_columns)],
    }
    branches = model.active_branches
    layers = []
    for layer_index in range(1, len(model.cross_layers) + 1):
        entries = []
        for branch in branches:
            weights = maps[f"layer_{layer_index}_{branch}"][0].float().cpu().numpy()  # (heads, Lq, Lk)
            contexts = [other for other in branches if other != branch]
            entries.append({
                "query": QUERY_NAME[branch],
                "query_labels": labels[branch],
                "context_segments": [
                    {"name": QUERY_NAME[other], "labels": labels[other]} for other in contexts
                ],
                "key_labels": [label for other in contexts for label in labels[other]],
                "weights": weights,
            })
        layers.append({"layer": layer_index, "maps": entries})
    return {"num_layers": len(model.cross_layers), "num_heads": int(layers[0]["maps"][0]["weights"].shape[0]),
            "layers": layers}
