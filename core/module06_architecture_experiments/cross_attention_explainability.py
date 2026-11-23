"""Post-hoc explanations for the trained three-modality cross-attention model."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EXPLAINABLE_INPUT_KEYS = ("token_dna", "token_protein", "bio_features", "geom_features")


def _model_inputs(model, fixed_batch, values, return_attention=False):
    return model(
        fixed_batch["v_dna"],
        fixed_batch["v_prot"],
        values["bio_features"],
        values.get("geom_features"),
        token_dna=values["token_dna"],
        token_dna_mask=fixed_batch["token_dna_mask"],
        token_dna_variant_index=fixed_batch["token_dna_variant_index"],
        token_protein=values["token_protein"],
        token_protein_mask=fixed_batch["token_protein_mask"],
        token_protein_variant_index=fixed_batch["token_protein_variant_index"],
        return_attention=return_attention,
    )


def move_batch_to_device(batch, device):
    """Move tensor fields while leaving IDs and other metadata on the host."""
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def empirical_baseline(sample_batch, background_batch):
    """Build a data-based baseline from mean token states and median scalar features."""
    baseline = {}
    for modality in ("dna", "protein"):
        key = f"token_{modality}"
        mask = background_batch[f"token_{modality}_mask"].unsqueeze(-1).to(torch.float32)
        values = background_batch[key].to(torch.float32)
        counts = mask.sum(dim=0, keepdim=True).clamp_min(1.0)
        baseline[key] = (values * mask).sum(dim=0, keepdim=True) / counts
    for key in ("bio_features", "geom_features"):
        if sample_batch.get(key) is not None:
            baseline[key] = background_batch[key].to(torch.float32).median(dim=0, keepdim=True).values
    return baseline


def alt_equals_ref_baseline(sample_batch):
    """Baseline for the allele contrast: retain ref and set alt states to ref states."""
    baseline = {
        key: sample_batch[key].to(torch.float32).clone()
        for key in EXPLAINABLE_INPUT_KEYS
        if sample_batch.get(key) is not None
    }
    for key in ("token_dna", "token_protein"):
        baseline[key][:, 1] = baseline[key][:, 0]
    return baseline


def integrated_gradients(
    model, sample_batch, baseline, *, device, n_steps=32, internal_batch_size=8,
):
    """Riemann-trapezoid IG for one sample's classifier logit."""
    if int(n_steps) < 2:
        raise ValueError("n_steps must be at least 2")
    model.eval()
    batch = move_batch_to_device(sample_batch, device)
    keys = [key for key in EXPLAINABLE_INPUT_KEYS if batch.get(key) is not None]
    inputs = {key: batch[key].to(torch.float32) for key in keys}
    if any(value.ndim == 0 or value.shape[0] != 1 for value in inputs.values()):
        raise ValueError("integrated_gradients expects exactly one sample")
    baselines = {key: baseline[key].to(device=device, dtype=torch.float32) for key in keys}
    if any(inputs[key].shape != baselines[key].shape for key in keys):
        raise ValueError("Every IG baseline tensor must have the same shape as its input")

    def score(values):
        return _model_inputs(model, batch, values, return_attention=False)

    with torch.no_grad():
        actual_logit = float(score(inputs).reshape(-1)[0].item())
        baseline_logit = float(score(baselines).reshape(-1)[0].item())

    gradient_sums = {key: torch.zeros_like(inputs[key]) for key in keys}
    alphas = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=device)
    chunk_size = max(1, int(internal_batch_size))
    for start in range(0, len(alphas), chunk_size):
        chunk = alphas[start:start + chunk_size]
        interpolated = {}
        for key in keys:
            view_shape = (len(chunk),) + (1,) * (inputs[key].ndim - 1)
            alpha = chunk.view(view_shape)
            values = baselines[key] + alpha * (inputs[key] - baselines[key])
            interpolated[key] = values.detach().requires_grad_(True)
        # Expand fixed inputs and masks to the integration-step batch.
        fixed = dict(batch)
        for key, value in list(fixed.items()):
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
                fixed[key] = value.expand((len(chunk),) + tuple(value.shape[1:]))
        logits = _model_inputs(model, fixed, interpolated, return_attention=False).reshape(-1)
        gradients = torch.autograd.grad(logits.sum(), tuple(interpolated.values()))
        weights = torch.ones(len(chunk), device=device, dtype=torch.float32)
        global_indices = torch.arange(start, start + len(chunk), device=device)
        weights[(global_indices == 0) | (global_indices == len(alphas) - 1)] = 0.5
        for key, gradient in zip(keys, gradients):
            view_shape = (len(chunk),) + (1,) * (gradient.ndim - 1)
            gradient_sums[key] += (gradient.detach() * weights.view(view_shape)).sum(dim=0, keepdim=True)

    average_gradients = {key: value / float(n_steps) for key, value in gradient_sums.items()}
    attributions = {
        key: (inputs[key] - baselines[key]) * average_gradients[key]
        for key in keys
    }
    completeness_residual = actual_logit - baseline_logit - sum(
        float(value.sum().item()) for value in attributions.values()
    )
    return {
        "attributions": {key: value.detach().cpu() for key, value in attributions.items()},
        "actual_logit": actual_logit,
        "baseline_logit": baseline_logit,
        "completeness_residual": completeness_residual,
    }


def select_stratified_indices(dataset, max_samples, seed=42):
    """Select a reproducible, approximately class-balanced sample from a split."""
    frame = dataset.df.reset_index(drop=True)
    label_col = next((column for column in ("Pathogenicity_Label", "Label", "label", "target", "Target")
                      if column in frame.columns), None)
    if label_col is None:
        raise KeyError("No label column found for stratified explanation sampling")
    labels = pd.to_numeric(frame[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
    rng = np.random.default_rng(int(seed))
    per_class = max(1, int(max_samples) // 2)
    chosen = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        if len(candidates):
            chosen.extend(rng.choice(candidates, size=min(per_class, len(candidates)), replace=False).tolist())
    if len(chosen) < int(max_samples):
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(chosen, dtype=int))
        extra = rng.choice(remaining, size=min(int(max_samples) - len(chosen), len(remaining)), replace=False)
        chosen.extend(extra.tolist())
    return sorted(chosen)


def attention_summary_rows(attention_maps, *, seed, dataset, variant_id,
                           token_radius, bio_feature_names, geom_feature_names):
    """Reduce head/query maps to mean key weights while retaining pair and layer."""
    rows = []
    token_labels = list(range(-int(token_radius), int(token_radius) + 1))
    key_labels = {
        "dna": [f"DNA:{position:+d}" for position in token_labels],
        "protein": [f"Protein:{position:+d}" for position in token_labels],
        "bio_geometry": [*map(str, bio_feature_names), *map(str, geom_feature_names)],
    }
    for name, tensor in attention_maps.items():
        parts = name.split("_")
        layer = int(parts[1])
        pair = "_".join(parts[2:])
        query, context = pair.split("_to_", 1)
        weights = tensor.detach().float().cpu().numpy()
        mean_key_weights = weights.mean(axis=(0, 1, 2))  # batch, heads, query positions
        context_labels = key_labels[context]
        if len(context_labels) != len(mean_key_weights):
            context_labels = [f"{context}:{index}" for index in range(len(mean_key_weights))]
        for key_label, mean_weight in zip(context_labels, mean_key_weights):
            rows.append({
                "Seed": int(seed), "Dataset": str(dataset), "Variant_ID": str(variant_id),
                "Layer": layer, "Query_Modality": query,
                "Context_Modality": context, "Context_Token": key_label,
                "Mean_Attention_Over_Heads_And_Queries": float(mean_weight),
            })
    return rows


def occlusion_logit_deltas(
    model, sample_batch, *, device, token_radius, bio_feature_names,
    geom_feature_names, feature_baseline, chunk_size=16,
):
    """Measure logit change after ref-replacing alt tokens or median-replacing features."""
    model.eval()
    batch = move_batch_to_device(sample_batch, device)
    original = {key: batch[key].to(torch.float32).clone()
                for key in EXPLAINABLE_INPUT_KEYS if batch.get(key) is not None}

    def score(values):
        with torch.no_grad():
            return _model_inputs(model, batch, values, return_attention=False).reshape(-1)

    actual_logit = float(score(original)[0].item())
    perturbations = []
    for modality in ("dna", "protein"):
        key = f"token_{modality}"
        for position in range(2 * int(token_radius) + 1):
            values = {name: tensor.clone() for name, tensor in original.items()}
            values[key][:, 1, position] = values[key][:, 0, position]
            relative_position = position - int(token_radius)
            perturbations.append((f"{modality.upper()}_alt_to_ref_{relative_position:+d}", values))
    for key, names in (("bio_features", bio_feature_names), ("geom_features", geom_feature_names)):
        if key not in original:
            continue
        median = feature_baseline[key].to(device=device, dtype=torch.float32).reshape(-1)
        for index, name in enumerate(names):
            values = {field: tensor.clone() for field, tensor in original.items()}
            values[key][:, index] = median[index]
            perturbations.append((str(name), values))

    rows = []
    for start in range(0, len(perturbations), max(1, int(chunk_size))):
        chunk = perturbations[start:start + max(1, int(chunk_size))]
        count = len(chunk)
        fixed = dict(batch)
        for key, value in list(fixed.items()):
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
                fixed[key] = value.expand((count,) + tuple(value.shape[1:]))
        stacked = {
            key: torch.cat([values[key] for _, values in chunk], dim=0)
            for key in original
        }
        with torch.no_grad():
            perturbed_logits = _model_inputs(model, fixed, stacked, return_attention=False).reshape(-1)
        for (name, _), logit in zip(chunk, perturbed_logits):
            value = float(logit.item())
            rows.append({
                "Feature_or_Position": name,
                "Actual_Logit": actual_logit,
                "Occluded_Logit": value,
                "Logit_Delta_Actual_Minus_Occluded": actual_logit - value,
            })
    return rows


def upsert_csv(path, rows, keys):
    """Atomically upsert explanation rows so notebook execution can be resumed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        return
    current = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    combined = pd.concat((current, incoming), ignore_index=True)
    combined = combined.drop_duplicates(subset=list(keys), keep="last")
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    for attempt in range(5):
        try:
            temporary.replace(path)
            break
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            if attempt == 4:
                raise PermissionError(
                    f"Cannot replace {path}. Close any application using the file and retry. "
                    f"The pending data remains in {temporary}."
                ) from error
            time.sleep(0.25 * 2 ** attempt)
