"""Audit geometry coverage, distributions, drift, leakage, and redundancy."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


GEOM_COLUMNS = ["LLR", "LVD_L2", "LVD_Cosine", "LID"]
LABEL_COLUMNS = ["Pathogenicity_Label", "Label", "label", "target", "Target"]


def _read_table(path):
    return pd.read_parquet(path)


def _resolve(path_candidates):
    for path in path_candidates:
        if os.path.exists(path):
            return path
    return None


def _bio_path(base_dir, split):
    return _resolve([
        f"{base_dir}/processed_parquet/{split}_normalized.parquet",
        f"{base_dir}/processed_parquet/{split}_full_seq_final_normalized.parquet",
        f"{base_dir}/processed_parquet/{split}_full_seq_after_vep_final_normalized.parquet",
    ])


def _geom_path(base_dir, split, config_name, normalized):
    suffix = "_geom_norm.parquet" if normalized else "_geom.parquet"
    return f"{base_dir}/geometry/{split}/{config_name}{suffix}"


def _as_numeric(df, columns):
    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _finite_values(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def _quantile(values, q):
    return float(np.quantile(values, q)) if len(values) else np.nan


def _ks_statistic(reference, values):
    reference = np.sort(reference[np.isfinite(reference)])
    values = np.sort(values[np.isfinite(values)])
    if len(reference) == 0 or len(values) == 0:
        return np.nan
    points = np.sort(np.unique(np.concatenate([reference, values])))
    ref_cdf = np.searchsorted(reference, points, side="right") / len(reference)
    value_cdf = np.searchsorted(values, points, side="right") / len(values)
    return float(np.max(np.abs(ref_cdf - value_cdf)))


def _pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    if np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return np.nan
    x_rank = _rank(x[mask])
    y_rank = _rank(y[mask])
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return np.nan
    return _pearson(x_rank, y_rank)


def _load_sampled_embeddings(path, sample_size, rng):
    import torch

    data = torch.load(path, weights_only=False)
    metadata = np.asarray(data["metadata"])
    e_ref = data["E_ref"]
    e_alt = data["E_alt"]
    n = len(metadata)
    indices = np.arange(n)
    if n > sample_size:
        indices = rng.choice(indices, size=sample_size, replace=False)
    if hasattr(e_ref, "detach"):
        ref = e_ref[indices].detach().cpu().numpy().astype(np.float32)
        alt = e_alt[indices].detach().cpu().numpy().astype(np.float32)
    else:
        ref = np.asarray(e_ref)[indices].astype(np.float32)
        alt = np.asarray(e_alt)[indices].astype(np.float32)
    return metadata[indices], alt - ref


def _pca_redundancy_audit(base_dir, config_name, splits, train_splits, geom_frames, sample_size, rng):
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return pd.DataFrame([{"status": "sklearn_missing"}])

    train_parts = []
    for split in train_splits:
        path = f"{base_dir}/fm_embeddings/{split}/{config_name}.pt"
        if not os.path.exists(path):
            return pd.DataFrame([{"status": "missing_embedding", "split": split, "path": path}])
        _, delta = _load_sampled_embeddings(path, sample_size, rng)
        train_parts.append(delta)

    train_delta = np.concatenate(train_parts, axis=0)
    n_components = min(32, train_delta.shape[0], train_delta.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_delta)

    rows = []
    for split in splits:
        path = f"{base_dir}/fm_embeddings/{split}/{config_name}.pt"
        if not os.path.exists(path) or split not in geom_frames:
            continue
        metadata, delta = _load_sampled_embeddings(path, sample_size, rng)
        geom = geom_frames[split].set_index("Variant_ID").reindex(metadata)[GEOM_COLUMNS]
        valid = geom.notna().all(axis=1).to_numpy()
        if valid.sum() < 3:
            continue
        pcs = pca.transform(delta[valid])
        for column in GEOM_COLUMNS:
            values = geom.loc[valid, column].to_numpy(dtype=np.float64)
            correlations = np.array([abs(_pearson(values, pcs[:, i])) for i in range(n_components)])
            best = int(np.nanargmax(correlations)) if np.isfinite(correlations).any() else -1
            rows.append({
                "split": split,
                "feature": column,
                "sample_count": int(valid.sum()),
                "best_abs_pearson_with_delta_pca": float(correlations[best]) if best >= 0 else np.nan,
                "best_pca_component": best + 1 if best >= 0 else np.nan,
                "pca_components_checked": n_components,
            })
    return pd.DataFrame(rows)


def audit_geometry(
    base_dir="D:/variant_data",
    config_name="nt_v2_500m_mean",
    splits=None,
    train_splits=None,
    output_dir=None,
    sample_size=10000,
    random_state=42,
):
    splits = splits or ["train1", "train2", "train3", "val", "test", "clinvarhq", "uniprot", "proteingym"]
    train_splits = train_splits or ["train1", "train2", "train3"]
    output_dir = output_dir or f"{base_dir}/geometry_audit/{config_name}"
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(random_state)

    coverage_rows = []
    distribution_rows = []
    drift_rows = []
    label_rows = []
    geom_frames = {}

    for split in splits:
        raw_path = _geom_path(base_dir, split, config_name, normalized=False)
        norm_path = _geom_path(base_dir, split, config_name, normalized=True)
        bio_path = _bio_path(base_dir, split)
        if not os.path.exists(raw_path) and not os.path.exists(norm_path):
            coverage_rows.append({"split": split, "status": "missing_geometry"})
            continue

        geom_path = norm_path if os.path.exists(norm_path) else raw_path
        geom = _as_numeric(_read_table(geom_path), GEOM_COLUMNS)
        geom_frames[split] = geom[["Variant_ID", *GEOM_COLUMNS]].copy()
        bio = _read_table(bio_path) if bio_path else None

        geom_ids = pd.Index(geom["Variant_ID"].dropna().unique())
        bio_ids = pd.Index(bio["Variant_ID"].dropna().unique()) if bio is not None else pd.Index([])
        overlap = geom_ids.intersection(bio_ids)
        coverage_rows.append({
            "split": split,
            "status": "ok",
            "geometry_path": geom_path,
            "bio_path": bio_path,
            "geometry_rows": int(len(geom)),
            "geometry_unique_variant_ids": int(len(geom_ids)),
            "bio_rows": int(len(bio)) if bio is not None else np.nan,
            "bio_unique_variant_ids": int(len(bio_ids)) if bio is not None else np.nan,
            "overlap_unique_variant_ids": int(len(overlap)),
            "geometry_only_ids": int(len(geom_ids.difference(bio_ids))) if bio is not None else np.nan,
            "bio_only_ids": int(len(bio_ids.difference(geom_ids))) if bio is not None else np.nan,
            "duplicate_geometry_ids": int(len(geom) - len(geom_ids)),
        })

        for column in GEOM_COLUMNS:
            values = _finite_values(geom[column])
            q1, q3 = _quantile(values, 0.25), _quantile(values, 0.75)
            iqr = q3 - q1 if np.isfinite(q1) and np.isfinite(q3) else np.nan
            lower = q1 - 1.5 * iqr if np.isfinite(iqr) else np.nan
            upper = q3 + 1.5 * iqr if np.isfinite(iqr) else np.nan
            distribution_rows.append({
                "split": split,
                "feature": column,
                "rows": int(len(geom)),
                "missing_or_non_numeric": int(len(geom) - len(values)),
                "finite_count": int(len(values)),
                "zero_count": int(np.sum(values == 0)) if len(values) else 0,
                "zero_fraction": float(np.mean(values == 0)) if len(values) else np.nan,
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "std": float(np.std(values)) if len(values) else np.nan,
                "min": float(np.min(values)) if len(values) else np.nan,
                "q01": _quantile(values, 0.01),
                "q50": _quantile(values, 0.50),
                "q99": _quantile(values, 0.99),
                "max": float(np.max(values)) if len(values) else np.nan,
                "iqr_outlier_count": int(np.sum((values < lower) | (values > upper))) if np.isfinite(lower) else 0,
            })

        if bio is not None:
            label_col = next((c for c in LABEL_COLUMNS if c in bio.columns), None)
            if label_col is not None:
                joined = geom[["Variant_ID", *GEOM_COLUMNS]].merge(
                    bio[["Variant_ID", label_col]], on="Variant_ID", how="inner"
                )
                labels = pd.to_numeric(joined[label_col], errors="coerce").to_numpy(dtype=np.float64)
                valid_label_mask = np.isfinite(labels)
                for column in GEOM_COLUMNS:
                    values = pd.to_numeric(joined[column], errors="coerce").to_numpy(dtype=np.float64)
                    label_rows.append({
                        "split": split,
                        "feature": column,
                        "label_column": label_col,
                        "sample_count": int((np.isfinite(values) & valid_label_mask).sum()),
                        "pearson_with_label": _pearson(values, labels),
                        "spearman_with_label": _spearman(values, labels),
                    })

    distribution = pd.DataFrame(distribution_rows)
    train_reference = {}
    for column in GEOM_COLUMNS:
        parts = []
        for split in train_splits:
            if split in geom_frames:
                parts.append(_finite_values(geom_frames[split][column]))
        train_reference[column] = np.concatenate(parts) if parts else np.array([])

    for split, geom in geom_frames.items():
        for column in GEOM_COLUMNS:
            values = _finite_values(geom[column])
            reference = train_reference[column]
            drift_rows.append({
                "reference_splits": "+".join(train_splits),
                "split": split,
                "feature": column,
                "reference_count": int(len(reference)),
                "split_count": int(len(values)),
                "mean_shift": float(np.mean(values) - np.mean(reference)) if len(values) and len(reference) else np.nan,
                "std_ratio": float(np.std(values) / np.std(reference)) if len(values) and len(reference) and np.std(reference) > 0 else np.nan,
                "ks_statistic": _ks_statistic(reference, values),
                "reference_q01": _quantile(reference, 0.01),
                "split_q01": _quantile(values, 0.01),
                "reference_q99": _quantile(reference, 0.99),
                "split_q99": _quantile(values, 0.99),
            })

    coverage_df = pd.DataFrame(coverage_rows)
    distribution_df = pd.DataFrame(distribution_rows)
    drift_df = pd.DataFrame(drift_rows)
    label_df = pd.DataFrame(label_rows)
    pca_df = _pca_redundancy_audit(
        base_dir,
        config_name,
        splits,
        train_splits,
        geom_frames,
        sample_size,
        rng,
    )

    index_path = f"{base_dir}/faiss_indexes/{config_name}.index"
    index_ntotal = np.nan
    if os.path.exists(index_path):
        try:
            import faiss
            index_ntotal = int(faiss.read_index(index_path).ntotal)
        except Exception:
            index_ntotal = np.nan

    expected_train_rows = int(sum(
        row.get("geometry_unique_variant_ids", 0)
        for row in coverage_rows
        if row.get("split") in train_splits
    ))
    leakage_check = pd.DataFrame([{
        "config_name": config_name,
        "faiss_index_path": index_path,
        "faiss_index_vectors": index_ntotal,
        "expected_vectors_from_train_splits": expected_train_rows,
        "index_matches_all_declared_train_splits": bool(
            np.isfinite(index_ntotal) and index_ntotal == expected_train_rows
        ),
        "warning": (
            "Index/scaler fit scope must match the active batch train split; "
            "the current notebook fits geometry context on train1+train2+train3."
        ),
    }])

    summary = {
        "config_name": config_name,
        "splits": splits,
        "train_splits": train_splits,
        "geometry_columns": GEOM_COLUMNS,
        "reports": {
            "coverage": f"{output_dir}/geometry_audit_coverage.csv",
            "distribution": f"{output_dir}/geometry_audit_distribution.csv",
            "drift": f"{output_dir}/geometry_audit_drift.csv",
            "label_association": f"{output_dir}/geometry_audit_label_association.csv",
            "pca_redundancy": f"{output_dir}/geometry_audit_pca_redundancy.csv",
            "leakage_check": f"{output_dir}/geometry_audit_leakage_check.csv",
        },
    }

    coverage_df.to_csv(summary["reports"]["coverage"], index=False)
    distribution_df.to_csv(summary["reports"]["distribution"], index=False)
    drift_df.to_csv(summary["reports"]["drift"], index=False)
    label_df.to_csv(summary["reports"]["label_association"], index=False)
    pca_df.to_csv(summary["reports"]["pca_redundancy"], index=False)
    leakage_check.to_csv(summary["reports"]["leakage_check"], index=False)
    with open(f"{output_dir}/geometry_audit_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


if __name__ == "__main__":
    result = audit_geometry()
    print(json.dumps(result, indent=2))
