"""Versioned geometry extraction with batch-specific context and robust normalization."""

import gc
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from core.module03_geometry_extraction.extractor import LatentGeometryCalculator
from core.module04_geo_bio_context_normalization.normalizer import (
    AdvancedFeatureNormalizer,
    GEOM_V2_COLUMNS,
)


@dataclass
class GeometryV2Config:
    base_dir: str = "D:/variant_data"
    contexts: list[str] = field(default_factory=lambda: ["train1", "train2", "train3"])
    splits: list[str] = field(
        default_factory=lambda: ["train1", "train2", "train3", "val", "test", "clinvarhq", "uniprot", "proteingym"]
    )
    models: list[str] = field(
        default_factory=lambda: [
            "nt_v1_500m", "nt_v3_650m", "nt_v2_500m",
            "esm1b_650m", "esm1v_650m", "esm2_650m", "esmc_600m",
        ]
    )
    poolings: list[str] = field(default_factory=lambda: ["cls", "center", "mean"])
    k_neighbors: int = 32
    epsilon: float = 1e-8
    resume: bool = True
    # Keep the original k=32 output paths backward compatible. New k values
    # should use a separate root so their normalized artifacts never collide.
    geometry_root: str = "geometry_v2"
    normalization_root: str = "normalization_artifacts_v2"


def _embedding_path(base_dir, split, config_id):
    return f"{base_dir}/fm_embeddings/{split}/{config_id}.pt"


def _raw_output_path(base_dir, context, split, config_id, geometry_root="geometry_v2"):
    return f"{base_dir}/{geometry_root}/{context}/{split}/{config_id}_geom_v2.parquet"


def _norm_output_path(base_dir, context, split, config_id, geometry_root="geometry_v2"):
    return f"{base_dir}/{geometry_root}/{context}/{split}/{config_id}_geom_v2_norm.parquet"


def _index_path(base_dir, context, config_id):
    return f"{base_dir}/faiss_indexes_v2/{context}/{config_id}.index"


def _artifact_dir(base_dir, context, normalization_root="normalization_artifacts_v2"):
    return f"{base_dir}/{normalization_root}/{context}"


def _load_delta(path):
    data = torch.load(path, weights_only=False)
    metadata = np.asarray(data["metadata"])
    e_ref = data["E_ref"]
    e_alt = data["E_alt"]
    delta = (e_alt - e_ref).detach().cpu().numpy().astype(np.float32)
    return data, metadata, e_ref, e_alt, delta


def _extract_raw_geometry(calculator, data, metadata, e_ref, e_alt, exclude_self):
    llr = data["llr"].detach().cpu().numpy().reshape(-1).astype(np.float32)
    e_ref_device = e_ref.to(calculator.device) if hasattr(calculator, "device") else e_ref.cuda()
    e_alt_device = e_alt.to(calculator.device) if hasattr(calculator, "device") else e_alt.cuda()
    geom = calculator.extract_geometry_features(
        e_ref_device,
        e_alt_device,
        exclude_self=exclude_self,
    )
    l2 = geom["LVD_L2"].cpu().numpy().reshape(-1).astype(np.float32)
    cosine = geom["LVD_Cosine"].cpu().numpy().reshape(-1).astype(np.float32)
    lid = geom["LID"].cpu().numpy().reshape(-1).astype(np.float32)
    ref_norm = torch.linalg.vector_norm(e_ref_device.float(), ord=2, dim=1).cpu().numpy()
    relative_l2 = l2 / (ref_norm + 1e-8)
    return pd.DataFrame({
        "Variant_ID": metadata,
        "LLR": llr,
        "LVD_L2": l2,
        "LVD_Cosine": cosine,
        "LID": lid,
        "LVD_Relative": relative_l2.astype(np.float32),
    })


def _build_context_index(config: GeometryV2Config, context: str, config_id: str):
    path = _index_path(config.base_dir, context, config_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    calculator = LatentGeometryCalculator(k_neighbors=config.k_neighbors, epsilon=config.epsilon)
    if config.resume and os.path.exists(path):
        calculator.load_global_index(path)
        return calculator

    embedding = _embedding_path(config.base_dir, context, config_id)
    if not os.path.exists(embedding):
        raise FileNotFoundError(f"Missing context embedding: {embedding}")
    data, _, _, _, delta = _load_delta(embedding)
    del data
    calculator.build_and_save_global_index(delta, path)
    calculator.load_global_index(path)
    del delta
    gc.collect()
    return calculator


def run_geometry_v2(config: GeometryV2Config | None = None):
    config = config or GeometryV2Config()
    normalizer = AdvancedFeatureNormalizer(epsilon=config.epsilon)
    output_rows = []

    for context in config.contexts:
        artifact_dir = _artifact_dir(config.base_dir, context, config.normalization_root)
        os.makedirs(artifact_dir, exist_ok=True)

        for model_name in config.models:
            for pooling in config.poolings:
                config_id = f"{model_name}_{pooling}"
                norm_paths = [
                    _norm_output_path(
                        config.base_dir, context, split, config_id, config.geometry_root
                    )
                    for split in config.splits
                ]
                if config.resume and all(os.path.exists(path) for path in norm_paths):
                    print(f"[V2 RESUME] Skip {context}/{config_id}")
                    continue

                print(f"[V2] context={context} | config={config_id}")
                calculator = _build_context_index(config, context, config_id)
                raw_frames = {}
                try:
                    for split in config.splits:
                        embedding = _embedding_path(config.base_dir, split, config_id)
                        raw_path = _raw_output_path(
                            config.base_dir, context, split, config_id, config.geometry_root
                        )
                        if config.resume and os.path.exists(raw_path):
                            raw_frames[split] = pd.read_parquet(raw_path)
                            continue
                        if not os.path.exists(embedding):
                            print(f"[V2 WARN] Missing embedding: {embedding}")
                            continue

                        data, metadata, e_ref, e_alt, _ = _load_delta(embedding)
                        frame = _extract_raw_geometry(
                            calculator,
                            data,
                            metadata,
                            e_ref,
                            e_alt,
                            exclude_self=(split == context),
                        )
                        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                        frame.to_parquet(raw_path, index=False)
                        raw_frames[split] = frame
                        del data, e_ref, e_alt
                        gc.collect()

                    if context not in raw_frames:
                        raise RuntimeError(f"Context geometry missing for {context}/{config_id}")

                    train_frame = raw_frames[context]
                    normalizer.fit_transform_geom_v2(train_frame, artifact_dir, config_id)
                    for split, frame in raw_frames.items():
                        norm_frame = normalizer.transform_geom_v2(frame, artifact_dir, config_id)
                        norm_path = _norm_output_path(
                            config.base_dir, context, split, config_id, config.geometry_root
                        )
                        os.makedirs(os.path.dirname(norm_path), exist_ok=True)
                        norm_frame.to_parquet(norm_path, index=False)
                        output_rows.append({
                            "context": context,
                            "split": split,
                            "config": config_id,
                            "rows": len(norm_frame),
                            "output": norm_path,
                        })
                finally:
                    calculator.index = None
                    del raw_frames
                    gc.collect()

    return pd.DataFrame(output_rows)


if __name__ == "__main__":
    result = run_geometry_v2()
    print(result.to_string(index=False))
