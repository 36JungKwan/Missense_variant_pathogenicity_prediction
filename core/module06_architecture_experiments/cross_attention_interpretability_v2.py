"""Utilities for the staged, resumable TriMVPP interpretability rerun."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cross_attention_explainability import move_batch_to_device, _model_inputs

LABEL_CANDIDATES = ("Pathogenicity_Label", "Label", "label", "target", "Target")
AF_STRATA = ("0", "(0,1e-5]", "(1e-5,1e-4]", ">1e-4")


def find_label_column(frame: pd.DataFrame) -> str:
    for column in LABEL_CANDIDATES:
        if column in frame.columns:
            return column
    raise KeyError(f"No label column found; expected one of {LABEL_CANDIDATES}")


def build_af_stratified_manifest(dataset_frame: pd.DataFrame, raw_metadata: pd.DataFrame,
                                 dataset_name: str, per_cell: int = 25,
                                 seed: int = 2026) -> pd.DataFrame:
    """Select up to per_cell rows per label x raw gnomAD-exome-AF stratum."""
    if "Variant_ID" not in dataset_frame or "Variant_ID" not in raw_metadata:
        raise KeyError("Both dataset and raw metadata must contain Variant_ID")
    if "gnomADe_AF" not in raw_metadata:
        raise KeyError("Raw metadata must contain gnomADe_AF")
    metadata = raw_metadata[["Variant_ID", "gnomADe_AF"]].copy()
    metadata["Variant_ID"] = metadata["Variant_ID"].astype(str)
    if metadata["Variant_ID"].duplicated().any():
        examples = metadata.loc[metadata["Variant_ID"].duplicated(), "Variant_ID"].head(5).tolist()
        raise ValueError(f"Raw AF metadata has duplicate Variant_ID values, examples: {examples}")

    label_column = find_label_column(dataset_frame)
    frame = dataset_frame[["Variant_ID"]].copy().reset_index(drop=True)
    frame["Sample_Index"] = np.arange(len(frame), dtype=np.int64)
    frame["True_Label"] = pd.to_numeric(dataset_frame[label_column], errors="coerce").to_numpy()
    if frame["True_Label"].isna().any():
        raise ValueError(f"Missing/non-numeric labels in {dataset_name}/{label_column}")
    frame["True_Label"] = frame["True_Label"].astype(int)
    frame["Variant_ID"] = frame["Variant_ID"].astype(str)
    frame = frame.merge(metadata, on="Variant_ID", how="left", validate="one_to_one")
    frame["gnomADe_AF"] = pd.to_numeric(frame["gnomADe_AF"], errors="coerce").fillna(0.0)
    frame["AF_Stratum"] = pd.cut(
        frame["gnomADe_AF"],
        bins=[-np.inf, 0.0, 1e-5, 1e-4, np.inf],
        labels=list(AF_STRATA),
        include_lowest=True,
        right=True,
    ).astype(str)
    rng = np.random.default_rng(int(seed))
    selected = []
    for (label, stratum), group in frame.groupby(["True_Label", "AF_Stratum"], sort=True, observed=True):
        candidates = group.index.to_numpy()
        count = min(int(per_cell), len(candidates))
        if count == 0:
            continue
        chosen = rng.choice(candidates, size=count, replace=False)
        part = frame.loc[chosen].copy()
        part["Sampling_Weight"] = float(len(group)) / float(count)
        selected.append(part)
    if not selected:
        raise ValueError(f"No variants selected for {dataset_name}")
    result = pd.concat(selected, ignore_index=True)
    result["Dataset"] = str(dataset_name)
    order = rng.permutation(len(result))
    result = result.iloc[order].reset_index(drop=True)
    result["Selection_Order"] = np.arange(len(result), dtype=np.int64)
    return result[["Dataset", "Sample_Index", "Variant_ID", "True_Label", "gnomADe_AF",
                   "AF_Stratum", "Sampling_Weight", "Selection_Order"]]


def _spread_sample_one_label(frame: pd.DataFrame, count: int,
                             rng: np.random.Generator) -> pd.DataFrame:
    pools = {
        stratum: rng.permutation(group.index.to_numpy()).tolist()
        for stratum, group in frame.groupby("AF_Stratum", sort=False, observed=True)
    }
    selected = []
    strata = list(pools)
    while len(selected) < int(count):
        progressed = False
        for stratum in rng.permutation(strata):
            if pools[stratum] and len(selected) < int(count):
                selected.append(pools[stratum].pop())
                progressed = True
        if not progressed:
            break
    return frame.loc[selected].copy()


def select_spread_subset(manifest: pd.DataFrame, *, per_label: int,
                         seed: int, attention: bool = False) -> pd.DataFrame:
    """Select a fixed count per label while round-robining over AF strata."""
    rng = np.random.default_rng(int(seed))
    selected = []
    for dataset, dataset_frame in manifest.groupby("Dataset", sort=False):
        for label in (0, 1):
            candidates = dataset_frame[dataset_frame["True_Label"].eq(label)]
            part = _spread_sample_one_label(candidates, per_label, rng)
            if len(part) < int(per_label):
                raise ValueError(
                    f"Only {len(part)} samples available for {dataset}, label={label}; "
                    f"requested {per_label}"
                )
            selected.append(part)
    result = pd.concat(selected, ignore_index=True)
    result["For_Attention"] = bool(attention)
    return result.sort_values(["Dataset", "True_Label", "AF_Stratum", "Variant_ID"]).reset_index(drop=True)


def safe_variant_filename(variant_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(variant_id)).strip("_") or "sample"


class PartitionedParquetStore:
    """Write immutable stage chunks for cheap resume; consolidate only at stage end."""
    def __init__(self, output_root: str | Path):
        self.root = Path(output_root) / "parquet_partitions"
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, stage: str, dataset: str, seed: int, chunk_index: int) -> Path:
        return (self.root / str(stage) / str(dataset) / f"seed_{int(seed)}"
                / f"chunk_{int(chunk_index):05d}.parquet")

    def is_complete(self, stage: str, dataset: str, seed: int, chunk_index: int) -> bool:
        return self.path(stage, dataset, seed, chunk_index).is_file()

    def write(self, stage: str, dataset: str, seed: int, chunk_index: int,
              frame: pd.DataFrame) -> Path:
        path = self.path(stage, dataset, seed, chunk_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".tmp.parquet")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        return path

    def read(self, stage: str) -> pd.DataFrame:
        stage_root = self.root / str(stage)
        if not stage_root.is_dir():
            return pd.DataFrame()
        files = sorted(stage_root.rglob("chunk_*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)

    def export_csv(self, stage: str, destination: str | Path) -> pd.DataFrame:
        frame = self.read(stage)
        if not frame.empty:
            frame.to_csv(destination, index=False)
        return frame


def attention_query_summary_rows(attention_maps, *, seed: int, dataset: str,
                                 variant_id: str, true_label: int, token_radius: int,
                                 bio_feature_names, geom_feature_names):
    """Keep head-mean key weights for protein-center and gnomADe_AF queries."""
    key_labels = {
        "dna": [f"DNA:{position:+d}" for position in range(-int(token_radius), int(token_radius) + 1)],
        "protein": [f"Protein:{position:+d}" for position in range(-int(token_radius), int(token_radius) + 1)],
        "bio_geometry": [*map(str, bio_feature_names), *map(str, geom_feature_names)],
    }
    bio_names = list(map(str, bio_feature_names))
    if "gnomADe_AF" not in bio_names:
        raise KeyError("gnomADe_AF is missing from the biological query-token list")
    query_specs = {
        "protein": [(int(token_radius), "Protein:+0")],
        "bio_geometry": [(bio_names.index("gnomADe_AF"), "gnomADe_AF")],
    }
    rows = []
    for name, tensor in attention_maps.items():
        parts = name.split("_")
        layer = int(parts[1])
        query, context = "_".join(parts[2:]).split("_to_", 1)
        if query not in query_specs:
            continue
        weights = tensor.detach().float().cpu().numpy()
        if weights.ndim != 4 or weights.shape[0] != 1:
            raise ValueError(f"Expected [1, heads, queries, keys] attention tensor, got {weights.shape}")
        for query_index, query_label in query_specs[query]:
            if query_index >= weights.shape[2]:
                raise IndexError(f"Query {query_label} is outside attention map {name}: {weights.shape}")
            mean_key_weights = weights[0, :, query_index, :].mean(axis=0)
            contexts = key_labels[context]
            if len(contexts) != len(mean_key_weights):
                contexts = [f"{context}:{index}" for index in range(len(mean_key_weights))]
            for key_label, mean_weight in zip(contexts, mean_key_weights):
                rows.append({
                    "Seed": int(seed), "Dataset": str(dataset), "Variant_ID": str(variant_id),
                    "True_Label": int(true_label), "Layer": layer,
                    "Query_Modality": query, "Query_Token": query_label,
                    "Context_Modality": context, "Context_Token": key_label,
                    "Mean_Attention_Over_Heads": float(mean_weight),
                })
    return rows


def detailed_occlusion_logit_deltas(model, sample_batch, *, device, token_radius,
                                     bio_feature_names, geom_feature_names,
                                     feature_baseline, chunk_size=16):
    """Individual token/feature occlusion plus biologically grouped perturbations."""
    model.eval()
    batch = move_batch_to_device(sample_batch, device)
    original = {
        key: batch[key].to(torch.float32).clone()
        for key in ("token_dna", "token_protein", "bio_features", "geom_features")
        if batch.get(key) is not None
    }

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
            perturbations.append((
                "individual", f"{modality.upper()}_alt_to_ref_{position-int(token_radius):+d}",
                1, values,
            ))
    for key, names in (("bio_features", list(bio_feature_names)),
                       ("geom_features", list(geom_feature_names))):
        if key not in original:
            continue
        median = feature_baseline[key].to(device=device, dtype=torch.float32).reshape(-1)
        for index, name in enumerate(names):
            values = {field: tensor.clone() for field, tensor in original.items()}
            values[key][:, index] = median[index]
            perturbations.append(("individual", str(name), 1, values))

    bio_names = list(map(str, bio_feature_names))
    geom_names = list(map(str, geom_feature_names))
    grouped_features = [
        ("AF", [name for name in ("AF", "gnomADe_AF") if name in bio_names], "bio_features", bio_names),
        ("Conservation", [name for name in bio_names
                          if name not in {"AF", "gnomADe_AF"}
                          and not name.startswith("SpliceAI_")],
         "bio_features", bio_names),
        ("SpliceAI", [name for name in bio_names if name.startswith("SpliceAI_")], "bio_features", bio_names),
        ("Geometry", geom_names, "geom_features", geom_names),
    ]
    for group_name, names, key, columns in grouped_features:
        if not names or key not in original:
            continue
        indices = [columns.index(name) for name in names]
        values = {field: tensor.clone() for field, tensor in original.items()}
        median = feature_baseline[key].to(device=device, dtype=torch.float32).reshape(-1)
        values[key][:, indices] = median[indices].view(1, -1)
        perturbations.append(("group", group_name, len(indices), values))

    center = int(token_radius)
    protein_center = {field: tensor.clone() for field, tensor in original.items()}
    protein_center["token_protein"][:, 1, center] = protein_center["token_protein"][:, 0, center]
    perturbations.append(("group", "Protein_variant_center_alt_to_ref", 1, protein_center))

    protein_neighbors = {field: tensor.clone() for field, tensor in original.items()}
    neighbor_indices = [index for index in range(2 * center + 1) if index != center]
    protein_neighbors["token_protein"][:, 1, neighbor_indices] = protein_neighbors["token_protein"][:, 0, neighbor_indices]
    perturbations.append(("group", "Protein_neighbor_tokens_alt_to_ref", len(neighbor_indices), protein_neighbors))

    dna_window = {field: tensor.clone() for field, tensor in original.items()}
    dna_window["token_dna"][:, 1, :] = dna_window["token_dna"][:, 0, :]
    perturbations.append(("group", "DNA_window_alt_to_ref", 2 * center + 1, dna_window))

    rows = []
    step = max(1, int(chunk_size))
    for start in range(0, len(perturbations), step):
        chunk = perturbations[start:start + step]
        count = len(chunk)
        fixed = dict(batch)
        for key, value in list(fixed.items()):
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
                fixed[key] = value.expand((count,) + tuple(value.shape[1:]))
        stacked = {
            key: torch.cat([values[key] for _, _, _, values in chunk], dim=0)
            for key in original
        }
        with torch.no_grad():
            perturbed_logits = _model_inputs(model, fixed, stacked, return_attention=False).reshape(-1)
        for (level, name, number_changed, _), logit in zip(chunk, perturbed_logits):
            occluded_logit = float(logit.item())
            rows.append({
                "Occlusion_Level": level,
                "Feature_or_Position": name,
                "Number_Features_or_Tokens_Changed": int(number_changed),
                "Actual_Logit": actual_logit,
                "Occluded_Logit": occluded_logit,
                "Logit_Delta_Actual_Minus_Occluded": actual_logit - occluded_logit,
            })
    return rows


class CrossAttentionInterpretabilityExperiment:
    """No-training A/D/B/C/E analysis with per-seed/dataset resumable chunks."""

    def __init__(self, *, output_root, store, seeds, datasets_to_explain,
                 sample_manifest, attention_manifest, sensitivity_manifest,
                 source_info, model_config, bio_feature_names, geom_feature_names,
                 device, make_loader, load_seed_model, background_baselines,
                 train_feature_medians, af_zero_values, af_median_values,
                 ig_steps=64, ig_retry_steps=128, ig_internal_batch_size=8,
                 ig_chunk_size=5, sensitivity_chunk_size=4,
                 occlusion_chunk_size=16, attention_chunk_size=4,
                 mcc_reference=None, mcc_tolerance=0.002,
                 allow_low_background_stability=False):
        self.output_root = Path(output_root)
        self.store = store
        self.seeds = list(map(int, seeds))
        self.datasets_to_explain = dict(datasets_to_explain)
        self.dataset_to_split = {
            str(name): str(split) for split, name in self.datasets_to_explain.items()
        }
        self.sample_manifest = sample_manifest.copy()
        self.attention_manifest = attention_manifest.copy()
        self.sensitivity_manifest = sensitivity_manifest.copy()
        self.source_info = source_info
        self.model_config = dict(model_config)
        self.bio_feature_names = list(map(str, bio_feature_names))
        self.geom_feature_names = list(map(str, geom_feature_names))
        self.device = device
        self.make_loader = make_loader
        self.load_seed_model = load_seed_model
        self.background_baselines = background_baselines
        self.train_feature_medians = train_feature_medians
        self.af_zero_values = np.asarray(af_zero_values, dtype=np.float32).reshape(-1)
        self.af_median_values = np.asarray(af_median_values, dtype=np.float32).reshape(-1)
        self.ig_steps = int(ig_steps)
        self.ig_retry_steps = int(ig_retry_steps)
        self.ig_internal_batch_size = int(ig_internal_batch_size)
        self.ig_chunk_size = int(ig_chunk_size)
        self.sensitivity_chunk_size = int(sensitivity_chunk_size)
        self.occlusion_chunk_size = int(occlusion_chunk_size)
        self.attention_chunk_size = int(attention_chunk_size)
        self.mcc_reference = dict(mcc_reference or {})
        self.mcc_tolerance = float(mcc_tolerance)
        self.allow_low_background_stability = bool(allow_low_background_stability)
        self._loader_cache = {}
        self.prediction_metrics = pd.DataFrame()
        self.background_stability = pd.DataFrame()

    def _loader(self, dataset_name):
        dataset_name = str(dataset_name)
        if dataset_name not in self._loader_cache:
            split_name = self.dataset_to_split[dataset_name]
            self._loader_cache[dataset_name] = self.make_loader(split_name, self.seeds[0])
        return self._loader_cache[dataset_name]

    def _batch_inputs(self, batch):
        moved = move_batch_to_device(batch, self.device)
        values = {
            key: moved[key].to(torch.float32)
            for key in ("token_dna", "token_protein", "bio_features", "geom_features")
            if moved.get(key) is not None
        }
        return moved, values

    def _sample_record(self, seed, dataset_name, sample_row):
        source = self.source_info[int(seed)]
        return {
            "Seed": int(seed), "Dataset": str(dataset_name),
            "Variant_ID": str(sample_row["Variant_ID"]),
            "True_Label": int(sample_row["True_Label"]),
            "AF_Stratum": str(sample_row["AF_Stratum"]),
            "Sampling_Weight": float(sample_row["Sampling_Weight"]),
            "Checkpoint": source["checkpoint"], "Best_Epoch": source["best_epoch"],
            "Val_MCC": source["val_mcc"],
            "Decision_Threshold": source["threshold"],
        }

    @staticmethod
    def _chunk_rows(frame, chunk_size):
        ordered = frame.sort_values("Selection_Order").reset_index(drop=True)
        for chunk_index, start in enumerate(range(0, len(ordered), int(chunk_size))):
            yield chunk_index, ordered.iloc[start:start + int(chunk_size)]

    def run_predictions(self):
        """Stage A: predict every row on each held-out dataset and verify MCC."""
        from sklearn.metrics import matthews_corrcoef

        for dataset_name in self.datasets_to_explain.values():
            loader = self._loader(dataset_name)
            for seed in self.seeds:
                if all(self.store.is_complete("predictions", dataset_name, seed, i)
                       for i in range(len(loader))):
                    print(f"[RESUME] predictions {dataset_name} seed={seed}: complete")
                    continue
                model = self.load_seed_model(seed, loader.dataset)
                for chunk_index, batch in enumerate(loader):
                    if self.store.is_complete("predictions", dataset_name, seed, chunk_index):
                        continue
                    moved, values = self._batch_inputs(batch)
                    with torch.no_grad():
                        logits = _model_inputs(model, moved, values, return_attention=False).reshape(-1)
                    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
                    labels = moved["label"].reshape(-1).detach().cpu().numpy().astype(int)
                    ids = [str(value) for value in moved["variant_id"]]
                    threshold = float(self.source_info[seed]["threshold"])
                    frame = pd.DataFrame({
                        "Seed": seed, "Dataset": str(dataset_name), "Variant_ID": ids,
                        "True_Label": labels, "Logit": logits.detach().cpu().numpy(),
                        "Probability_Pathogenic": probabilities,
                        "Decision_Threshold": threshold,
                        "Predicted_Label": (probabilities >= threshold).astype(int),
                        "Checkpoint": self.source_info[seed]["checkpoint"],
                    })
                    self.store.write("predictions", dataset_name, seed, chunk_index, frame)
                    print(f"[A] {dataset_name} seed={seed}: batch {chunk_index+1}/{len(loader)} saved")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        predictions = self.store.export_csv(
            "predictions", self.output_root / "predictions_all.csv"
        )
        metric_rows = []
        for (seed, dataset_name), frame in predictions.groupby(["Seed", "Dataset"], sort=True):
            metric = float(matthews_corrcoef(frame["True_Label"], frame["Predicted_Label"]))
            expected = self.mcc_reference.get(int(seed)) if dataset_name == "Train3Val_Test" else None
            difference = abs(metric - float(expected)) if expected is not None else np.nan
            metric_rows.append({
                "Seed": int(seed), "Dataset": str(dataset_name), "N": len(frame), "MCC": metric,
                "Reference_MCC": expected, "Absolute_Difference": difference,
                "MCC_Match_Within_Tolerance": (difference <= self.mcc_tolerance)
                if expected is not None else np.nan,
                "Decision_Threshold": float(frame["Decision_Threshold"].iloc[0]),
            })
        self.prediction_metrics = pd.DataFrame(metric_rows)
        self.prediction_metrics.to_csv(
            self.output_root / "prediction_metrics_by_seed.csv", index=False
        )
        print("[A] Recomputed MCC and reference check:")
        try:
            from IPython.display import display
            display(self.prediction_metrics)
        except Exception:
            print(self.prediction_metrics.to_string(index=False))
        return predictions, self.prediction_metrics

    def run_af_interventions(self):
        """Stage D: change only AF and gnomADe_AF on all rows and seeds."""
        from sklearn.metrics import (
            accuracy_score, average_precision_score, matthews_corrcoef, roc_auc_score,
        )

        af_indices = [self.bio_feature_names.index(name) for name in ("AF", "gnomADe_AF")]
        if len(self.af_zero_values) != 2 or len(self.af_median_values) != 2:
            raise ValueError("AF intervention values must be [AF, gnomADe_AF]")
        for dataset_name in self.datasets_to_explain.values():
            loader = self._loader(dataset_name)
            for seed in self.seeds:
                if all(self.store.is_complete("af_intervention", dataset_name, seed, i)
                       for i in range(len(loader))):
                    print(f"[RESUME] AF interventions {dataset_name} seed={seed}: complete")
                    continue
                model = self.load_seed_model(seed, loader.dataset)
                for chunk_index, batch in enumerate(loader):
                    if self.store.is_complete("af_intervention", dataset_name, seed, chunk_index):
                        continue
                    moved, base_values = self._batch_inputs(batch)
                    labels = moved["label"].reshape(-1).detach().cpu().numpy().astype(int)
                    ids = [str(value) for value in moved["variant_id"]]
                    threshold = float(self.source_info[seed]["threshold"])
                    output_rows = []
                    for scenario in ("original", "af_zero", "af_median"):
                        values = {key: tensor.clone() for key, tensor in base_values.items()}
                        if scenario == "af_zero":
                            replacement = self.af_zero_values
                        elif scenario == "af_median":
                            replacement = self.af_median_values
                        else:
                            replacement = None
                        if replacement is not None:
                            values["bio_features"][:, af_indices] = torch.as_tensor(
                                replacement, device=self.device, dtype=torch.float32
                            ).view(1, -1)
                        with torch.no_grad():
                            logits = _model_inputs(model, moved, values, return_attention=False).reshape(-1)
                        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
                        predicted = (probabilities >= threshold).astype(int)
                        for variant_id, label, logit, probability, prediction in zip(
                            ids, labels, logits.detach().cpu().numpy(), probabilities, predicted
                        ):
                            output_rows.append({
                                "Seed": seed, "Dataset": str(dataset_name),
                                "Variant_ID": variant_id, "True_Label": int(label),
                                "Scenario": scenario, "Logit": float(logit),
                                "Probability_Pathogenic": float(probability),
                                "Predicted_Label": int(prediction),
                                "Decision_Threshold": threshold,
                                "Checkpoint": self.source_info[seed]["checkpoint"],
                            })
                    self.store.write(
                        "af_intervention", dataset_name, seed, chunk_index,
                        pd.DataFrame(output_rows),
                    )
                    print(f"[D] {dataset_name} seed={seed}: batch {chunk_index+1}/{len(loader)} saved")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        frame = self.store.export_csv(
            "af_intervention", self.output_root / "af_intervention.csv"
        )
        key_columns = ["Seed", "Dataset", "Variant_ID"]
        original = frame[frame["Scenario"].eq("original")][
            key_columns + ["Logit", "Predicted_Label"]
        ].rename(columns={
            "Logit": "Original_Logit",
            "Predicted_Label": "Original_Predicted_Label",
        })
        metric_rows = []
        for (seed, dataset_name, scenario), group in frame.groupby(
            ["Seed", "Dataset", "Scenario"], sort=True
        ):
            comparison = group.merge(
                original, on=key_columns, how="left", validate="one_to_one"
            )
            if comparison["Original_Logit"].isna().any():
                raise ValueError(
                    f"Could not align {scenario} AF rows to original predictions for "
                    f"{dataset_name}, seed={seed}"
                )
            labels = comparison["True_Label"].astype(int)
            predictions = comparison["Predicted_Label"].astype(int)
            probabilities = comparison["Probability_Pathogenic"].astype(float)
            try:
                auroc = float(roc_auc_score(labels, probabilities))
            except ValueError:
                auroc = np.nan
            try:
                auprc = float(average_precision_score(labels, probabilities))
            except ValueError:
                auprc = np.nan
            metric_rows.append({
                "Seed": int(seed), "Dataset": str(dataset_name), "Scenario": str(scenario),
                "N": len(group),
                "MCC": float(matthews_corrcoef(labels, predictions)),
                "Accuracy": float(accuracy_score(labels, predictions)),
                "AUROC": auroc,
                "AUPRC": auprc,
                "Mean_Absolute_Logit_Change_From_Original": float(
                    np.mean(np.abs(
                        comparison["Logit"].to_numpy()
                        - comparison["Original_Logit"].to_numpy()
                    ))
                ),
                "Prediction_Flip_Rate_From_Original": float(np.mean(
                    predictions.to_numpy()
                    != comparison["Original_Predicted_Label"].astype(int).to_numpy()
                )),
                "Decision_Threshold": float(group["Decision_Threshold"].iloc[0]),
            })
        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(self.output_root / "af_intervention_metrics.csv", index=False)
        return frame, metrics

    def _ig_result_rows(self, model, sample, sample_row, seed, dataset_name, mode,
                        baseline, background_seed):
        from .cross_attention_explainability import integrated_gradients, alt_equals_ref_baseline

        base = self._sample_record(seed, dataset_name, sample_row)
        if mode == "alt_vs_ref":
            baseline = alt_equals_ref_baseline(sample)
            background_seed_value = np.nan
            baseline_type = "alt_equals_ref"
        else:
            background_seed_value = int(background_seed)
            baseline_type = "empirical_train_background"

        def run_steps(step_count):
            return integrated_gradients(
                model, sample, baseline, device=self.device, n_steps=int(step_count),
                internal_batch_size=self.ig_internal_batch_size,
            )

        result = run_steps(self.ig_steps)

        def relative_residual(value):
            denominator = abs(float(value["actual_logit"]) - float(value["baseline_logit"]))
            residual = abs(float(value["completeness_residual"]))
            if denominator <= 1e-8:
                return 0.0 if residual <= 1e-8 else float("inf")
            return residual / denominator

        steps_used = self.ig_steps
        residual_ratio = relative_residual(result)
        if residual_ratio > 0.05 and self.ig_retry_steps > self.ig_steps:
            result = run_steps(self.ig_retry_steps)
            steps_used = self.ig_retry_steps
            residual_ratio = relative_residual(result)
        common = {
            **base, "Attribution_Mode": str(mode),
            "Background_Seed": background_seed_value, "Baseline_Type": baseline_type,
            "IG_Steps_Used": int(steps_used),
            "IG_Relative_Residual": float(residual_ratio),
            "IG_Residual_Over_5pct": bool(residual_ratio > 0.05),
        }
        completeness = {
            **common,
            "Actual_Pathogenic_Logit": float(result["actual_logit"]),
            "Baseline_Pathogenic_Logit": float(result["baseline_logit"]),
            "IG_Completeness_Residual": float(result["completeness_residual"]),
        }
        modality_rows, position_rows, feature_rows = [], [], []
        feature_names_by_key = {
            "bio_features": self.bio_feature_names,
            "geom_features": self.geom_feature_names,
        }
        for key, attribution in result["attributions"].items():
            modality = {
                "token_dna": "DNA", "token_protein": "Protein",
                "bio_features": "Bio_SpliceAI", "geom_features": "Geometry",
            }[key]
            modality_rows.append({
                **common, "Modality": modality,
                "Signed_Attribution_Sum": float(attribution.sum().item()),
                "Absolute_Attribution_Sum": float(attribution.abs().sum().item()),
            })
            if key in ("token_dna", "token_protein"):
                values = attribution[0].numpy()
                for allele_index, allele in enumerate(("Ref", "Alt")):
                    for position in range(values.shape[1]):
                        position_rows.append({
                            **common, "Modality": modality, "Allele": allele,
                            "Relative_Token_Position": position - int(self.model_config["k_radius"]),
                            "Signed_Attribution": float(values[allele_index, position].sum()),
                            "Absolute_Attribution": float(np.abs(values[allele_index, position]).sum()),
                        })
            else:
                values = attribution[0].numpy().reshape(-1)
                for feature_index, feature_name in enumerate(feature_names_by_key[key]):
                    feature_rows.append({
                        **common, "Feature": str(feature_name),
                        "Input_Value": float(sample[key].reshape(-1)[feature_index].item()),
                        "Signed_Attribution": float(values[feature_index]),
                        "Absolute_Attribution": float(abs(values[feature_index])),
                    })
        return modality_rows, position_rows, feature_rows, completeness

    def run_ig(self):
        """Stage B: AF-stratified IG for both baselines, with adaptive 64/128 steps."""
        stages = ("ig_modalities", "ig_positions", "ig_features", "ig_completeness")
        for dataset_name, rows in self.sample_manifest.groupby("Dataset", sort=False):
            loader = self._loader(dataset_name)
            for seed in self.seeds:
                model = None
                for chunk_index, sample_rows in self._chunk_rows(rows, self.ig_chunk_size):
                    if all(self.store.is_complete(stage, dataset_name, seed, chunk_index)
                           for stage in stages):
                        continue
                    if model is None:
                        model = self.load_seed_model(seed, loader.dataset)
                    outputs = {stage: [] for stage in stages}
                    for _, sample_row in sample_rows.iterrows():
                        sample = loader.dataset[int(sample_row["Sample_Index"])]
                        for mode in ("full_prediction", "alt_vs_ref"):
                            bg_seed = 2026 if mode == "full_prediction" else None
                            baseline = self.background_baselines[2026] if mode == "full_prediction" else None
                            modality_rows, position_rows, feature_rows, completeness = self._ig_result_rows(
                                model, sample, sample_row, seed, dataset_name,
                                mode, baseline, bg_seed,
                            )
                            outputs["ig_modalities"].extend(modality_rows)
                            outputs["ig_positions"].extend(position_rows)
                            outputs["ig_features"].extend(feature_rows)
                            outputs["ig_completeness"].append(completeness)
                    for stage, result_rows in outputs.items():
                        self.store.write(
                            stage, dataset_name, seed, chunk_index, pd.DataFrame(result_rows)
                        )
                    over = sum(bool(row["IG_Residual_Over_5pct"])
                               for row in outputs["ig_completeness"])
                    total = len(outputs["ig_completeness"])
                    print(f"[B] {dataset_name} seed={seed} chunk={chunk_index}: "
                          f"{len(sample_rows)} variants x 2 baselines; "
                          f"{over}/{total} residuals >5%")
                if model is not None:
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        names = {
            "ig_modalities": "ig_modality_attributions.csv",
            "ig_positions": "ig_token_positions.csv",
            "ig_features": "ig_tabular_features.csv",
            "ig_completeness": "ig_completeness.csv",
        }
        frames = {stage: self.store.export_csv(stage, self.output_root / filename)
                  for stage, filename in names.items()}
        completeness = frames["ig_completeness"]
        if not completeness.empty:
            over = completeness["IG_Residual_Over_5pct"].astype(bool).sum()
            print(f"[B] residuals still >5% after adaptive 64/128 steps: {int(over)}/{len(completeness)}")
        return frames

    def _sensitivity_unit_values(self, background_seed):
        background_seed = int(background_seed)
        if background_seed == 2026:
            feature_frame = self.store.read("ig_features")
            position_frame = self.store.read("ig_positions")
        else:
            feature_frame = self.store.read(f"ig_sens_{background_seed}_features")
            position_frame = self.store.read(f"ig_sens_{background_seed}_positions")
        frames = []
        if not feature_frame.empty:
            feature_frame = feature_frame[
                feature_frame["Attribution_Mode"].eq("full_prediction")
                & pd.to_numeric(feature_frame["Background_Seed"], errors="coerce").eq(background_seed)
            ].copy()
            feature_frame["Unit"] = "FEATURE:" + feature_frame["Feature"].astype(str)
            feature_frame["Importance"] = feature_frame["Absolute_Attribution"]
            frames.append(feature_frame)
        if not position_frame.empty:
            position_frame = position_frame[
                position_frame["Attribution_Mode"].eq("full_prediction")
                & pd.to_numeric(position_frame["Background_Seed"], errors="coerce").eq(background_seed)
            ].copy()
            position_frame["Unit"] = (
                "TOKEN:" + position_frame["Modality"].astype(str) + ":"
                + position_frame["Allele"].astype(str) + ":"
                + position_frame["Relative_Token_Position"].map(lambda value: f"{int(value):+d}")
            )
            position_frame["Importance"] = position_frame["Absolute_Attribution"]
            frames.append(position_frame)
        if not frames:
            return pd.DataFrame(columns=["Dataset", "Seed", "Variant_ID", "Unit", "Importance"])
        return pd.concat(frames, ignore_index=True)

    def run_ig_background_sensitivity(self):
        """Run full-prediction IG on 60 variants with two additional backgrounds."""
        stage_names = {}
        for background_seed in (2027, 2028):
            names = {
                "modalities": f"ig_sens_{background_seed}_modalities",
                "positions": f"ig_sens_{background_seed}_positions",
                "features": f"ig_sens_{background_seed}_features",
                "completeness": f"ig_sens_{background_seed}_completeness",
            }
            stage_names[background_seed] = names
            for dataset_name, rows in self.sensitivity_manifest.groupby("Dataset", sort=False):
                loader = self._loader(dataset_name)
                for seed in self.seeds:
                    model = None
                    for chunk_index, sample_rows in self._chunk_rows(
                        rows, self.sensitivity_chunk_size
                    ):
                        if all(self.store.is_complete(stage, dataset_name, seed, chunk_index)
                               for stage in names.values()):
                            continue
                        if model is None:
                            model = self.load_seed_model(seed, loader.dataset)
                        outputs = {stage: [] for stage in names.values()}
                        for _, sample_row in sample_rows.iterrows():
                            sample = loader.dataset[int(sample_row["Sample_Index"])]
                            modality_rows, position_rows, feature_rows, completeness = self._ig_result_rows(
                                model, sample, sample_row, seed, dataset_name,
                                "full_prediction", self.background_baselines[background_seed],
                                background_seed,
                            )
                            for row in modality_rows + position_rows + feature_rows:
                                row["Sensitivity_Background_Run"] = True
                            completeness["Sensitivity_Background_Run"] = True
                            outputs[names["modalities"]].extend(modality_rows)
                            outputs[names["positions"]].extend(position_rows)
                            outputs[names["features"]].extend(feature_rows)
                            outputs[names["completeness"]].append(completeness)
                        for stage, result in outputs.items():
                            self.store.write(stage, dataset_name, seed, chunk_index, pd.DataFrame(result))
                        print(f"[B-sensitivity] background={background_seed} "
                              f"{dataset_name} seed={seed} chunk={chunk_index} saved")
                    if model is not None:
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            for key, stage in names.items():
                self.store.export_csv(
                    stage, self.output_root / f"ig_sensitivity_bg{background_seed}_{key}.csv"
                )

        all_backgrounds = {2026: self._sensitivity_unit_values(2026)}
        all_backgrounds.update({
            bg: self._sensitivity_unit_values(bg) for bg in (2027, 2028)
        })
        stability_rows = []
        for dataset_name in self.datasets_to_explain.values():
            selected_ids = set(self.sensitivity_manifest.loc[
                self.sensitivity_manifest["Dataset"].eq(dataset_name), "Variant_ID"
            ].astype(str))
            for seed in self.seeds:
                ranked = {}
                for bg_seed, frame in all_backgrounds.items():
                    subset = frame[
                        frame["Dataset"].astype(str).eq(str(dataset_name))
                        & pd.to_numeric(frame["Seed"], errors="coerce").eq(int(seed))
                        & frame["Variant_ID"].astype(str).isin(selected_ids)
                    ]
                    ranked[bg_seed] = subset.groupby("Unit")["Importance"].mean().sort_index()
                for comparison_seed in (2027, 2028):
                    pair = pd.concat(
                        [ranked[2026].rename("primary"),
                         ranked[comparison_seed].rename("comparison")],
                        axis=1,
                    ).dropna()
                    correlation = (
                        float(pair["primary"].rank().corr(pair["comparison"].rank()))
                        if len(pair) > 1 else np.nan
                    )
                    stability_rows.append({
                        "Dataset": str(dataset_name), "Seed": int(seed),
                        "Primary_Background_Seed": 2026,
                        "Comparison_Background_Seed": int(comparison_seed),
                        "Spearman_Rank_Correlation": correlation, "N_Units": len(pair),
                        "Pass_0_8": bool(correlation >= 0.8)
                        if np.isfinite(correlation) else False,
                    })
        self.background_stability = pd.DataFrame(stability_rows)
        self.background_stability.to_csv(
            self.output_root / "ig_background_stability.csv", index=False
        )
        print("[B-sensitivity] Feature/token importance rank stability:")
        try:
            from IPython.display import display
            display(self.background_stability)
        except Exception:
            print(self.background_stability.to_string(index=False))
        failed = self.background_stability[~self.background_stability["Pass_0_8"]]
        if not failed.empty:
            print(f"[GATE] {len(failed)} comparisons have Spearman <0.8; "
                  "review ig_background_stability.csv before C/E.")
        else:
            print("[GATE] All background comparisons have Spearman >=0.8.")
        return self.background_stability

    def _require_background_stability(self):
        if self.background_stability.empty:
            path = self.output_root / "ig_background_stability.csv"
            if path.is_file():
                self.background_stability = pd.read_csv(path)
        if self.background_stability.empty:
            raise RuntimeError("Run the IG background-stability cell before occlusion or attention.")
        failed = self.background_stability[
            ~self.background_stability["Pass_0_8"].astype(bool)
        ]
        if not failed.empty and not self.allow_low_background_stability:
            raise RuntimeError(
                f"IG background sensitivity is below 0.8 for {len(failed)} comparisons. "
                "Review ig_background_stability.csv, then explicitly enable the notebook override."
            )

    def run_occlusion(self):
        """Stage C: individual and grouped occlusion over the IG sample manifest."""
        self._require_background_stability()
        stages = ("occlusion_individual", "occlusion_groups")
        for dataset_name, rows in self.sample_manifest.groupby("Dataset", sort=False):
            loader = self._loader(dataset_name)
            for seed in self.seeds:
                model = None
                for chunk_index, sample_rows in self._chunk_rows(rows, self.ig_chunk_size):
                    if all(self.store.is_complete(stage, dataset_name, seed, chunk_index)
                           for stage in stages):
                        continue
                    if model is None:
                        model = self.load_seed_model(seed, loader.dataset)
                    individual_rows, group_rows = [], []
                    for _, sample_row in sample_rows.iterrows():
                        sample = loader.dataset[int(sample_row["Sample_Index"])]
                        results = detailed_occlusion_logit_deltas(
                            model, sample, device=self.device,
                            token_radius=int(self.model_config["k_radius"]),
                            bio_feature_names=self.bio_feature_names,
                            geom_feature_names=self.geom_feature_names,
                            feature_baseline=self.train_feature_medians,
                            chunk_size=self.occlusion_chunk_size,
                        )
                        base = self._sample_record(seed, dataset_name, sample_row)
                        for result in results:
                            result.update(base)
                            if result["Occlusion_Level"] == "group":
                                group_rows.append(result)
                            else:
                                individual_rows.append(result)
                    self.store.write(
                        stages[0], dataset_name, seed, chunk_index, pd.DataFrame(individual_rows)
                    )
                    self.store.write(
                        stages[1], dataset_name, seed, chunk_index, pd.DataFrame(group_rows)
                    )
                    print(f"[C] {dataset_name} seed={seed} chunk={chunk_index}: "
                          f"{len(sample_rows)} variants, {len(individual_rows)} individual + "
                          f"{len(group_rows)} grouped rows")
                if model is not None:
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        individual = self.store.export_csv(
            "occlusion_individual", self.output_root / "occlusion_individual.csv"
        )
        grouped = self.store.export_csv(
            "occlusion_groups", self.output_root / "occlusion_groups.csv"
        )
        return individual, grouped

    def run_attention(self):
        """Stage E: save full maps as float16; summarize protein-center and AF queries."""
        self._require_background_stability()
        for dataset_name, rows in self.attention_manifest.groupby("Dataset", sort=False):
            loader = self._loader(dataset_name)
            for seed in self.seeds:
                model = None
                for chunk_index, sample_rows in self._chunk_rows(
                    rows, self.attention_chunk_size
                ):
                    if self.store.is_complete("attention_summary", dataset_name, seed, chunk_index):
                        continue
                    if model is None:
                        model = self.load_seed_model(seed, loader.dataset)
                    summary_rows = []
                    for _, sample_row in sample_rows.iterrows():
                        sample = loader.dataset[int(sample_row["Sample_Index"])]
                        variant_id = str(sample_row["Variant_ID"])
                        map_dir = self.output_root / "attention_maps" / dataset_name / f"seed_{seed}"
                        map_dir.mkdir(parents=True, exist_ok=True)
                        map_path = map_dir / f"{safe_variant_filename(variant_id)}.npz"
                        if map_path.is_file():
                            with np.load(map_path) as archive:
                                attention_maps = {
                                    key: torch.from_numpy(
                                        np.asarray(archive[key], dtype=np.float32)
                                    )
                                    for key in archive.files
                                }
                        else:
                            moved, values = self._batch_inputs(sample)
                            with torch.no_grad():
                                _, attention_maps = model(
                                    moved["v_dna"], moved["v_prot"], values["bio_features"],
                                    values.get("geom_features"),
                                    token_dna=values["token_dna"],
                                    token_dna_mask=moved["token_dna_mask"],
                                    token_dna_variant_index=moved["token_dna_variant_index"],
                                    token_protein=values["token_protein"],
                                    token_protein_mask=moved["token_protein_mask"],
                                    token_protein_variant_index=moved["token_protein_variant_index"],
                                    return_attention=True,
                                )
                            temporary = map_path.with_name(map_path.stem + ".tmp.npz")
                            np.savez_compressed(
                                temporary,
                                **{
                                    key: tensor.detach().cpu().numpy().astype(np.float16)
                                    for key, tensor in attention_maps.items()
                                },
                            )
                            temporary.replace(map_path)
                        results = attention_query_summary_rows(
                            attention_maps, seed=seed, dataset=dataset_name,
                            variant_id=variant_id,
                            true_label=int(sample_row["True_Label"]),
                            token_radius=int(self.model_config["k_radius"]),
                            bio_feature_names=self.bio_feature_names,
                            geom_feature_names=self.geom_feature_names,
                        )
                        base = self._sample_record(seed, dataset_name, sample_row)
                        for row in results:
                            row.update({
                                "AF_Stratum": base["AF_Stratum"],
                                "Sampling_Weight": base["Sampling_Weight"],
                                "Checkpoint": base["Checkpoint"],
                                "Best_Epoch": base["Best_Epoch"],
                                "Val_MCC": base["Val_MCC"],
                                "Decision_Threshold": base["Decision_Threshold"],
                            })
                        summary_rows.extend(results)
                    self.store.write(
                        "attention_summary", dataset_name, seed, chunk_index,
                        pd.DataFrame(summary_rows),
                    )
                    print(f"[E] {dataset_name} seed={seed} chunk={chunk_index}: "
                          f"{len(sample_rows)} variants; raw maps saved as float16")
                if model is not None:
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        return self.store.export_csv(
            "attention_summary", self.output_root / "attention_summary.csv"
        )
