"""SHAP explainers for the fixed TriMVPP three-modality Cross-Attention model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .cross_attention_explainability import move_batch_to_device
from .cross_attention_interpretability_v2 import safe_variant_filename


INPUT_KEYS = ("token_dna", "token_protein", "bio_features", "geom_features")
INPUT_TO_MODALITY = {
    "token_dna": "DNA",
    "token_protein": "Protein",
    "bio_features": "Bio_SpliceAI",
    "geom_features": "Geometry",
}
EXACT_GROUPS = tuple(INPUT_TO_MODALITY.items())


def _expand_fixed_batch(fixed_batch, batch_size):
    result = dict(fixed_batch)
    for key, value in list(result.items()):
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
            result[key] = value.expand((int(batch_size),) + tuple(value.shape[1:]))
    return result


class FixedContextInputModel(nn.Module):
    """Expose four differentiable tensors while fixing masks/metadata for one sample."""

    def __init__(self, model, fixed_batch):
        super().__init__()
        self.model = model
        self.fixed_batch = fixed_batch

    def forward(self, token_dna, token_protein, bio_features, geom_features):
        batch_size = int(token_dna.shape[0])
        fixed = _expand_fixed_batch(self.fixed_batch, batch_size)
        return self.model(
            fixed["v_dna"], fixed["v_prot"], bio_features, geom_features,
            token_dna=token_dna,
            token_dna_mask=fixed["token_dna_mask"],
            token_dna_variant_index=fixed["token_dna_variant_index"],
            token_protein=token_protein,
            token_protein_mask=fixed["token_protein_mask"],
            token_protein_variant_index=fixed["token_protein_variant_index"],
        ).reshape(-1, 1)


class ExactModalityCoalitionModel:
    """Model a 4-player coalition game with one fixed input/background pair."""

    def __init__(self, context_model, current_inputs, baseline_inputs, device):
        self.context_model = context_model
        self.current_inputs = current_inputs
        self.baseline_inputs = baseline_inputs
        self.device = device

    def __call__(self, coalition_rows):
        coalition_rows = np.asarray(coalition_rows, dtype=np.float32)
        if coalition_rows.ndim == 1:
            coalition_rows = coalition_rows[None, :]
        if coalition_rows.shape[1] != len(EXACT_GROUPS):
            raise ValueError(f"Exact SHAP expects {len(EXACT_GROUPS)} modality groups")
        batch_size = coalition_rows.shape[0]
        selectors = torch.as_tensor(coalition_rows, device=self.device, dtype=torch.float32)
        values = {}
        for group_index, (key, _) in enumerate(EXACT_GROUPS):
            current = self.current_inputs[key].expand(
                (batch_size,) + tuple(self.current_inputs[key].shape[1:])
            )
            baseline = self.baseline_inputs[key].expand(
                (batch_size,) + tuple(self.baseline_inputs[key].shape[1:])
            )
            shape = (batch_size,) + (1,) * (current.ndim - 1)
            values[key] = torch.where(
                selectors[:, group_index].view(shape).bool(), current, baseline
            )
        with torch.no_grad():
            output = self.context_model(
                values["token_dna"], values["token_protein"],
                values["bio_features"], values["geom_features"],
            )
        return output.detach().cpu().numpy().reshape(-1)


def _normalize_shap_values(values, input_tensors):
    if isinstance(values, tuple):
        values = values[0]
    if not isinstance(values, (list, tuple)):
        if len(input_tensors) != 1:
            raise TypeError("Expected one SHAP array per model input")
        values = [values]
    if len(values) != len(input_tensors):
        raise ValueError(
            f"SHAP returned {len(values)} input arrays for {len(input_tensors)} inputs"
        )
    normalized = []
    for value, input_tensor in zip(values, input_tensors):
        array = np.asarray(value)
        expected_shape = tuple(input_tensor.shape)
        if array.ndim == len(expected_shape) + 1 and array.shape[-1] == 1:
            array = array[..., 0]
        if tuple(array.shape) != expected_shape:
            raise ValueError(
                f"Unexpected SHAP shape {array.shape}; expected input shape {expected_shape}"
            )
        normalized.append(array.astype(np.float32, copy=False))
    return normalized


class CrossAttentionShapExperiment:
    """Run Exact (grouped), Gradient, and Deep SHAP with resumable chunk files."""

    def __init__(self, *, output_root, store, seeds, datasets_to_explain,
                 sample_manifest, source_info, model_config, bio_feature_names,
                 geom_feature_names, device, make_loader, load_seed_model,
                 background_inputs, exact_baseline_inputs, gradient_nsamples=128,
                 gradient_batch_size=16, chunk_size=4, deep_additivity_fallback=True):
        self.output_root = Path(output_root)
        self.store = store
        self.seeds = list(map(int, seeds))
        self.datasets_to_explain = dict(datasets_to_explain)
        self.dataset_to_split = {
            str(name): str(split) for split, name in self.datasets_to_explain.items()
        }
        self.sample_manifest = sample_manifest.copy()
        self.source_info = source_info
        self.model_config = dict(model_config)
        self.bio_feature_names = list(map(str, bio_feature_names))
        self.geom_feature_names = list(map(str, geom_feature_names))
        self.device = device
        self.make_loader = make_loader
        self.load_seed_model = load_seed_model
        self.background_inputs = {
            key: value.to(device=device, dtype=torch.float32)
            for key, value in background_inputs.items()
        }
        self.exact_baseline_inputs = {
            key: value.to(device=device, dtype=torch.float32)
            for key, value in exact_baseline_inputs.items()
        }
        self.gradient_nsamples = int(gradient_nsamples)
        self.gradient_batch_size = int(gradient_batch_size)
        self.chunk_size = max(1, int(chunk_size))
        self.deep_additivity_fallback = bool(deep_additivity_fallback)
        self.status_frames = {}

    @staticmethod
    def _chunks(frame, chunk_size):
        ordered = frame.sort_values("Selection_Order").reset_index(drop=True)
        for chunk_index, start in enumerate(range(0, len(ordered), int(chunk_size))):
            yield chunk_index, ordered.iloc[start:start + int(chunk_size)]

    def _status_chunk_complete(self, explainer_name, dataset_name, seed,
                               chunk_index, expected_samples):
        prefix = explainer_name.replace("Explainer", "").lower()
        status_path = self.store.path(
            f"shap_{prefix}_status", dataset_name, seed, chunk_index
        )
        if not status_path.is_file():
            return False
        try:
            status = pd.read_parquet(status_path)
        except Exception:
            return False
        return (
            len(status) == int(expected_samples)
            and status["Status"].astype(str).eq("completed").all()
        )

    def _record(self, seed, dataset_name, sample_row):
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

    def _sample_inputs(self, sample):
        moved = move_batch_to_device(sample, self.device)
        inputs = {
            key: moved[key].to(torch.float32)
            for key in INPUT_KEYS
        }
        return moved, inputs

    def _summarize_values(
        self, values, inputs, record, explainer_name, actual_output, baseline_output
    ):
        modality_rows, position_rows, feature_rows = [], [], []
        arrays = dict(zip(INPUT_KEYS, values))
        for key, modality in INPUT_TO_MODALITY.items():
            attribution = arrays[key]
            modality_rows.append({
                **record, "Explainer": explainer_name, "Modality": modality,
                "Signed_Attribution_Sum": float(attribution.sum()),
                "Absolute_Attribution_Sum": float(np.abs(attribution).sum()),
            })
            if key.startswith("token_"):
                token_values = attribution[0]
                for allele_index, allele in enumerate(("Ref", "Alt")):
                    for position in range(token_values.shape[1]):
                        token_attr = token_values[allele_index, position]
                        position_rows.append({
                            **record, "Explainer": explainer_name,
                            "Modality": modality, "Allele": allele,
                            "Relative_Token_Position": (
                                position - int(record.get("K_Radius", 16))
                            ),
                            "Signed_Attribution": float(token_attr.sum()),
                            "Absolute_Attribution": float(np.abs(token_attr).sum()),
                        })
            else:
                feature_names = (
                    self.bio_feature_names if key == "bio_features"
                    else self.geom_feature_names
                )
                feature_attr = attribution[0].reshape(-1)
                input_values = inputs[key][0].detach().cpu().numpy().reshape(-1)
                for index, name in enumerate(feature_names):
                    feature_rows.append({
                        **record, "Explainer": explainer_name, "Modality": modality,
                        "Feature": name, "Signed_Attribution": float(feature_attr[index]),
                        "Absolute_Attribution": float(abs(feature_attr[index])),
                        "Input_Value": float(input_values[index]),
                    })
        total_attr = float(sum(array.sum() for array in values))
        completeness = {
            **record, "Explainer": explainer_name,
            "Model_Logit": float(actual_output), "Background_Mean_Logit": float(baseline_output),
            "Sum_SHAP_Values": total_attr,
            "Additivity_Residual": float(actual_output - baseline_output - total_attr),
            "Relative_Additivity_Residual": float(
                abs(actual_output - baseline_output - total_attr)
                / max(abs(actual_output - baseline_output), 1e-8)
            ),
        }
        return modality_rows, position_rows, feature_rows, completeness

    def _run_exact(self, shap_module, context_model, inputs, record):
        model = ExactModalityCoalitionModel(
            context_model, inputs, self.exact_baseline_inputs, self.device
        )
        feature_names = [modality for _, modality in EXACT_GROUPS]
        masker = np.zeros((1, len(feature_names)), dtype=np.float32)
        explainer = shap_module.ExactExplainer(
            model, masker=masker, feature_names=feature_names
        )
        result = explainer(np.ones((1, len(feature_names)), dtype=np.float32))
        values = np.asarray(result.values, dtype=np.float32).reshape(-1)
        base_values = np.asarray(result.base_values, dtype=np.float32).reshape(-1)
        base_value = float(base_values[0])
        actual = float(model(np.ones((1, len(feature_names)), dtype=np.float32))[0])
        rows = []
        for name, value in zip(feature_names, values):
            rows.append({
                **record, "Explainer": "ExactExplainer", "Modality": name,
                "SHAP_Value_Logit": float(value), "Base_Logit": base_value,
                "Model_Logit": actual,
            })
        return rows, actual, base_value

    def _run_gradient_or_deep(self, shap_module, context_model, inputs, record,
                              explainer_name, seed):
        background = [self.background_inputs[key] for key in INPUT_KEYS]
        explain_inputs = [inputs[key] for key in INPUT_KEYS]
        if explainer_name == "GradientExplainer":
            explainer = shap_module.GradientExplainer(
                context_model, background,
                batch_size=self.gradient_batch_size,
                local_smoothing=0,
            )
            raw_values = explainer.shap_values(
                explain_inputs,
                nsamples=self.gradient_nsamples,
                rseed=int(seed),
            )
            additivity_checked = None
        else:
            explainer = shap_module.DeepExplainer(context_model, background)
            try:
                raw_values = explainer.shap_values(explain_inputs, check_additivity=True)
                additivity_checked = True
            except AssertionError:
                if not self.deep_additivity_fallback:
                    raise
                raw_values = explainer.shap_values(explain_inputs, check_additivity=False)
                additivity_checked = False

        values = _normalize_shap_values(raw_values, explain_inputs)
        with torch.no_grad():
            actual = float(context_model(*explain_inputs).reshape(-1)[0].item())
            background_logits = context_model(*background).reshape(-1)
            baseline = float(background_logits.mean().item())
        modality_rows, position_rows, feature_rows, completeness = self._summarize_values(
            values, inputs, record, explainer_name, actual, baseline
        )
        completeness["Explainer_Additivity_Check_Ran"] = additivity_checked
        return modality_rows, position_rows, feature_rows, completeness, values

    def run(self):
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "SHAP is required for this cell. Install it in the notebook kernel "
                "environment, then rerun the SHAP stage."
            ) from exc
        self.shap_version = str(getattr(shap, "__version__", "unknown"))

        explainers = ("ExactExplainer", "GradientExplainer", "DeepExplainer")
        for dataset_name, sample_rows in self.sample_manifest.groupby("Dataset", sort=False):
            chunks = list(self._chunks(sample_rows, self.chunk_size))
            pending = {}
            for seed in self.seeds:
                seed_chunks = []
                for chunk_index, chunk in chunks:
                    needs_run = {
                        name: not self._status_chunk_complete(
                            name, dataset_name, seed, chunk_index, len(chunk)
                        )
                        for name in explainers
                    }
                    if any(needs_run.values()):
                        seed_chunks.append((chunk_index, chunk, needs_run))
                if seed_chunks:
                    pending[seed] = seed_chunks
            if not pending:
                print(f"[SHAP RESUME] {dataset_name}: all seeds/explainers complete; skipping data load")
                continue

            split_name = self.dataset_to_split[str(dataset_name)]
            loader = self.make_loader(split_name, self.seeds[0])
            for seed, seed_chunks in pending.items():
                model = None
                for chunk_index, chunk, needs_run in seed_chunks:
                    if model is None:
                        model = self.load_seed_model(seed, loader.dataset)
                    chunk_outputs = {
                        "ExactExplainer": {"groups": [], "status": []},
                        "GradientExplainer": {
                            "modalities": [], "positions": [], "features": [], "completeness": [], "status": [],
                        },
                        "DeepExplainer": {
                            "modalities": [], "positions": [], "features": [], "completeness": [], "status": [],
                        },
                    }
                    for _, sample_row in chunk.iterrows():
                        sample = loader.dataset[int(sample_row["Sample_Index"])]
                        moved, inputs = self._sample_inputs(sample)
                        record = self._record(seed, dataset_name, sample_row)
                        record["K_Radius"] = int(self.model_config["k_radius"])
                        context_model = FixedContextInputModel(model, moved).to(self.device)
                        context_model.eval()
                        status_common = {
                            **record, "SHAP_Background_Count": int(
                                self.background_inputs["token_dna"].shape[0]
                            ), "SHAP_Version": self.shap_version,
                        }

                        if needs_run["ExactExplainer"]:
                            try:
                                group_rows, actual, base = self._run_exact(
                                    shap, context_model, inputs, record
                                )
                                chunk_outputs["ExactExplainer"]["groups"].extend(group_rows)
                                exact_status = {
                                    **status_common, "Explainer": "ExactExplainer", "Status": "completed",
                                    "Error_Type": "", "Error_Message": "",
                                    "Model_Logit": actual, "Background_Logit": base,
                                    "Additivity_Residual": float(
                                        actual - base - sum(row["SHAP_Value_Logit"] for row in group_rows)
                                    ),
                                    "Coalition_Count": 2 ** len(EXACT_GROUPS),
                                }
                            except Exception as exc:
                                exact_status = {
                                    **status_common, "Explainer": "ExactExplainer", "Status": "error",
                                    "Error_Type": type(exc).__name__, "Error_Message": str(exc)[:1000],
                                }
                            chunk_outputs["ExactExplainer"]["status"].append(exact_status)

                        for explainer_name in ("GradientExplainer", "DeepExplainer"):
                            if not needs_run[explainer_name]:
                                continue
                            try:
                                modality_rows, position_rows, feature_rows, completeness, arrays = (
                                    self._run_gradient_or_deep(
                                        shap, context_model, inputs, record,
                                        explainer_name, seed,
                                    )
                                )
                                raw_path = (
                                    self.output_root / "shap_values" / explainer_name
                                    / str(dataset_name) / f"seed_{seed}"
                                )
                                raw_path.mkdir(parents=True, exist_ok=True)
                                filename = (
                                    f"{int(sample_row['Sample_Index']):08d}_"
                                    f"{safe_variant_filename(str(sample_row['Variant_ID']))}"
                                )
                                destination = raw_path / f"{filename}.npz"
                                temporary = raw_path / f"{filename}.tmp.npz"
                                np.savez_compressed(
                                    temporary,
                                    **{key: value.astype(np.float32) for key, value in zip(INPUT_KEYS, arrays)},
                                )
                                temporary.replace(destination)
                                chunk_outputs[explainer_name]["modalities"].extend(modality_rows)
                                chunk_outputs[explainer_name]["positions"].extend(position_rows)
                                chunk_outputs[explainer_name]["features"].extend(feature_rows)
                                chunk_outputs[explainer_name]["completeness"].append(completeness)
                                status = {
                                    **status_common, "Explainer": explainer_name, "Status": "completed",
                                    "Error_Type": "", "Error_Message": "",
                                    "Relative_Additivity_Residual": completeness[
                                        "Relative_Additivity_Residual"
                                    ],
                                    "Explainer_Additivity_Check_Ran": completeness.get(
                                        "Explainer_Additivity_Check_Ran"
                                    ),
                                    "Raw_Attribution_File": str(destination),
                                }
                            except Exception as exc:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                status = {
                                    **status_common, "Explainer": explainer_name, "Status": "error",
                                    "Error_Type": type(exc).__name__, "Error_Message": str(exc)[:1000],
                                }
                            chunk_outputs[explainer_name]["status"].append(status)

                    for explainer_name, values in chunk_outputs.items():
                        stage_prefix = f"shap_{explainer_name.replace('Explainer', '').lower()}"
                        for result_name, rows in values.items():
                            if not rows:
                                continue
                            stage = f"{stage_prefix}_{result_name}"
                            self.store.write(
                                stage, dataset_name, seed, chunk_index, pd.DataFrame(rows)
                            )
                    print(f"[SHAP] {dataset_name} seed={seed} chunk={chunk_index}: "
                          f"{len(chunk)} samples; statuses saved for Exact/Gradient/Deep")
                if model is not None:
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            del loader

        output_filenames = {
            "shap_exact_groups": "shap_exact_groups.csv",
            "shap_exact_status": "shap_exact_status.csv",
            "shap_gradient_modalities": "shap_gradient_modalities.csv",
            "shap_gradient_positions": "shap_gradient_positions.csv",
            "shap_gradient_features": "shap_gradient_features.csv",
            "shap_gradient_completeness": "shap_gradient_completeness.csv",
            "shap_gradient_status": "shap_gradient_status.csv",
            "shap_deep_modalities": "shap_deep_modalities.csv",
            "shap_deep_positions": "shap_deep_positions.csv",
            "shap_deep_features": "shap_deep_features.csv",
            "shap_deep_completeness": "shap_deep_completeness.csv",
            "shap_deep_status": "shap_deep_status.csv",
        }
        return {
            name: self.store.export_csv(stage, self.output_root / filename)
            for stage, filename in output_filenames.items()
            for name in [stage]
        }
