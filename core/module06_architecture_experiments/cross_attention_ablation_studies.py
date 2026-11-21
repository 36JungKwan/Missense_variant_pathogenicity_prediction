from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn

from core.module05v2_fusion_classifier.dataset import BIO_CORE_COLUMNS, SPLICEAI_COLUMNS
from core.module05v2_fusion_classifier.resume_utils import (
    ensure_stage_signature,
    save_progress_csv,
    upsert_record,
)
from core.module06_architecture_experiments.architecture_experiment_runner import (
    _architecture_loader,
    run_architecture_experiment,
)
from core.module06_architecture_experiments.cross_attention_3modal_experiment import (
    CrossAttentionThreeModalities,
    _CrossAttentionBlock,
)
from core.module06_architecture_experiments.fusion_classifier_benchmark import (
    _fit_fixed_tree,
    _predict_proba,
    _save_tree_model,
    _tree_node_count,
    _compute_metrics,
)
from core.module06_architecture_experiments.token_topk_experiment import _VariantCenteredWindow
from core.module06_architecture_experiments.transformer_3modal_experiment import (
    _ScalarFeatureTokenizer,
)


GEOM_FEATURES = [
    f"{modality}_{feature}"
    for modality in ("dna", "prot")
    for feature in ("LVD_L2", "LVD_Cosine", "LID", "LVD_Relative")
]
PCA_COMPONENTS = 256
CLASSIFIERS = ("XGBoost", "CatBoost", "LightGBM")


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def _load_rows(path):
    path = Path(path)
    return pd.read_csv(path).to_dict("records") if path.is_file() else []


def _collect_feature_matrix(context, target, split_name, seed, *, is_train=False):
    pipeline = context["pipeline"]
    loader = _architecture_loader(
        pipeline, context, target, split_name, "center", context["DNA_MODEL"],
        context["PROTEIN_MODEL"], is_train, "train3", seed, "delta",
    )
    dna, protein, bio, geometry, labels, ids = [], [], [], [], [], []
    dataset = loader.dataset
    dna_dim = int(dataset.v_dna_tensor.shape[1]) if getattr(dataset, "has_dna", False) else 0
    protein_dim = int(dataset.v_prot_tensor.shape[1]) if getattr(dataset, "has_prot", False) else 0
    bio_dim = int(dataset.bio_tensor.shape[1]) if getattr(dataset, "has_bio", False) else 0
    geom_dim = int(dataset.geom_tensor.shape[1]) if getattr(dataset, "has_geom", False) else 0
    for batch in loader:
        if batch.get("v_dna") is not None:
            dna.append(batch["v_dna"].cpu().numpy())
        if batch.get("v_prot") is not None:
            protein.append(batch["v_prot"].cpu().numpy())
        if batch.get("bio_features") is not None:
            bio.append(batch["bio_features"].cpu().numpy())
        if batch.get("geom_features") is not None:
            geometry.append(batch["geom_features"].cpu().numpy())
        labels.append(batch["label"].reshape(-1).cpu().numpy().astype(np.int32))
        ids.extend(str(value) for value in batch["variant_id"])
    del loader
    modality_parts = [np.concatenate(values, axis=0) for values in (dna, protein, bio, geometry) if values]
    matrix = np.concatenate(modality_parts, axis=1).astype(np.float32, copy=False)
    return matrix, np.concatenate(labels), ids, {
        "dna": dna_dim, "protein": protein_dim, "bio": bio_dim, "geometry": geom_dim,
    }


def _apply_pca(train, other, dims, components, seed):
    dna_dim, protein_dim = dims["dna"], dims["protein"]
    if not dna_dim or not protein_dim:
        raise ValueError("Pure ML PCA requires both DNA and protein embeddings.")
    pca_dna = PCA(
        n_components=min(int(components), len(train), dna_dim),
        svd_solver="randomized", random_state=int(seed),
    ).fit(train[:, :dna_dim])
    pca_protein = PCA(
        n_components=min(int(components), len(train), protein_dim),
        svd_solver="randomized", random_state=int(seed),
    ).fit(train[:, dna_dim:dna_dim + protein_dim])

    def transform(matrix):
        dna = pca_dna.transform(matrix[:, :dna_dim])
        protein = pca_protein.transform(matrix[:, dna_dim:dna_dim + protein_dim])
        tabular = matrix[:, dna_dim + protein_dim:]
        return np.concatenate((dna, protein, tabular), axis=1).astype(np.float32, copy=False)

    return transform(train), [transform(matrix) for matrix in other], pca_dna, pca_protein


def _save_validation(path, row):
    rows = _load_rows(path)
    rows = upsert_record(rows, row, ["Seed", "Classifier", "PCA"])
    save_progress_csv(rows, path)
    return row


def _pure_ml_seed_complete(output_root, seed, datasets):
    for classifier in CLASSIFIERS:
        for label in ("NoPCA", "PCA"):
            run_dir = Path(output_root) / classifier / label / f"seed_{seed}"
            model_path = run_dir / "best_model"
            model_path = model_path.with_suffix(
                ".json" if classifier == "XGBoost" else ".txt" if classifier == "LightGBM" else ".cbm"
            )
            val_rows = _load_rows(run_dir / "validation_metrics.csv")
            test_rows = _load_rows(run_dir / "test_metrics.csv")
            prediction_rows = _load_rows(run_dir / "test_predictions.csv")
            if not model_path.is_file() or not val_rows:
                return False
            for dataset in datasets:
                name = str(dataset["name"])
                metric = next((row for row in test_rows if str(row.get("Dataset")) == name), None)
                n_predictions = sum(1 for row in prediction_rows if str(row.get("Dataset")) == name)
                if metric is None or pd.isna(metric.get("MCC")) or n_predictions < int(float(metric.get("N_Test_Samples", 0))):
                    return False
    return True


def run_pure_ml_pca_study(*, context, target, seeds, classifier_parameters,
                          output_root, pca_components=PCA_COMPONENTS, pca_modes=None,
                          display_fn=print):
    """Evaluate the three fixed ML classifiers with separate DNA/protein PCA and without PCA."""
    pca_modes = pca_modes or {"NoPCA": False, "PCA": True}
    if isinstance(pca_modes, dict):
        pca_modes = list(pca_modes.items())
    pca_modes = [(str(name), bool(enabled)) for name, enabled in pca_modes]
    mode_names = {name for name, _ in pca_modes}
    if mode_names != {"NoPCA", "PCA"} or len(pca_modes) != 2:
        raise ValueError(
            "Pure ML comparison requires exactly two modes: {'NoPCA': False, 'PCA': True}."
        )
    if dict(pca_modes) != {"NoPCA": False, "PCA": True}:
        raise ValueError("NoPCA must disable PCA and PCA must enable it.")
    output_root = Path(output_root) / "pure_ml_pca"
    output_root.mkdir(parents=True, exist_ok=True)
    signature = {
        "study": "pure_ml_pca_vs_no_pca",
        "target": target,
        "seeds": list(map(int, seeds)),
        "classifiers": list(CLASSIFIERS),
        "pca_modes": dict(pca_modes),
        "classifier_parameters": classifier_parameters,
        "pca_components": int(pca_components),
        "pca_fit_scope": "train3_only; separate DNA and protein; preserve tabular features",
        "input_representation": "center-pooled E_alt_minus_E_ref + selected biological/geometry features",
    }
    ensure_stage_signature(output_root, "pure_ml_pca_ablation", signature)
    summary_path = output_root / "test_metrics_by_seed.csv"
    summary_rows = _load_rows(summary_path)
    for seed in map(int, seeds):
        if _pure_ml_seed_complete(output_root, seed, context["datasets"]):
            print(f"[RESUME] Pure ML seed {seed}: all six classifier/PCA runs complete; skipping feature loading.")
            for classifier in CLASSIFIERS:
                for label, _use_pca in pca_modes:
                    run_test_rows = _load_rows(
                        output_root / classifier / label / f"seed_{seed}" / "test_metrics.csv"
                    )
                    for row in run_test_rows:
                        summary_rows = upsert_record(
                            summary_rows, row,
                            ["Classifier", "PCA", "Seed", "Dataset"],
                        )
            save_progress_csv(summary_rows, summary_path)
            continue
        print(f"[Pure ML] Loading fixed train/validation/test features for seed {seed}...")
        train, y_train, _, dims = _collect_feature_matrix(context, target, "train3", seed, is_train=True)
        validation, y_val, _, _ = _collect_feature_matrix(context, target, "val", seed)
        tests = {}
        for split in context["datasets"]:
            tests[split["name"]] = _collect_feature_matrix(context, target, split["test"], seed)

        for classifier in CLASSIFIERS:
            source = classifier_parameters[classifier]
            base_params = dict(source["params"])
            for label, use_pca in pca_modes:
                run_dir = output_root / classifier / label / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                model_path = run_dir / "best_model"
                if classifier == "XGBoost":
                    model_path = model_path.with_suffix(".json")
                elif classifier == "LightGBM":
                    model_path = model_path.with_suffix(".txt")
                else:
                    model_path = model_path.with_suffix(".cbm")
                val_path = run_dir / "validation_metrics.csv"
                test_path = run_dir / "test_metrics.csv"
                predictions_path = run_dir / "test_predictions.csv"
                ensure_stage_signature(run_dir, f"pure_ml_{classifier}_{label}_seed_{seed}", {
                    "target": target, "seed": seed, "classifier": classifier,
                    "parameters": base_params, "pca": use_pca,
                    "pca_components": int(pca_components), "input": "delta + tabular",
                })
                saved_validation = _load_rows(val_path)
                val_record = next((row for row in saved_validation
                                   if int(float(row.get("Seed", -1))) == seed), None)
                saved_test = _load_rows(test_path)
                prediction_rows = _load_rows(predictions_path)
                expected_datasets = [str(item["name"]) for item in context["datasets"]]
                complete_tests = all(any(
                    str(row.get("Dataset")) == name and pd.notna(row.get("MCC"))
                    for row in saved_test
                ) and any(
                    str(row.get("Dataset")) == name
                    for row in prediction_rows
                ) for name in expected_datasets)

                if val_record is not None and model_path.is_file() and complete_tests:
                    print(f"[RESUME] PureML {classifier}/{label}/seed_{seed}: complete; skipping fit and test.")
                    for row in saved_test:
                        summary_rows = upsert_record(
                            summary_rows, row,
                            ["Classifier", "PCA", "Seed", "Dataset"],
                        )
                    save_progress_csv(summary_rows, summary_path)
                    continue

                x_train, x_val = train, validation
                x_tests = {name: bundle[0] for name, bundle in tests.items()}
                pca_dna = pca_protein = None
                if use_pca:
                    x_train, transformed, pca_dna, pca_protein = _apply_pca(
                        train, [validation, *x_tests.values()], dims, pca_components, seed
                    )
                    x_val = transformed[0]
                    x_tests = dict(zip(x_tests, transformed[1:]))
                    joblib.dump(pca_dna, run_dir / "pca_dna.joblib")
                    joblib.dump(pca_protein, run_dir / "pca_protein.joblib")

                if val_record is not None and model_path.is_file():
                    from core.module06_architecture_experiments.fusion_classifier_benchmark import _load_tree_model
                    model = _load_tree_model(classifier, model_path)
                    threshold = float(val_record["Decision_Threshold"])
                    fit_seconds = float(val_record.get("Fit_Seconds", 0.0))
                else:
                    params = dict(base_params)
                    started = time.perf_counter()
                    model = _fit_fixed_tree(classifier, params, seed, x_train, y_train, x_val, y_val)
                    fit_seconds = time.perf_counter() - started
                    probabilities = _predict_proba(classifier, model, x_val)
                    threshold, val_mcc = context["pipeline"]._select_validation_threshold(y_val, probabilities)
                    val_metrics = _compute_metrics(context["pipeline"], y_val, probabilities, threshold)
                    _save_tree_model(classifier, model, model_path)
                    val_record = {
                        "Branch": "Pure_ML", "Target_Config": target["name"],
                        "Classifier": classifier, "PCA": use_pca, "PCA_Mode": label, "Seed": seed,
                        "Input_Representation": "delta", "PCA_Components": int(pca_components) if use_pca else 0,
                        "Decision_Threshold": float(threshold), "Val_MCC": float(val_mcc),
                        "Fit_Seconds": round(fit_seconds, 3),
                        "Legacy_Parameter_Source": source.get("source_csv"),
                        "Legacy_Parameter_Trial": source.get("source_trial"),
                        "Legacy_Parameter_Val_MCC": source.get("source_val_mcc"),
                        "Hyperparameters": json.dumps(params, sort_keys=True, default=_json_safe),
                        **{f"Val_{name}": value for name, value in val_metrics.items()},
                    }
                    _save_validation(val_path, val_record)

                for dataset in context["datasets"]:
                    dataset_name = str(dataset["name"])
                    if any(str(row.get("Dataset")) == dataset_name for row in saved_test) and any(
                        str(row.get("Dataset")) == dataset_name for row in prediction_rows
                    ):
                        saved_row = next(row for row in saved_test if str(row.get("Dataset")) == dataset_name)
                        summary_rows = upsert_record(
                            summary_rows, saved_row,
                            ["Classifier", "PCA", "Seed", "Dataset"],
                        )
                        save_progress_csv(summary_rows, summary_path)
                        continue
                    x_test = x_tests[dataset_name]
                    y_test = tests[dataset_name][1]
                    ids = tests[dataset_name][2]
                    probabilities = _predict_proba(classifier, model, x_test)
                    metrics = _compute_metrics(context["pipeline"], y_test, probabilities, threshold)
                    row = {
                        "Run_ID": Path(context["pipeline"].batch_run_dir).name,
                        "Branch": "Pure_ML", "Target_Config": target["name"],
                        "Dataset": dataset_name, "Classifier": classifier,
                        "PCA": use_pca, "PCA_Mode": label, "Seed": seed, "Input_Representation": "delta",
                        "Decision_Threshold": threshold, "N_Test_Samples": len(y_test),
                        "Fit_Seconds": round(fit_seconds, 3),
                        "Hyperparameters": json.dumps(base_params, sort_keys=True, default=_json_safe),
                        **metrics,
                    }
                    saved_test = upsert_record(saved_test, row, ["Seed", "Classifier", "PCA", "Dataset"])
                    prediction_rows = [r for r in prediction_rows if not (
                        str(r.get("Dataset")) == dataset_name and int(float(r.get("Seed", -1))) == seed
                    )]
                    prediction_rows.extend({
                        "Classifier": classifier, "PCA": use_pca, "PCA_Mode": label, "Seed": seed,
                        "Dataset": dataset_name, "Variant_ID": str(variant_id),
                        "Label": int(label), "Probability_Pathogenic": float(probability),
                        "Prediction": int(probability >= threshold),
                    } for variant_id, label, probability in zip(ids, y_test, probabilities))
                    save_progress_csv(saved_test, test_path)
                    save_progress_csv(prediction_rows, predictions_path)
                    summary_rows = upsert_record(
                        summary_rows, row, ["Classifier", "PCA", "Seed", "Dataset"]
                    )
                    save_progress_csv(summary_rows, summary_path)
                    print(f"[Pure ML] {classifier}/{label}/seed_{seed}/{dataset_name}: MCC={metrics['MCC']:.4f}")

    return pd.DataFrame(_load_rows(summary_path))


class CrossAttentionSubsetModel(nn.Module):
    """The original tri-modal cross-attention with only configured branches present."""

    def __init__(self, dimensions, config, active_branches):
        super().__init__()
        self.active_branches = tuple(active_branches)
        if not self.active_branches:
            raise ValueError("At least one modality branch must be active.")
        model_dim = int(config["model_dim"])
        dropout = float(config["dropout"])
        feature_dropout = float(config["feature_dropout"])
        self.k_radius = int(config["k_radius"])
        self.pooling = str(config["pooling"])
        if "dna" in self.active_branches:
            self.dna_modality = nn.Parameter(torch.zeros(model_dim))
            self.dna_window = _VariantCenteredWindow(dimensions["token_dna"], model_dim, dropout)
        if "protein" in self.active_branches:
            self.protein_modality = nn.Parameter(torch.zeros(model_dim))
            self.protein_window = _VariantCenteredWindow(dimensions["token_protein"], model_dim, dropout)
        if "bio" in self.active_branches:
            self.bio_modality = nn.Parameter(torch.zeros(model_dim))
            self.geometry_modality = nn.Parameter(torch.zeros(model_dim))
            self.bio_tokenizer = _ScalarFeatureTokenizer(dimensions["bio"], model_dim, feature_dropout)
            self.geom_tokenizer = _ScalarFeatureTokenizer(dimensions["geometry"], model_dim, feature_dropout)

        self.cross_layers = nn.ModuleList()
        n_attention_layers = int(config["num_layers"]) if len(self.active_branches) > 1 else 0
        for _ in range(n_attention_layers):
            self.cross_layers.append(nn.ModuleDict({
                branch: _CrossAttentionBlock(
                    model_dim, int(config["num_heads"]), int(config["ffn_multiplier"]),
                    dropout, float(config["attention_dropout"]),
                )
                for branch in self.active_branches
            }))
        self.pool_scores = nn.ModuleDict({
            branch: nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
            for branch in self.active_branches
        })
        self.output_norm = nn.LayerNorm(model_dim * len(self.active_branches))
        self.classifier = nn.Sequential(
            nn.Linear(model_dim * len(self.active_branches), model_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def _pool(self, tokens, mask, branch):
        if self.pooling == "mean":
            weights = mask.unsqueeze(-1).to(tokens.dtype)
            return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        scores = self.pool_scores[branch](tokens).squeeze(-1).masked_fill(~mask, -torch.inf)
        return (tokens * torch.softmax(scores, dim=1).unsqueeze(-1)).sum(dim=1)

    def encode_features(self, v_dna=None, v_prot=None, bio_features=None, geom_features=None,
                        token_dna=None, token_dna_mask=None, token_dna_variant_index=None,
                        token_protein=None, token_protein_mask=None,
                        token_protein_variant_index=None):
        states, masks = {}, {}
        if "dna" in self.active_branches:
            states["dna"], masks["dna"] = self.dna_window(
                token_dna, token_dna_mask, token_dna_variant_index,
                self.k_radius, self.dna_modality,
            )
        if "protein" in self.active_branches:
            states["protein"], masks["protein"] = self.protein_window(
                token_protein, token_protein_mask, token_protein_variant_index,
                self.k_radius, self.protein_modality,
            )
        if "bio" in self.active_branches:
            device = next(value for value in (bio_features, geom_features, v_dna, v_prot,
                token_dna, token_protein) if value is not None).device
            tokens = []
            if self.bio_tokenizer.n_features:
                tokens.append(self.bio_tokenizer(bio_features) + self.bio_modality.view(1, 1, -1))
            if self.geom_tokenizer.n_features:
                tokens.append(self.geom_tokenizer(geom_features) + self.geometry_modality.view(1, 1, -1))
            if not tokens:
                raise ValueError("Bio branch was enabled without biological or geometry inputs.")
            states["bio"] = torch.cat(tokens, dim=1)
            masks["bio"] = torch.ones(states["bio"].shape[:2], dtype=torch.bool, device=device)

        for layer in self.cross_layers:
            old_states = states
            updated = dict(old_states)
            if len(self.active_branches) > 1:
                for branch in self.active_branches:
                    context_tokens = torch.cat([
                        old_states[name] for name in self.active_branches if name != branch
                    ], dim=1)
                    context_mask = torch.cat([
                        masks[name] for name in self.active_branches if name != branch
                    ], dim=1)
                    updated[branch] = layer[branch](old_states[branch], context_tokens, context_mask)
            states = updated
        pooled = [self._pool(states[name], masks[name], name) for name in self.active_branches]
        return self.output_norm(torch.cat(pooled, dim=1))

    def forward(self, *args, **kwargs):
        return self.classifier(self.encode_features(*args, **kwargs)).squeeze(-1)


def _subset_builder(dimensions, config):
    return CrossAttentionSubsetModel(dimensions, config, config["ablation_active_branches"])


def _feature_target(base_target, bio_features, geom_features, label):
    bio_features = list(bio_features)
    geom_features = list(geom_features)
    active = ["dna", "prot"]
    if any(name in BIO_CORE_COLUMNS for name in bio_features):
        active.append("bio_core")
    if any(name in SPLICEAI_COLUMNS for name in bio_features):
        active.append("spliceai")
    if geom_features:
        active.append("geom")
    target = dict(base_target)
    target.update({
        "name": "B_with_geometry",
        "ablation": label,
        "active_modalities": active,
        "bio_feature_names": bio_features,
        "geom_feature_names": geom_features,
    })
    return target


def _modalities_target(base_target, modalities, bio_features, geom_features, label):
    target = _feature_target(base_target, bio_features, geom_features, label)
    active = list(modalities)
    target["active_modalities"] = active
    return target


def _run_dl_condition(*, context, base_target, condition, seed, config, output_root,
                      study, display_fn=print):
    target = dict(condition["target"])
    model_config = dict(config)
    model_config["seed"] = int(seed)
    active_branches = []
    modalities = {str(item).lower() for item in target["active_modalities"]}
    if "dna" in modalities:
        active_branches.append("dna")
    if "prot" in modalities:
        active_branches.append("protein")
    if target.get("bio_feature_names") or target.get("geom_feature_names"):
        active_branches.append("bio")
    model_config["ablation_active_branches"] = active_branches
    uses_original_full_model = set(active_branches) == {"dna", "protein", "bio"}
    builder = (
        (lambda dims, cfg: CrossAttentionThreeModalities(
            dims["token_dna"], dims["token_protein"], dims["bio"], dims["geometry"], cfg
        ))
        if uses_original_full_model else _subset_builder
    )
    run_key = f"{study}_{_slug(condition['label'])}_seed_{int(seed)}"
    exp_context = dict(context)
    exp_context["TARGET_CONFIGS"] = [target]
    exp_context["token_modalities"] = [
        name for branch, name in (("dna", "dna"), ("protein", "protein"))
        if branch in active_branches
    ]
    result = run_architecture_experiment(
        context=exp_context,
        experiment_config=model_config,
        model_builder=builder,
        architecture="CrossAttention_ThreeModalities",
        variant=f"{study}_{condition['label']}",
        experiment_key=f"cross_attention_ablation_{run_key}",
        source=f"module06_{study}",
        display_fn=lambda *_args, **_kwargs: None,
        manifest_architecture=f"cross_attention_active_branches_{'-'.join(active_branches)}_v1",
        embedding_representation="ref_alt_concat",
        evaluate_test=True,
    )
    validation = result["validation_metrics"].copy()
    test = result["test_metrics"].copy()
    if not validation.empty:
        validation["Study"] = study
        validation["Condition"] = condition["label"]
        validation["Seed"] = int(seed)
        save_progress_csv(validation.to_dict("records"), Path(result["output_dir"]) / "study_validation_metrics.csv")
    if not test.empty:
        test["Study"] = study
        test["Condition"] = condition["label"]
        test["Seed"] = int(seed)
        save_progress_csv(test.to_dict("records"), Path(result["output_dir"]) / "study_test_metrics.csv")
    display_fn(f"[DONE] {study}/{condition['label']}/seed_{seed}")
    return validation, test


def feature_group_conditions(base_target):
    bio = list(BIO_CORE_COLUMNS) + list(SPLICEAI_COLUMNS)
    geom = list(GEOM_FEATURES)
    groups = [
        ("Drop_AF", ["AF", "gnomADe_AF"]),
        ("Drop_Conservation", [name for name in BIO_CORE_COLUMNS if name not in {"AF", "gnomADe_AF"}]),
        ("Drop_SpliceAI", list(SPLICEAI_COLUMNS)),
        ("Drop_All_Geometry", list(GEOM_FEATURES)),
        ("Drop_DNA_Geometry", [name for name in GEOM_FEATURES if name.startswith("dna_")]),
        ("Drop_Protein_Geometry", [name for name in GEOM_FEATURES if name.startswith("prot_")]),
    ]
    return [{
        "label": label,
        "target": _feature_target(base_target, [name for name in bio if name not in removed],
                                   [name for name in geom if name not in removed], label),
    } for label, removed in groups]


def feature_28_conditions(base_target):
    bio = list(BIO_CORE_COLUMNS) + list(SPLICEAI_COLUMNS)
    geom = list(GEOM_FEATURES)
    return [{
        "label": f"Drop_{name}",
        "target": _feature_target(
            base_target, [col for col in bio if col != name],
            [col for col in geom if col != name], f"drop_feature_{name}",
        ),
    } for name in (*bio, *geom)]


def modality_conditions(base_target):
    core = list(BIO_CORE_COLUMNS)
    splice = list(SPLICEAI_COLUMNS)
    geom = list(GEOM_FEATURES)
    specs = [
        ("Bio_Core_Only", ["bio_core"], core, []),
        ("DNA_LM_Only", ["dna"], [], []),
        ("Protein_LM_Only", ["prot"], [], []),
        ("SpliceAI_Only", ["spliceai"], splice, []),
        ("Geometry_Only", ["geom"], [], geom),
        ("DNA_Protein_Only", ["dna", "prot"], [], []),
        ("Bio_SpliceAI_Geometry_Only", ["bio_core", "spliceai", "geom"], [*core, *splice], geom),
        ("Protein_Bio_SpliceAI_Geometry", ["prot", "bio_core", "spliceai", "geom"], [*core, *splice], geom),
        ("DNA_Bio_SpliceAI_Geometry", ["dna", "bio_core", "spliceai", "geom"], [*core, *splice], geom),
        ("DNA_Protein_Bio_Core", ["dna", "prot", "bio_core"], core, []),
    ]
    return [{
        "label": label,
        "target": _modalities_target(base_target, active, bio, geometry, label),
    } for label, active, bio, geometry in specs]


def run_cross_attention_ablation_study(*, context, base_target, conditions, seeds,
                                       model_config, output_root, study, baseline_test,
                                       display_fn=print):
    output_root = Path(output_root) / study
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_stage_signature(output_root, f"cross_attention_{study}", {
        "study": study, "conditions": [condition["target"] for condition in conditions],
        "seeds": list(map(int, seeds)), "model_config": model_config,
        "baseline_source": context.get("baseline_source"),
        "target": base_target,
    }, mutable_paths=(("model_config", "batch_size"),)
    if context.get("allow_batch_size_resume") else ())
    validation_rows, test_rows = [], []
    for condition in conditions:
        for seed in map(int, seeds):
            print(f"[START] {study}/{condition['label']}/seed_{seed}")
            validation, test = _run_dl_condition(
                context=context, base_target=base_target, condition=condition,
                seed=seed, config=model_config, output_root=output_root,
                study=study, display_fn=display_fn,
            )
            validation_rows.extend(validation.to_dict("records"))
            test_rows.extend(test.to_dict("records"))
    validation_frame = pd.DataFrame(validation_rows)
    test_frame = pd.DataFrame(test_rows)
    save_progress_csv(validation_frame.to_dict("records"), output_root / "validation_metrics_by_seed.csv")
    save_progress_csv(test_frame.to_dict("records"), output_root / "test_metrics_by_seed.csv")
    comparison = test_frame.merge(
        baseline_test[["Seed", "Dataset", "MCC", "AUROC"]].rename(columns={
            "MCC": "Baseline_MCC", "AUROC": "Baseline_AUROC",
        }), on=["Seed", "Dataset"], how="left", validate="many_to_one",
    ) if not test_frame.empty else pd.DataFrame()
    if not comparison.empty:
        comparison["Delta_MCC"] = comparison["MCC"] - comparison["Baseline_MCC"]
        comparison["Delta_AUROC"] = comparison["AUROC"] - comparison["Baseline_AUROC"]
        save_progress_csv(comparison.to_dict("records"), output_root / "delta_vs_baseline_by_seed.csv")
        summary = comparison.groupby(["Condition", "Dataset"], dropna=False)[
            ["MCC", "AUROC", "Baseline_MCC", "Baseline_AUROC", "Delta_MCC", "Delta_AUROC"]
        ].agg(["mean", "std", "count"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        save_progress_csv(summary.to_dict("records"), output_root / "delta_mean_std_by_condition_dataset.csv")
    else:
        summary = pd.DataFrame()
    return {"validation": validation_frame, "test": test_frame,
            "comparison": comparison, "summary": summary}

