"""Step 3 FM pooling ablation: full tokens and Module 2 pooled vectors."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from core.module06_architecture_experiments.token_embedding_cache import (
    _atomic_torch_save,
    resolve_token_embedding_cache,
)
from core.module06_architecture_experiments.token_topk_experiment import (
    _TokenCacheDataset,
    _VariantCenteredWindow,
)
from core.module06_architecture_experiments.cross_attention_3modal_experiment import (
    _CrossAttentionBlock,
)
from core.module06_architecture_experiments.transformer_3modal_experiment import (
    _ScalarFeatureTokenizer,
)


REPORT_SEEDS = (5, 6, 7, 37, 42)
POOLED_MODES = ("center", "mean", "cls")
WINDOW_MODES = {"window_radius4_len9": 4, "window_radius8_len17": 8}
POOLED_TRAINED_MODES = ("pooled_center", "pooled_mean", "pooled_cls")
TRAINED_MODES = (*WINDOW_MODES, *POOLED_TRAINED_MODES)
STEP2_WINDOW_MODE = "window_radius16_len33"


def _atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _manifest_signature(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pooled_vectors_from_token_cache(*, token_cache_index, model_configs,
                                          split_configs, output_root,
                                          poolings=POOLED_MODES):
    """Derive Module 2 CLS/center/mean vectors once from the full token cache.

    Mean pooling follows Module 2 exactly: all positions with attention_mask=1
    contribute, including valid special tokens. Each cache stores ref and alt
    separately and records the source token-cache manifest digest for safe resume.
    """
    output_root = Path(output_root)
    index = {"schema_version": 1, "root": str(output_root.resolve()), "models": {}}
    for model in model_configs:
        model_name = str(model["name"])
        index["models"][model_name] = {}
        for split in split_configs:
            split_name = str(split["name"])
            entry = token_cache_index["models"][model_name][split_name]
            if entry.get("status") != "complete":
                raise RuntimeError(f"Token cache is incomplete: {model_name}/{split_name}")
            source_dir = Path(resolve_token_embedding_cache(token_cache_index, model_name, split_name))
            source_manifest_path = source_dir / "manifest.json"
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            source_digest = _manifest_signature(source_manifest_path)
            destinations = {
                pooling: output_root / model_name / pooling / f"{split_name}.pt"
                for pooling in poolings
            }
            manifests = {pooling: destination.with_suffix(".manifest.json")
                         for pooling, destination in destinations.items()}
            reusable = {}
            for pooling in poolings:
                expected = {
                    "schema_version": 1,
                    "model_name": model_name,
                    "split": split_name,
                    "pooling": pooling,
                    "source_token_manifest_sha256": source_digest,
                    "n_samples": int(source_manifest["n_samples"]),
                    "hidden_size": int(source_manifest["hidden_size"]),
                    "pooling_definition": (
                        "hidden_state_at_variant_token_index" if pooling == "center"
                        else "attention_masked_mean_including_valid_special_tokens" if pooling == "mean"
                        else "hidden_state_at_token_index_zero"
                    ),
                }
                saved = None
                if manifests[pooling].is_file():
                    try:
                        saved = json.loads(manifests[pooling].read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        saved = None
                reusable[pooling] = (
                    saved == expected and destinations[pooling].is_file()
                )
            if all(reusable.values()):
                for pooling in poolings:
                    index["models"][model_name].setdefault(split_name, {})[pooling] = {
                        "status": "complete", "path": str(destinations[pooling].resolve()),
                        "manifest": str(manifests[pooling].resolve()),
                    }
                print(f"[POOL CACHE RESUME] {model_name}/{split_name}")
                continue

            n_shards = int(source_manifest["n_shards"])
            hidden_size = int(source_manifest["hidden_size"])
            accumulated = {
                pooling: {"ref": [], "alt": [], "metadata": []}
                for pooling in poolings if not reusable[pooling]
            }
            started = time.perf_counter()
            for shard_index in range(n_shards):
                shard_path = source_dir / f"shard_{shard_index:06d}.pt"
                if not shard_path.is_file():
                    raise FileNotFoundError(f"Missing full-token shard: {shard_path}")
                if not accumulated:
                    break
                shard = torch.load(shard_path, map_location="cpu", weights_only=False)
                ids = [str(value) for value in shard["variant_ids"]]
                n = len(ids)
                rows = torch.arange(n)
                for pooling, values in accumulated.items():
                    for allele in ("ref", "alt"):
                        hidden = shard[f"token_hidden_{allele}"]
                        mask = shard[f"attention_mask_{allele}"].bool()
                        indices = shard[f"variant_token_index_{allele}"].long()
                        if pooling == "cls":
                            if not mask[:, 0].all():
                                raise ValueError(f"Token index 0 is masked in {shard_path}")
                            vector = hidden[:, 0, :]
                        elif pooling == "center":
                            if not mask[rows, indices].all():
                                raise ValueError(f"Variant token is masked in {shard_path}")
                            vector = hidden[rows, indices]
                        elif pooling == "mean":
                            weights = mask.unsqueeze(-1).to(torch.float32)
                            denominator = weights.sum(dim=1).clamp_min(1.0)
                            vector = (hidden.to(torch.float32) * weights).sum(dim=1) / denominator
                        else:
                            raise ValueError(f"Unsupported pooling mode: {pooling}")
                        values[allele].append(vector.contiguous().to(torch.float16))
                    values["metadata"].extend(ids)
                del shard

            for pooling, values in accumulated.items():
                destination = destinations[pooling]
                manifest_path = manifests[pooling]
                payload = {
                    "metadata": values["metadata"],
                    "E_ref": torch.cat(values["ref"], dim=0),
                    "E_alt": torch.cat(values["alt"], dim=0),
                }
                if (
                    len(payload["metadata"]) != int(source_manifest["n_samples"])
                    or tuple(payload["E_ref"].shape) != (int(source_manifest["n_samples"]), hidden_size)
                    or tuple(payload["E_alt"].shape) != (int(source_manifest["n_samples"]), hidden_size)
                ):
                    raise ValueError(f"Pooled-vector cache shape mismatch for {model_name}/{split_name}/{pooling}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                _atomic_torch_save(payload, destination)
                expected = {
                    "schema_version": 1, "model_name": model_name, "split": split_name,
                    "pooling": pooling, "source_token_manifest_sha256": source_digest,
                    "n_samples": int(source_manifest["n_samples"]), "hidden_size": hidden_size,
                    "pooling_definition": (
                        "hidden_state_at_variant_token_index" if pooling == "center"
                        else "attention_masked_mean_including_valid_special_tokens" if pooling == "mean"
                        else "hidden_state_at_token_index_zero"
                    ),
                }
                _atomic_json(expected, manifest_path)
                index["models"][model_name].setdefault(split_name, {})[pooling] = {
                    "status": "complete", "path": str(destination.resolve()),
                    "manifest": str(manifest_path.resolve()),
                }
            for pooling in poolings:
                if reusable[pooling]:
                    index["models"][model_name].setdefault(split_name, {})[pooling] = {
                        "status": "complete", "path": str(destinations[pooling].resolve()),
                        "manifest": str(manifests[pooling].resolve()),
                    }
            print(
                f"[POOL CACHE] {model_name}/{split_name}: "
                f"created {len(accumulated)} pooling(s) in {time.perf_counter()-started:.1f}s"
            )
    return index


class _PooledFMVectorDataset(Dataset):
    """Add ref-alt-concatenated FM vectors to a tabular/geometry dataset."""

    def __init__(self, base_dataset, pooled_index, pooling, split_name,
                 dna_model_name, protein_model_name):
        self.base_dataset = base_dataset
        self.df = base_dataset.df
        ids = self.df["Variant_ID"].astype(str).tolist()
        arrays = {}
        for modality, model_name in (("dna", dna_model_name), ("protein", protein_model_name)):
            item = pooled_index["models"][model_name][split_name][pooling]
            if item.get("status") != "complete":
                raise RuntimeError(f"Pooled FM vector cache is incomplete: {model_name}/{split_name}/{pooling}")
            data = torch.load(item["path"], map_location="cpu", weights_only=False)
            lookup = {str(value): i for i, value in enumerate(data["metadata"])}
            missing = [variant_id for variant_id in ids if variant_id not in lookup]
            if missing:
                raise KeyError(f"Pooled cache {model_name}/{pooling} misses Variant_IDs: {missing[:3]}")
            rows = torch.tensor([lookup[variant_id] for variant_id in ids], dtype=torch.long)
            refs = data["E_ref"].index_select(0, rows).to(torch.float32)
            alts = data["E_alt"].index_select(0, rows).to(torch.float32)
            arrays[modality] = torch.cat((refs, alts), dim=1).contiguous()
        self.v_dna_tensor = arrays["dna"]
        self.v_prot_tensor = arrays["protein"]

    def __len__(self):
        return len(self.base_dataset)

    def __getattr__(self, name):
        if name == "base_dataset":
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    def __getitem__(self, index):
        sample = dict(self.base_dataset[index])
        sample["v_dna"] = self.v_dna_tensor[index]
        sample["v_prot"] = self.v_prot_tensor[index]
        return sample


def add_pooled_vectors_to_loader(*, loader, pooled_index, pooling, split_name,
                                 dna_model_name, protein_model_name, is_train,
                                 batch_size=None):
    dataset = _PooledFMVectorDataset(
        loader.dataset, pooled_index, pooling, split_name,
        dna_model_name, protein_model_name,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size or loader.batch_size),
        shuffle=bool(is_train),
        num_workers=0,
        pin_memory=loader.pin_memory,
        generator=loader.generator,
    )


class _PooledPairProjector(nn.Module):
    """Project one pooled ref/alt pair as the variant-centered sequence token."""

    def __init__(self, input_size, model_dim, dropout, max_relative_radius):
        super().__init__()
        self.projection = nn.Linear(input_size, model_dim)
        self.position = nn.Embedding(2 * max_relative_radius + 1, model_dim)
        self.variant = nn.Parameter(torch.zeros(model_dim))
        self.modality_dropout = nn.Dropout(dropout)
        self.center_position_id = int(max_relative_radius)
        nn.init.normal_(self.variant, std=0.02)

    def forward(self, pooled_ref_alt, modality_embedding):
        result = self.projection(pooled_ref_alt).unsqueeze(1)
        result = result + self.position.weight[self.center_position_id].view(1, 1, -1)
        result = result + self.variant.view(1, 1, -1)
        result = result + modality_embedding.view(1, 1, -1)
        return self.modality_dropout(result)


class PoolingAblationCrossAttention(nn.Module):
    """Three-branch Cross-Attention head over windows or pooled ref/alt vectors."""

    def __init__(self, dna_dim, protein_dim, bio_dim, geom_dim, config):
        super().__init__()
        self.input_mode = str(config["input_mode"])
        self.model_dim = int(config["model_dim"])
        dropout = float(config["dropout"])
        feature_dropout = float(config.get("feature_dropout", dropout))
        max_radius = int(config.get("max_relative_radius", 600))
        if self.input_mode in WINDOW_MODES:
            self.k_radius = WINDOW_MODES[self.input_mode]
            self.dna_input = _VariantCenteredWindow(
                dna_dim, self.model_dim, dropout
            )
            self.protein_input = _VariantCenteredWindow(
                protein_dim, self.model_dim, dropout
            )
        elif self.input_mode.startswith("pooled_"):
            self.dna_input = _PooledPairProjector(
                dna_dim, self.model_dim, dropout, max_radius
            )
            self.protein_input = _PooledPairProjector(
                protein_dim, self.model_dim, dropout, max_radius
            )
        else:
            raise ValueError(f"Unsupported Step 3 input_mode: {self.input_mode}")

        self.dna_modality = nn.Parameter(torch.zeros(self.model_dim))
        self.protein_modality = nn.Parameter(torch.zeros(self.model_dim))
        self.bio_modality = nn.Parameter(torch.zeros(self.model_dim))
        self.geometry_modality = nn.Parameter(torch.zeros(self.model_dim))
        self.bio_tokenizer = _ScalarFeatureTokenizer(bio_dim, self.model_dim, feature_dropout)
        self.geom_tokenizer = _ScalarFeatureTokenizer(geom_dim, self.model_dim, feature_dropout)
        self.cross_layers = nn.ModuleList()
        for _ in range(int(config["num_layers"])):
            self.cross_layers.append(nn.ModuleDict({
                modality: _CrossAttentionBlock(
                    model_dim=self.model_dim,
                    num_heads=int(config["num_heads"]),
                    ffn_multiplier=int(config["ffn_multiplier"]),
                    dropout=dropout,
                    attention_dropout=float(config["attention_dropout"]),
                ) for modality in ("dna", "protein", "bio")
            }))
        self.output_norm = nn.LayerNorm(self.model_dim * 3)
        self.classifier = nn.Sequential(
            nn.Linear(self.model_dim * 3, self.model_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.model_dim * 2, self.model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.model_dim, 1),
        )

    @staticmethod
    def _masked_mean(tokens, mask):
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(self, v_dna, v_prot, bio_features, geom_features=None,
                token_dna=None, token_dna_mask=None, token_dna_variant_index=None,
                token_protein=None, token_protein_mask=None,
                token_protein_variant_index=None):
        if self.input_mode in WINDOW_MODES:
            if token_dna is None or token_protein is None:
                raise ValueError("Window-radius pooling ablation requires both token-cache inputs")
            dna, dna_mask = self.dna_input(
                token_dna, token_dna_mask, token_dna_variant_index,
                self.k_radius, self.dna_modality,
            )
            protein, protein_mask = self.protein_input(
                token_protein, token_protein_mask, token_protein_variant_index,
                self.k_radius, self.protein_modality,
            )
        else:
            dna = self.dna_input(v_dna, self.dna_modality)
            protein = self.protein_input(v_prot, self.protein_modality)
            dna_mask = torch.ones(dna.shape[:2], dtype=torch.bool, device=dna.device)
            protein_mask = torch.ones(protein.shape[:2], dtype=torch.bool, device=protein.device)

        bio = self.bio_tokenizer(bio_features) + self.bio_modality.view(1, 1, -1)
        bio_mask = torch.ones(bio.shape[:2], dtype=torch.bool, device=bio.device)
        if geom_features is not None:
            geometry = self.geom_tokenizer(geom_features) + self.geometry_modality.view(1, 1, -1)
            bio = torch.cat((bio, geometry), dim=1)
            bio_mask = torch.cat((
                bio_mask,
                torch.ones(geometry.shape[:2], dtype=torch.bool, device=geometry.device),
            ), dim=1)

        for layer in self.cross_layers:
            old_dna, old_protein, old_bio = dna, protein, bio
            dna = layer["dna"](
                old_dna, torch.cat((old_protein, old_bio), dim=1),
                torch.cat((protein_mask, bio_mask), dim=1),
            )
            protein = layer["protein"](
                old_protein, torch.cat((old_dna, old_bio), dim=1),
                torch.cat((dna_mask, bio_mask), dim=1),
            )
            bio = layer["bio"](
                old_bio, torch.cat((old_dna, old_protein), dim=1),
                torch.cat((dna_mask, protein_mask), dim=1),
            )

        # Output pooling after Cross-Attention is fixed to mean for every option.
        fused = self.output_norm(torch.cat((
            self._masked_mean(dna, dna_mask),
            self._masked_mean(protein, protein_mask),
            self._masked_mean(bio, bio_mask),
        ), dim=1))
        return self.classifier(fused).squeeze(-1)


def _builder(dimensions, config):
    window_mode = config["input_mode"] in WINDOW_MODES
    return PoolingAblationCrossAttention(
        dna_dim=dimensions["token_dna"] if window_mode else dimensions["dna"],
        protein_dim=dimensions["token_protein"] if window_mode else dimensions["protein"],
        bio_dim=dimensions["bio"],
        geom_dim=dimensions["geometry"],
        config=config,
    )


def load_step2_seed_configs(step2_runs_root, seeds=REPORT_SEEDS):
    """Read the fixed Step 2 hyperparameters and verify one config across seeds."""
    root = Path(step2_runs_root)
    configs, source_manifests = {}, {}
    for seed in seeds:
        run_key = f"step2_nt_v1_dna1001_esm1b_prot501_seed{int(seed)}"
        run_dir = root / run_key
        manifest_path = run_dir / "_resume_manifest.json"
        checkpoint_path = run_dir / "checkpoints" / "B_with_geometry" / "best_model.pth"
        validation_path = run_dir / f"{run_key}_validation_metrics.csv"
        test_path = run_dir / f"{run_key}_test_metrics.csv"
        for path in (manifest_path, checkpoint_path, validation_path, test_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required Step 2 artifact is missing: {path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved = payload.get("config", {})
        config_key = f"{run_key}_config"
        config = saved.get(config_key)
        if not isinstance(config, dict) or int(config.get("seed", -1)) != int(seed):
            raise ValueError(f"Step 2 manifest has no matching seed config: {manifest_path}")
        configs[int(seed)] = dict(config)
        source_manifests[int(seed)] = {
            "manifest": str(manifest_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "validation_metrics": str(validation_path.resolve()),
            "test_metrics": str(test_path.resolve()),
        }
    reference = {k: v for k, v in configs[int(seeds[0])].items() if k != "seed"}
    for seed, config in configs.items():
        if {k: v for k, v in config.items() if k != "seed"} != reference:
            raise ValueError(f"Step 2 hyperparameters differ for seed {seed}")
    return configs, source_manifests


def run_step3_pooling_ablation(*, pipeline_factory, target_configs, datasets,
                               model_configs, token_cache_index, pooled_vector_index,
                               step2_configs, step2_manifests, step_root,
                               seeds=REPORT_SEEDS, device="cuda",
                               pooled_batch_size=64, display_fn=print):
    """Train radius-4/8 and pooled center/mean/CLS; reuse Step 2 radius-16."""
    from core.module06_architecture_experiments.architecture_experiment_runner import (
        run_architecture_experiment,
    )

    step_root = Path(step_root)
    runs_root = step_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_records = []
    for mode in TRAINED_MODES:
        for seed in seeds:
            config = dict(step2_configs[int(seed)])
            config["input_mode"] = mode
            config["pooling"] = "mean"  # Fixed post-Cross-Attention output pooling.
            config["max_relative_radius"] = 600
            config["gradient_accumulation_steps"] = 1
            if mode in WINDOW_MODES:
                config["k_radius"] = WINDOW_MODES[mode]
            loader_batch_size = int(pooled_batch_size)
            pipeline = pipeline_factory(seed, model_configs)
            input_signature = {
                "stage": "step3_pooling_ablation",
                "input_mode": mode,
                "dna_model": model_configs[0]["name"],
                "protein_model": model_configs[1]["name"],
                "dna_length": 1001,
                "protein_length": 501,
                "geometry_features": list(target_configs[0]["geom_feature_names"]),
                "geometry_root": target_configs[0].get("geometry_root"),
                "geometry_context": "train3",
                "pooled_vector_manifest": (
                    pooled_vector_index["models"][model_configs[0]["name"]]["train3"][mode.removeprefix("pooled_")]["manifest"]
                    if mode.startswith("pooled_") else None
                ),
                "token_window_radius": WINDOW_MODES.get(mode),
                "token_cache_index_path": (
                    token_cache_index.get("index_path") if mode in WINDOW_MODES else None
                ),
                "token_cache_manifests": (
                    [
                        token_cache_index["models"][model["name"]]["train3"]["manifest"]
                        for model in model_configs
                    ] if mode in WINDOW_MODES else None
                ),
                "output_pooling_after_cross_attention": "mean",
                "source_step2_hyperparameter_manifest": step2_manifests[int(seed)]["manifest"],
                "effective_batch_size": loader_batch_size,
            }
            context = {
                "pipeline": pipeline,
                "TARGET_CONFIGS": target_configs,
                "datasets": datasets,
                "DNA_MODEL": model_configs[0],
                "PROTEIN_MODEL": model_configs[1],
                "DEVICE": device,
                "token_embedding_cache_index": token_cache_index if mode in WINDOW_MODES else None,
                "pooled_fm_vector_index": pooled_vector_index if mode.startswith("pooled_") else None,
                "pooled_fm_pooling": mode.removeprefix("pooled_") if mode.startswith("pooled_") else None,
                "token_batch_size": loader_batch_size,
                "token_only": True,
                "stage_root": str(runs_root),
                "input_data_signature": input_signature,
            }
            key = f"step3_pooling_{mode}_seed{int(seed)}"
            result = run_architecture_experiment(
                context=context,
                experiment_config=config,
                model_builder=_builder,
                architecture="CrossAttention_ThreeModalities_FM_PoolingAblation",
                variant=f"{mode}_ref_alt_concat",
                experiment_key=key,
                source="fm_ablation_step3",
                display_fn=display_fn,
                manifest_architecture="cross_attention_3modal_fm_pooling_ablation",
                embedding_representation="ref_alt_concat",
                evaluate_test=True,
            )
            run_records.append({"mode": mode, "seed": int(seed), **result})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {"status": "complete", "runs": run_records, "output_root": str(step_root.resolve())}


def summarize_step3_pooling_ablation(*, step2_runs_root, step3_root,
                                     seeds=REPORT_SEEDS):
    """Combine trained pooling options with the existing Step 2 radius-16 runs."""
    step2_runs_root = Path(step2_runs_root)
    step3_root = Path(step3_root)
    validation_frames, test_frames = [], []
    for mode in TRAINED_MODES:
        for seed in seeds:
            run_key = f"step3_pooling_{mode}_seed{int(seed)}"
            run_dir = step3_root / "runs" / run_key
            val_path = run_dir / f"{run_key}_validation_metrics.csv"
            test_path = run_dir / f"{run_key}_test_metrics.csv"
            if val_path.is_file():
                frame = pd.read_csv(val_path)
                frame["Pooling_Ablation_Mode"] = mode
                frame["Checkpoint_Path"] = str((run_dir / "checkpoints" / "B_with_geometry" / "best_model.pth").resolve())
                frame["Source_Experiment_Key"] = run_key
                validation_frames.append(frame)
            if test_path.is_file():
                frame = pd.read_csv(test_path)
                frame["Pooling_Ablation_Mode"] = mode
                frame["Checkpoint_Path"] = str((run_dir / "checkpoints" / "B_with_geometry" / "best_model.pth").resolve())
                frame["Source_Experiment_Key"] = run_key
                test_frames.append(frame)

    for seed in seeds:
        run_key = f"step2_nt_v1_dna1001_esm1b_prot501_seed{int(seed)}"
        run_dir = step2_runs_root / run_key
        checkpoint = run_dir / "checkpoints" / "B_with_geometry" / "best_model.pth"
        val_path = run_dir / f"{run_key}_validation_metrics.csv"
        test_path = run_dir / f"{run_key}_test_metrics.csv"
        if not all(path.is_file() for path in (checkpoint, val_path, test_path)):
            raise FileNotFoundError(f"Missing Step 2 radius-16 artifact for seed {seed}: {run_dir}")
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)
        if len(val) != 1 or len(test) != 4:
            raise ValueError(f"Unexpected Step 2 metrics row counts for seed {seed}: {len(val)}, {len(test)}")
        for frame in (val, test):
            frame["Pooling_Ablation_Mode"] = STEP2_WINDOW_MODE
            frame["Checkpoint_Path"] = str(checkpoint.resolve())
            frame["Source_Experiment_Key"] = run_key
            frame["Metric_Source"] = "reused_step2"
        validation_frames.append(val)
        test_frames.append(test)

    if not validation_frames or not test_frames:
        raise FileNotFoundError("No completed Step 3 pooling metrics found.")
    validation = pd.concat(validation_frames, ignore_index=True, sort=False)
    test = pd.concat(test_frames, ignore_index=True, sort=False)
    validation["Seed"] = pd.to_numeric(validation["Seed"], errors="coerce").astype("Int64")
    test["Seed"] = pd.to_numeric(test["Seed"], errors="coerce").astype("Int64")
    validation = validation.drop_duplicates(["Pooling_Ablation_Mode", "Seed", "Target_Config"], keep="last")
    test = test.drop_duplicates(["Pooling_Ablation_Mode", "Seed", "Target_Config", "Dataset"], keep="last")
    validation.to_csv(step3_root / "step3_validation_metrics_by_mode_seed.csv", index=False)
    test.to_csv(step3_root / "step3_test_metrics_by_mode_dataset_seed.csv", index=False)

    val_metrics = [c for c in validation.select_dtypes(include="number").columns if c != "Seed"]
    test_metrics = [c for c in test.select_dtypes(include="number").columns if c != "Seed"]
    val_summary = validation.groupby(["Pooling_Ablation_Mode", "Target_Config"], dropna=False)[val_metrics].agg(["mean", "std"])
    val_summary.columns = [f"{name}_{stat}" for name, stat in val_summary.columns]
    val_summary["N_Seeds"] = validation.groupby(["Pooling_Ablation_Mode", "Target_Config"], dropna=False)["Seed"].nunique()
    val_summary = val_summary.reset_index().sort_values("Val_MCC_mean", ascending=False)
    test_summary = test.groupby(["Pooling_Ablation_Mode", "Target_Config", "Dataset"], dropna=False)[test_metrics].agg(["mean", "std"])
    test_summary.columns = [f"{name}_{stat}" for name, stat in test_summary.columns]
    test_summary["N_Seeds"] = test.groupby(["Pooling_Ablation_Mode", "Target_Config", "Dataset"], dropna=False)["Seed"].nunique()
    test_summary = test_summary.reset_index().sort_values(["Dataset", "MCC_mean"], ascending=[True, False])
    val_summary.to_csv(step3_root / "step3_validation_mean_std_by_mode.csv", index=False)
    test_summary.to_csv(step3_root / "step3_test_mean_std_by_mode_dataset.csv", index=False)
    return {
        "validation_by_seed": validation,
        "test_by_seed": test,
        "validation_mean_std": val_summary,
        "test_mean_std_by_dataset": test_summary,
        "output_root": str(step3_root.resolve()),
    }
