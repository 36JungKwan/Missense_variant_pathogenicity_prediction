"""Transformer over variant-centered paired windows from full FM token caches."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from core.module06_architecture_experiments.token_embedding_cache import (
    resolve_token_embedding_cache,
)
from core.module06_architecture_experiments.architecture_experiment_runner import (
    run_architecture_optuna,
)


DEFAULT_TOKEN_TOPK_CONFIG = {
    "k_radius": 8,
    "model_dim": 256,
    "num_heads": 8,
    "num_layers": 2,
    "ffn_multiplier": 4,
    "dropout": 0.2,
    "attention_dropout": 0.1,
    "bio_dim": 128,
    "max_epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "early_stop_patience": 6,
    "lr_factor": 0.5,
    "lr_patience": 3,
    "seed": 42,
}
DEFAULT_TOKEN_TOPK_SEARCH_SPACE = {
    # k counts tokens on one side; the complete centered window is 2*k+1.
    "k_radius": {"type": "categorical", "choices": [4, 8, 16, 32, 64]},
    "model_dim": {"type": "categorical", "choices": [128, 256, 512]},
    "num_heads": {"type": "categorical", "choices": [4, 8]},
    "num_layers": {"type": "int", "low": 1, "high": 4},
    "ffn_multiplier": {"type": "categorical", "choices": [2, 4, 8]},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5},
    "attention_dropout": {"type": "float", "low": 0.0, "high": 0.4},
    "bio_dim": {"type": "categorical", "choices": [64, 128, 256, 512]},
    "learning_rate": {"type": "float", "low": 1e-6, "high": 2e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 0.2, "log": True},
}
DEFAULT_TOKEN_TOPK_N_TRIALS = 40
TOKEN_MODEL_NAMES = {
    "dna": "nt_v1_500m_dna_1001",
    "protein": "esm2_650m_prot_501",
}


class _TokenCacheDataset(Dataset):
    """Lazy shard-backed token lookup keyed by Variant_ID, with a bounded LRU."""

    def __init__(self, base_dataset, cache_index, split_name, dna_model_name,
                 protein_model_name, max_cached_shards=32):
        self.base_dataset = base_dataset
        self.df = base_dataset.df
        self.token_hidden_sizes = {}
        self.paths = {}
        self.id_lookup = {}
        self._shards = OrderedDict()
        self.max_cached_shards = int(max_cached_shards)
        for modality, model_name in (("dna", dna_model_name), ("protein", protein_model_name)):
            folder = resolve_token_embedding_cache(cache_index, model_name, split_name)
            folder = Path(folder)
            manifest = __import__("json").loads((folder / "manifest.json").read_text(encoding="utf-8"))
            self.token_hidden_sizes[modality] = int(manifest["hidden_size"])
            batch_size = int(manifest["batch_size"])
            expected_shards = int(manifest["n_shards"])
            lookup = {}
            base_ids = self.df["Variant_ID"].astype(str).tolist()
            source_ids = pd.read_parquet(
                manifest["input_path"], columns=["Variant_ID"]
            )["Variant_ID"].astype(str).tolist()
            if len(source_ids) != int(manifest["n_samples"]):
                raise KeyError(
                    f"Token cache manifest row count does not match its source parquet for "
                    f"{model_name}/{split_name}: {manifest['n_samples']} vs {len(source_ids)}"
                )
            # Match IDs against the original extraction parquet, rather than
            # assuming that geometry merges preserved row order. This avoids
            # opening every large hidden-state shard just to build the index.
            source_lookup = {variant_id: row for row, variant_id in enumerate(source_ids)}
            missing = [variant_id for variant_id in base_ids if variant_id not in source_lookup]
            if missing:
                raise KeyError(
                    f"Token cache {model_name}/{split_name} is missing {len(missing)} "
                    f"Module 5 variants; examples: {missing[:3]}"
                )
            for variant_id in base_ids:
                row = source_lookup[variant_id]
                shard_number, shard_row = divmod(row, batch_size)
                if shard_number >= expected_shards:
                    raise RuntimeError(f"Token cache manifest has too few shards: {folder}")
                shard_path = folder / f"shard_{shard_number:06d}.pt"
                if not shard_path.is_file():
                    raise FileNotFoundError(f"Token cache shard is missing: {shard_path}")
                lookup[variant_id] = (str(shard_path), shard_row)
            self.paths[modality] = lookup

    def __len__(self):
        return len(self.base_dataset)

    def __getattr__(self, name):
        # Keep the original dataset's feature tensors/metadata visible to the
        # shared Module 6 runner (dimensions, labels, and prediction IDs).
        if name == "base_dataset":
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    def _read(self, modality, variant_id):
        path, row = self.paths[modality][str(variant_id)]
        key = (modality, path)
        if key not in self._shards:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            self._shards[key] = shard
            self._shards.move_to_end(key)
            while len(self._shards) > self.max_cached_shards:
                self._shards.popitem(last=False)
        else:
            self._shards.move_to_end(key)
        shard = self._shards[key]
        if str(shard["variant_ids"][row]) != str(variant_id):
            raise ValueError(
                f"Token cache row order does not match the Module 5 split for "
                f"{variant_id!r}; regenerate token cache for {path} using the same source parquet."
            )
        hidden = torch.stack((shard["token_hidden_ref"][row], shard["token_hidden_alt"][row]))
        masks = torch.stack((shard["attention_mask_ref"][row], shard["attention_mask_alt"][row]))
        indices = torch.stack((
            shard["variant_token_index_ref"][row],
            shard["variant_token_index_alt"][row],
        )).to(torch.long)
        return hidden, masks.bool(), indices

    def __getitem__(self, index):
        sample = dict(self.base_dataset[index])
        variant_id = str(sample["variant_id"])
        for modality in ("dna", "protein"):
            hidden, masks, indices = self._read(modality, variant_id)
            sample[f"token_{modality}"] = hidden
            sample[f"token_{modality}_mask"] = masks
            sample[f"token_{modality}_variant_index"] = indices
        return sample


def _token_collate(samples):
    token_keys = {
        f"token_{modality}{suffix}"
        for modality in ("dna", "protein")
        for suffix in ("", "_mask", "_variant_index")
    }
    base = default_collate([
        {key: value for key, value in sample.items() if key not in token_keys}
        for sample in samples
    ])
    for modality in ("dna", "protein"):
        token_values = [sample[f"token_{modality}"] for sample in samples]
        masks = [sample[f"token_{modality}_mask"] for sample in samples]
        indices = torch.stack([sample[f"token_{modality}_variant_index"] for sample in samples])
        max_tokens = max(value.shape[1] for value in token_values)
        hidden_size = token_values[0].shape[-1]
        padded = torch.zeros((len(samples), 2, max_tokens, hidden_size), dtype=token_values[0].dtype)
        padded_mask = torch.zeros((len(samples), 2, max_tokens), dtype=torch.bool)
        for row, (value, mask) in enumerate(zip(token_values, masks)):
            length = value.shape[1]
            padded[row, :, :length] = value
            padded_mask[row, :, :length] = mask
        base[f"token_{modality}"] = padded
        base[f"token_{modality}_mask"] = padded_mask
        base[f"token_{modality}_variant_index"] = indices
    return base


def add_token_cache_to_loader(loader, token_index, split_name,
                              dna_model_name, protein_model_name, is_train=False,
                              batch_size=None):
    wrapped = _TokenCacheDataset(
        base_dataset=loader.dataset,
        cache_index=token_index,
        split_name=split_name,
        dna_model_name=dna_model_name,
        protein_model_name=protein_model_name,
    )
    return DataLoader(
        wrapped,
        batch_size=int(batch_size or loader.batch_size),
        shuffle=bool(is_train),
        num_workers=0,
        pin_memory=loader.pin_memory,
        generator=loader.generator,
        collate_fn=_token_collate,
    )


class _VariantCenteredWindow(nn.Module):
    def __init__(self, input_dim, model_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(input_dim * 2, model_dim)
        self.relative_position_embedding = nn.Embedding(129, model_dim)
        self.variant_embedding = nn.Parameter(torch.zeros(model_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.variant_embedding, std=0.02)

    def forward(self, tokens, mask, variant_indices, k_radius, modality_embedding):
        batch, alleles, length, hidden = tokens.shape
        if alleles != 2:
            raise ValueError(f"Expected ref/alt allele axis of length 2, got {alleles}")
        offsets = torch.arange(-int(k_radius), int(k_radius) + 1, device=tokens.device)
        positions = variant_indices.unsqueeze(-1) + offsets.view(1, 1, -1)
        in_bounds = (positions >= 0) & (positions < length)
        safe_positions = positions.clamp(0, max(length - 1, 0))
        gather_positions = safe_positions.unsqueeze(-1).expand(-1, -1, -1, hidden)
        local_tokens = tokens.gather(2, gather_positions)
        local_masks = mask.gather(2, safe_positions) & in_bounds

        ref_tokens, alt_tokens = local_tokens[:, 0], local_tokens[:, 1]
        ref_mask, alt_mask = local_masks[:, 0], local_masks[:, 1]
        # Pair ref/alt at each relative position, producing one window token
        # instead of passing two nearly duplicate local sequences downstream.
        ref_tokens = ref_tokens * ref_mask.unsqueeze(-1).to(ref_tokens.dtype)
        alt_tokens = alt_tokens * alt_mask.unsqueeze(-1).to(alt_tokens.dtype)
        paired = torch.cat((ref_tokens, alt_tokens), dim=-1).to(self.projection.weight.dtype)
        window_mask = ref_mask | alt_mask

        projected = self.projection(paired)
        # Keep the existing 129-entry embedding/checkpoint shape; larger
        # windows reuse the boundary position embedding beyond radius 64.
        relative_ids = (offsets + 64).clamp(
            min=0, max=self.relative_position_embedding.num_embeddings - 1
        )
        projected = projected + self.relative_position_embedding(relative_ids).unsqueeze(0)
        center_mask = offsets.eq(0).view(1, -1, 1)
        projected = projected + center_mask.to(projected.dtype) * self.variant_embedding
        projected = projected + modality_embedding.view(1, 1, -1)
        return self.dropout(projected), window_mask


class VariantCenteredTokenTransformer(nn.Module):
    """Shared Transformer over paired ref/alt windows centered on each variant."""

    def __init__(self, dna_dim, protein_dim, bio_dim, geom_dim, config):
        super().__init__()
        model_dim = int(config["model_dim"])
        dropout = float(config["dropout"])
        self.k_radius = int(config["k_radius"])
        self.dna_modality = nn.Parameter(torch.zeros(model_dim))
        self.protein_modality = nn.Parameter(torch.zeros(model_dim))
        self.dna_window = _VariantCenteredWindow(dna_dim, model_dim, dropout)
        self.protein_window = _VariantCenteredWindow(protein_dim, model_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=int(config["num_heads"]),
            dim_feedforward=model_dim * int(config["ffn_multiplier"]),
            dropout=float(config["attention_dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(config["num_layers"]))
        self.sequence_norm = nn.LayerNorm(model_dim)
        bio_input_dim = int(bio_dim) + int(geom_dim)
        self.bio_encoder = nn.Sequential(
            nn.LayerNorm(bio_input_dim),
            nn.Linear(bio_input_dim, int(config["bio_dim"])),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fused_dim = model_dim + int(config["bio_dim"])
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, max(model_dim // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(model_dim // 2, 32), 1),
        )

    def forward(self, v_dna, v_prot, bio_features, geom_features=None,
                token_dna=None, token_dna_mask=None, token_dna_variant_index=None,
                token_protein=None, token_protein_mask=None,
                token_protein_variant_index=None):
        if token_dna is None or token_protein is None:
            raise ValueError("VariantCenteredTokenTransformer requires DNA and protein token cache inputs.")
        dna, dna_mask = self.dna_window(
            token_dna, token_dna_mask, token_dna_variant_index,
            self.k_radius, self.dna_modality,
        )
        protein, protein_mask = self.protein_window(
            token_protein, token_protein_mask, token_protein_variant_index,
            self.k_radius, self.protein_modality,
        )
        tokens = torch.cat((dna, protein), dim=1)
        padding_mask = ~torch.cat((dna_mask, protein_mask), dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
        sequence = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        biological = bio_features if geom_features is None else torch.cat((bio_features, geom_features), dim=1)
        return self.classifier(torch.cat((self.sequence_norm(sequence), self.bio_encoder(biological)), dim=1)).squeeze(-1)


def _builder(dimensions, config):
    return VariantCenteredTokenTransformer(
        dna_dim=dimensions["token_dna"],
        protein_dim=dimensions["token_protein"],
        bio_dim=dimensions["bio"],
        geom_dim=dimensions["geometry"],
        config=config,
    )


def run_token_topk_optuna(context, base_config=None, search_space=None,
                          n_trials=None, display_fn=None):
    """Run the variant-centered token-window Transformer using the shared runner."""
    if context.get("token_embedding_cache_index") is None:
        raise KeyError(
            "token_embedding_cache_index is required. Run the token extraction cell first."
        )
    base = dict(DEFAULT_TOKEN_TOPK_CONFIG if base_config is None else base_config)
    space = dict(DEFAULT_TOKEN_TOPK_SEARCH_SPACE if search_space is None else search_space)
    cache_index = context["token_embedding_cache_index"]
    context = dict(context)
    context["input_data_signature"] = {
        "token_cache_index_path": cache_index.get("index_path"),
        "token_cache_root": cache_index.get("cache_root"),
        "token_models": TOKEN_MODEL_NAMES,
        "architecture": "paired_ref_alt_variant_centered_window_then_shared_transformer",
        "k_semantics": "k_tokens_per_side; window_length=2*k+1",
        "token_batch_size": context.get("token_batch_size"),
    }
    return run_architecture_optuna(
        context=context,
        base_config=base,
        search_space=space,
        n_trials=DEFAULT_TOKEN_TOPK_N_TRIALS if n_trials is None else int(n_trials),
        model_builder=_builder,
        architecture="Transformer_VariantCenteredWindow",
        variant_builder=lambda cfg: (
            f"radius_{cfg['k_radius']}_window_{2 * int(cfg['k_radius']) + 1}_layers_{cfg['num_layers']}_"
            f"dim_{cfg['model_dim']}"
        ),
        experiment_key="transformer_variant_centered_window",
        source="module06_variant_centered_token_transformer",
        display_fn=display_fn,
        manifest_architecture="paired_ref_alt_variant_centered_window_transformer_with_bio_head",
    )
