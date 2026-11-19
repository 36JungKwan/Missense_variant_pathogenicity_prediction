"""Resumable, in-memory variant-window cache for the FM-pair ablation."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

from core.module06_architecture_experiments.token_embedding_cache import _atomic_torch_save
from core.module06_architecture_experiments.token_topk_experiment import _TokenCacheDataset


def _window_shard(source_path, destination, radius):
    source_path = Path(source_path)
    stat = source_path.stat()
    signature = {
        "version": 1,
        "radius": radius,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }
    if destination.is_file():
        cached = torch.load(destination, map_location="cpu", weights_only=True)
        if cached["signature"] == signature:
            return cached

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    offsets = torch.arange(-radius, radius + 1)
    windows, masks = [], []
    for allele in ("ref", "alt"):
        hidden = source[f"token_hidden_{allele}"]
        mask = source[f"attention_mask_{allele}"].bool()
        indices = source[f"variant_token_index_{allele}"].long()
        positions = indices[:, None] + offsets
        valid = (positions >= 0) & (positions < hidden.shape[1])
        positions = positions.clamp(0, hidden.shape[1] - 1)
        windows.append(hidden.gather(
            1, positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        ))
        masks.append(mask.gather(1, positions) & valid)
    cached = {
        "signature": signature,
        "variant_ids": [str(value) for value in source["variant_ids"]],
        "hidden": torch.stack(windows, dim=1),
        "mask": torch.stack(masks, dim=1),
    }
    _atomic_torch_save(cached, destination)
    return cached


class _WindowTokenDataset(_TokenCacheDataset):
    def __init__(self, base_dataset, cache_index, split_name, dna_model_name,
                 protein_model_name, radius):
        super().__init__(
            base_dataset, cache_index, split_name, dna_model_name, protein_model_name
        )
        self.radius = int(radius)
        if self.radius < 0:
            raise ValueError("Token window radius must be non-negative")
        self.windows = {}
        self.window_masks = {}
        base_ids = self.df["Variant_ID"].astype(str).tolist()
        for modality in ("dna", "protein"):
            started = time.perf_counter()
            grouped = defaultdict(list)
            for output_row, variant_id in enumerate(base_ids):
                path, source_row = self.paths[modality][variant_id]
                grouped[path].append((output_row, source_row, variant_id))
            count = len(base_ids)
            width = 2 * self.radius + 1
            hidden_size = self.token_hidden_sizes[modality]
            windows = torch.empty((count, 2, width, hidden_size), dtype=torch.float16)
            masks = torch.empty((count, 2, width), dtype=torch.bool)
            for path, rows in tqdm(
                grouped.items(), desc=f"Window cache {modality}/{split_name}", unit="shard"
            ):
                path = Path(path)
                destination = path.parent / f"_window_cache_v1_k{self.radius}" / path.name
                shard = _window_shard(path, destination, self.radius)
                expected_shape = (len(shard["variant_ids"]), 2, width, hidden_size)
                if (
                    shard["hidden"].shape != expected_shape
                    or shard["hidden"].dtype != torch.float16
                    or shard["mask"].shape != expected_shape[:3]
                    or shard["mask"].dtype != torch.bool
                ):
                    raise ValueError(f"Invalid window-cache tensor shape or dtype: {destination}")
                for _, source_row, variant_id in rows:
                    if shard["variant_ids"][source_row] != variant_id:
                        raise ValueError(
                            f"Token cache row order mismatch for {variant_id!r}: {path}"
                        )
                output_rows = torch.tensor([row[0] for row in rows], dtype=torch.long)
                source_rows = torch.tensor([row[1] for row in rows], dtype=torch.long)
                windows[output_rows] = shard["hidden"][source_rows]
                masks[output_rows] = shard["mask"][source_rows]
            self.windows[modality] = windows
            self.window_masks[modality] = masks
            print(
                f"[Window cache] {modality}/{split_name}: {count} samples ready in "
                f"{time.perf_counter() - started:.1f}s"
            )

    def __getitem__(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        batch = default_collate([self.base_dataset[index] for index in indices])
        rows = torch.as_tensor(indices, dtype=torch.long)
        for modality in ("dna", "protein"):
            batch[f"token_{modality}"] = self.windows[modality].index_select(0, rows)
            batch[f"token_{modality}_mask"] = self.window_masks[modality].index_select(0, rows)
            batch[f"token_{modality}_variant_index"] = torch.full(
                (len(indices), 2), self.radius, dtype=torch.long
            )
        return batch


def add_window_cache_to_loader(loader, token_index, split_name, dna_model_name,
                               protein_model_name, is_train, batch_size, radius):
    dataset = _WindowTokenDataset(
        loader.dataset, token_index, split_name, dna_model_name, protein_model_name, radius
    )
    sampler = (
        RandomSampler(dataset, generator=loader.generator)
        if is_train else SequentialSampler(dataset)
    )
    batches = BatchSampler(sampler, batch_size=int(batch_size or loader.batch_size), drop_last=False)
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=batches,
        num_workers=0,
        pin_memory=loader.pin_memory,
        generator=loader.generator,
    )
