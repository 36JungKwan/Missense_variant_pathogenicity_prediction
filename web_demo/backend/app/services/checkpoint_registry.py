"""Checkpoint Registry: ánh xạ tổ hợp modality -> file .pth của TriMVPP, kiểm tra và cache model đã nạp."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from app.core.config import settings
from app.core.constants import BIO_COLUMNS, GEOM_COLUMNS, MODALITY_ORDER
from app.core.errors import (
    CheckpointIncompatibleError,
    CheckpointMissingError,
    NoModalitySelectedError,
    UnsupportedCombinationError,
)
from app.ml_core.trimvpp_model import TriMVPP

logger = logging.getLogger("app.services.checkpoint_registry")

ARCHITECTURE = "TriMVPP"
VALID_TOGGLE_KEYS = ("111", "110", "101", "011", "100", "010", "001")


def _key_to_modalities(key: str) -> Tuple[str, ...]:
    """'101' -> (dna, bio, geom)."""
    key = key.strip()
    if key not in VALID_TOGGLE_KEYS:
        raise ValueError(f"Invalid combination key '{key}'; expected one of {', '.join(VALID_TOGGLE_KEYS)}.")
    selected = {"dna"} if key[0] == "1" else set()
    if key[1] == "1":
        selected.add("prot")
    if key[2] == "1":
        selected.update({"bio", "geom"})
    return tuple(m for m in MODALITY_ORDER if m in selected)


def _branches(modalities: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(b for b, m in (("dna", "dna"), ("protein", "prot"), ("bio", "bio")) if m in modalities)


def resolve_device() -> torch.device:
    wanted = settings.INFERENCE_DEVICE.lower()
    if wanted in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class LoadedModel:
    model: torch.nn.Module
    architecture: str
    modalities: Tuple[str, ...]
    bio_columns: List[str]
    geom_columns: List[str]
    threshold: float
    tag: str  # định danh checkpoint, dùng làm khoá cache trong bảng predictions
    path: Path
    device: torch.device
    test_mcc: Optional[float] = None

    @property
    def tabular_columns(self) -> List[str]:
        return self.bio_columns + self.geom_columns

    def forward_logit(self, **inputs: torch.Tensor) -> torch.Tensor:
        """Đầu vào đúng như TriMVPP.forward; trả logit (B,)."""
        return self.model(**inputs).reshape(-1)


def _read_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(hasattr(v, "shape") for v in state.values()):
        raise ValueError("file does not contain a raw state_dict")
    return state


class CheckpointRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[Path, float], LoadedModel] = {}
        self._entries: Dict[Tuple[str, ...], Tuple[str, Path]] = {}
        for key, raw_path in settings.MODEL_CHECKPOINT_REGISTRY.items():
            self._entries[_key_to_modalities(key)] = (key, settings.resolve_path(raw_path))
        self._thresholds = {
            _key_to_modalities(k): float(v) for k, v in settings.MODEL_DECISION_THRESHOLDS.items()
        }
        self._test_mcc = {_key_to_modalities(k): float(v) for k, v in settings.MODEL_TEST_MCC.items()}

    def get(self, modalities: Tuple[str, ...]) -> LoadedModel:
        modalities = tuple(m for m in MODALITY_ORDER if m in modalities)
        if not modalities:
            raise NoModalitySelectedError("At least one modality (DNA, Protein, Bio+Geom) must be enabled.")
        entry = self._entries.get(modalities)
        if entry is None:
            raise UnsupportedCombinationError(
                f"No checkpoint is registered for the combination {list(modalities)}. "
                "Bio and Geometry are toggled together (use_biogeom); valid combinations are "
                f"{', '.join(VALID_TOGGLE_KEYS)} over (use_dna, use_prot, use_biogeom).",
                active_modalities=list(modalities),
            )
        key, path = entry
        if not path.is_file():
            raise CheckpointMissingError(
                f"Checkpoint for combination {key} {list(modalities)} is not on disk yet. "
                f"Copy the trained .pth to '{path}' to activate it (no code change needed).",
                status="checkpoint_missing",
                combination=key,
                active_modalities=list(modalities),
                expected_path=str(path),
            )
        cache_key = (path, path.stat().st_mtime)
        with self._lock:
            if cache_key not in self._cache:
                self._cache[cache_key] = self._load(path, modalities)
            return self._cache[cache_key]

    def _load(self, path: Path, modalities: Tuple[str, ...]) -> LoadedModel:
        if modalities not in self._thresholds:
            raise CheckpointIncompatibleError(f"No decision threshold configured for {list(modalities)}.")
        device = resolve_device()
        try:
            model = TriMVPP(_branches(modalities))
            model.load_state_dict(_read_state_dict(path), strict=True)
        except Exception as exc:
            raise CheckpointIncompatibleError(f"Cannot load checkpoint {path.name}: {exc}") from exc

        model.to(device).eval()
        has_bio = "bio" in modalities
        threshold = self._thresholds[modalities]
        logger.info("Loaded checkpoint %s as %s for %s (threshold=%s) on %s",
                    path.name, ARCHITECTURE, list(modalities), threshold, device)
        return LoadedModel(
            model=model,
            architecture=ARCHITECTURE,
            modalities=modalities,
            bio_columns=list(BIO_COLUMNS) if has_bio else [],
            geom_columns=list(GEOM_COLUMNS) if has_bio else [],
            threshold=threshold,
            tag=f"{path.name}@{int(path.stat().st_mtime)}",
            path=path,
            device=device,
            test_mcc=self._test_mcc.get(modalities),
        )

    def status(self) -> List[Dict[str, Any]]:
        rows = []
        for modalities, (key, path) in self._entries.items():
            rows.append({
                "combination": key,
                "active_modalities": list(modalities),
                "checkpoint": str(path),
                "status": "available" if path.is_file() else "checkpoint_missing",
                "decision_threshold": self._thresholds.get(modalities),
                "test_mcc": self._test_mcc.get(modalities),
            })
        return rows


checkpoint_registry = CheckpointRegistry()
