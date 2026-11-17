"""Checkpoint Registry: ánh xạ tổ hợp modality -> file .pth, tự nhận diện kiến trúc và cache model.

Kiến trúc được suy ra từ chính state_dict nên không cần sửa code khi copy thêm checkpoint:
  * `MultiStrategyFusionModel` (ml_core/fusion_model.py)  - khoá `dna_proj.*`, `bio_geom_upsample.*`
  * `RouterExpertCorrection`  (router_expert_model.py)    - khoá `dna_projection.*`, `router.*`
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from app.core.config import settings
from app.core.constants import BIO_COLUMN_SETS, GEOM_COLUMNS, MODALITY_ORDER
from app.core.errors import (
    CheckpointIncompatibleError,
    CheckpointMissingError,
    NoModalitySelectedError,
    UnsupportedCombinationError,
)
from app.ml_core.fusion_model import MultiStrategyFusionModel
from app.services.router_expert_model import RouterExpertCorrection

VALID_TOGGLE_KEYS = ("111", "110", "101", "011", "100", "010", "001")


def _key_to_modalities(key: str) -> Tuple[str, ...]:
    """'101' -> (dna, bio, geom); 'dna,prot,bio' -> (dna, prot, bio)."""
    key = key.strip().lower()
    if len(key) == 3 and set(key) <= {"0", "1"}:
        selected = set()
        if key[0] == "1":
            selected.add("dna")
        if key[1] == "1":
            selected.add("prot")
        if key[2] == "1":
            selected.update({"bio", "geom"})
    else:
        selected = {part.strip() for part in key.replace("+", ",").split(",") if part.strip()}
        unknown = selected - set(MODALITY_ORDER)
        if unknown:
            raise ValueError(f"Unknown modality in MODEL_CHECKPOINT_REGISTRY key '{key}': {sorted(unknown)}")
    return tuple(m for m in MODALITY_ORDER if m in selected)


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

    @property
    def tabular_columns(self) -> List[str]:
        return self.bio_columns + self.geom_columns

    def forward_logit(
        self,
        dna: Optional[torch.Tensor],
        prot: Optional[torch.Tensor],
        tabular: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """dna/prot: (B, D) vector delta E_alt - E_ref; tabular: (B, n_bio + n_geom) đã chuẩn hoá."""
        n_bio = len(self.bio_columns)
        bio = tabular[:, :n_bio] if tabular is not None and n_bio else None
        geom = tabular[:, n_bio:] if tabular is not None and self.geom_columns else None
        if self.architecture == "RouterExpertCorrection":
            return self.model(dna, prot, bio, geom).reshape(-1)
        return self.model(
            v_dna=dna if "dna" in self.modalities else None,
            v_prot=prot if "prot" in self.modalities else None,
            bio_features=bio,
            geom_features=geom,
        ).reshape(-1)


def _extract_state_dict(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict"):
            if isinstance(ckpt.get(key), dict):
                meta = {k: v for k, v in ckpt.items() if k != key}
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict) or not all(hasattr(v, "shape") for v in ckpt.values()):
        raise ValueError("file does not contain a state_dict")
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
    return state, meta


def _split_tabular(total_dim: int, modalities: Tuple[str, ...], path: Path) -> Tuple[List[str], List[str]]:
    geom_cols = list(GEOM_COLUMNS) if "geom" in modalities else []
    bio_dim = total_dim - len(geom_cols)
    if "bio" not in modalities:
        if bio_dim != 0:
            raise CheckpointIncompatibleError(
                f"{path.name}: tabular input has {total_dim} dims but the requested combination has no Bio branch."
            )
        return [], geom_cols
    if bio_dim not in BIO_COLUMN_SETS:
        raise CheckpointIncompatibleError(
            f"{path.name}: Bio branch has {bio_dim} dims (tabular {total_dim} - geometry {len(geom_cols)}); "
            f"supported Bio dims are {sorted(BIO_COLUMN_SETS)}."
        )
    return list(BIO_COLUMN_SETS[bio_dim]), geom_cols


def _build_router_expert(state, meta, modalities, path) -> Tuple[torch.nn.Module, List[str], List[str]]:
    if not {"dna", "prot", "bio"} <= set(modalities):
        raise CheckpointIncompatibleError(
            f"{path.name} is a RouterExpertCorrection checkpoint, which always needs DNA + Protein + Bio; "
            f"it cannot serve the combination {list(modalities)}."
        )
    bio_cols, geom_cols = _split_tabular(state["biological_projection.0.weight"].shape[1], modalities, path)
    hparams = meta.get("hparams") or meta.get("config") or {}
    model = RouterExpertCorrection(
        dna_dim=state["dna_projection.0.weight"].shape[1],
        protein_dim=state["protein_projection.0.weight"].shape[1],
        biological_dim=state["biological_projection.0.weight"].shape[1],
        embedding_dim=state["dna_projection.0.weight"].shape[0],
        expert_hidden_dim=state["dna_expert.input_projection.0.weight"].shape[0],
        router_hidden_dim=state["router.0.weight"].shape[0],
        correction_hidden_dim=state["correction.0.weight"].shape[0],
        router_temperature=float(hparams.get("router_temperature", settings.ROUTER_TEMPERATURE)),
    )
    return model, bio_cols, geom_cols


def _build_multi_strategy(state, meta, modalities, path) -> Tuple[torch.nn.Module, List[str], List[str]]:
    """Dựng MultiStrategyFusionModel nguyên bản; chỉ truyền những tham số đọc được từ checkpoint."""
    kwargs: Dict[str, Any] = {"active_modalities": list(modalities)}
    has_dna, has_prot = "dna_proj.0.weight" in state, "prot_proj.0.weight" in state
    if has_dna != ("dna" in modalities) or has_prot != ("prot" in modalities):
        raise CheckpointIncompatibleError(
            f"{path.name} was trained with dna={has_dna}, prot={has_prot}, "
            f"which does not match the requested combination {list(modalities)}."
        )
    seq_key = "dna_proj.0.weight" if has_dna else ("prot_proj.0.weight" if has_prot else None)
    kwargs["dna_in_dim"] = state["dna_proj.0.weight"].shape[1] if has_dna else 0
    kwargs["prot_in_dim"] = state["prot_proj.0.weight"].shape[1] if has_prot else 0
    if seq_key:
        kwargs["fusion_dim"] = state[seq_key].shape[0]

    bio_cols: List[str] = []
    geom_cols: List[str] = []
    has_tabular = "bio_geom_upsample.0.weight" in state
    if has_tabular != bool({"bio", "geom"} & set(modalities)):
        raise CheckpointIncompatibleError(
            f"{path.name}: Bio/Geom branch presence does not match the requested combination {list(modalities)}."
        )
    if has_tabular:
        weight = state["bio_geom_upsample.0.weight"]
        bio_cols, geom_cols = _split_tabular(weight.shape[1], modalities, path)
        kwargs["bio_upsample_dim"] = weight.shape[0]
        if bio_cols:
            kwargs["bio_in_dim"] = len(bio_cols)

    # fusion_strategy: ưu tiên metadata lưu kèm, sau đó suy ra từ tham số; không thấy gì thì để mặc định của class.
    strategy = (meta.get("hparams") or meta.get("config") or {}).get("fusion_strategy") or meta.get("fusion_strategy")
    if not strategy:
        if "attn_p2d.in_proj_weight" in state:
            strategy = "cross_attention"
        elif "modality_embed" in state:
            strategy = "transformer"
        elif "gate.weight" in state:
            strategy = "gating"
    if strategy:
        kwargs["fusion_strategy"] = strategy
    return MultiStrategyFusionModel(**kwargs), bio_cols, geom_cols


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
        device = resolve_device()
        try:
            state, meta = _extract_state_dict(torch.load(path, map_location="cpu", weights_only=False))
            if "biological_projection.0.weight" in state and "router.0.weight" in state:
                architecture = "RouterExpertCorrection"
                model, bio_cols, geom_cols = _build_router_expert(state, meta, modalities, path)
            elif "classifier_hidden.0.weight" in state and "output_head.weight" in state:
                architecture = "MultiStrategyFusionModel"
                model, bio_cols, geom_cols = _build_multi_strategy(state, meta, modalities, path)
            else:
                raise ValueError("state_dict matches neither MultiStrategyFusionModel nor RouterExpertCorrection")
            model.load_state_dict(state, strict=True)
        except CheckpointIncompatibleError:
            raise
        except Exception as exc:
            raise CheckpointIncompatibleError(f"Cannot load checkpoint {path.name}: {exc}") from exc

        model.to(device).eval()
        threshold = float(meta.get("threshold", self._thresholds.get(modalities, 0.5)))
        print(f"[CheckpointRegistry] Loaded {path.name} as {architecture} for {list(modalities)} "
              f"(bio={len(bio_cols)}, geom={len(geom_cols)}, threshold={threshold}) on {device}")
        return LoadedModel(
            model=model,
            architecture=architecture,
            modalities=modalities,
            bio_columns=bio_cols,
            geom_columns=geom_cols,
            threshold=threshold,
            tag=f"{path.name}@{int(path.stat().st_mtime)}",
            path=path,
            device=device,
        )

    def status(self) -> List[Dict[str, Any]]:
        rows = []
        for modalities, (key, path) in self._entries.items():
            rows.append({
                "combination": key,
                "active_modalities": list(modalities),
                "checkpoint": str(path),
                "status": "available" if path.is_file() else "checkpoint_missing",
            })
        return rows


checkpoint_registry = CheckpointRegistry()
