"""Small helpers for safe, configuration-aware experiment resumption."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1

def reject_legacy_stage_artifacts(stage_dir: str | Path, stage_name: str):
    """Prevent changed training-selection policies from silently reusing old checkpoints."""
    directory = Path(stage_dir)
    manifest = directory / "_resume_manifest.json"
    if directory.exists() and not manifest.exists() and any(path.is_file() for path in directory.rglob("*")):
        raise ValueError(
            f"Cannot resume legacy {stage_name} artifacts in {directory}: they were created "
            "without the current validation-optimized checkpoint policy. Start a new batch run."
        )

def _jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)

def ensure_stage_signature(stage_dir: str | Path, stage_name: str, config: dict):
    """Create/check a stage manifest; reject reusing artifacts with changed settings."""
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config_value = _jsonable(config)
    canonical = json.dumps(config_value, sort_keys=True, separators=(",", ":"))
    manifest = {
        "version": MANIFEST_VERSION,
        "stage": stage_name,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "config": config_value,
    }
    path = directory / "_resume_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("stage") != stage_name or existing.get("config_sha256") != manifest["config_sha256"]:
            raise ValueError(
                f"Cannot resume {stage_name} in {directory}: its saved configuration differs. "
                "Use the original settings or start a new batch run."
            )
        return path
    legacy_artifacts = any(p.is_file() and p.name != path.name for p in directory.iterdir())
    if legacy_artifacts:
        warnings.warn(
            f"No resume manifest exists for legacy stage {stage_name} in {directory}; "
            "the current settings are being recorded, but prior settings cannot be verified.",
            RuntimeWarning,
            stacklevel=2,
        )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def save_progress_csv(rows, path):
    """Atomically persist completed candidate rows during a long-running stage."""
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)

def upsert_record(rows, row, keys):
    """Replace a candidate's previous progress record, avoiding duplicate rows."""
    return [item for item in rows if not all(item.get(key) == row.get(key) for key in keys)] + [row]
