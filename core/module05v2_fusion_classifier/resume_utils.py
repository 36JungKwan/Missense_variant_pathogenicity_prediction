"""Small helpers for safe, configuration-aware experiment resumption."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
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

def _without_mutable_paths(config: dict, mutable_paths):
    """Copy a JSON-safe config while removing explicitly mutable leaf paths."""
    value = _jsonable(config)
    for path_parts in mutable_paths or ():
        current = value
        for part in tuple(path_parts)[:-1]:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(str(part))
        if isinstance(current, dict) and path_parts:
            current.pop(str(tuple(path_parts)[-1]), None)
    return value


def ensure_stage_signature(
    stage_dir: str | Path,
    stage_name: str,
    config: dict,
    mutable_paths=(),
):
    """Create/check a manifest, optionally allowing selected config leaves to change.

    Mutable leaves are still written to the manifest, but excluded from its
    compatibility hash. This is intended for settings such as batch size that
    can change safely when resuming from a saved epoch boundary.
    """
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config_value = _jsonable(config)
    signature_value = _without_mutable_paths(config_value, mutable_paths)
    canonical = json.dumps(signature_value, sort_keys=True, separators=(",", ":"))
    manifest = {
        "version": MANIFEST_VERSION,
        "stage": stage_name,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "config": config_value,
    }
    path = directory / "_resume_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_signature = _without_mutable_paths(
            existing.get("config", {}), mutable_paths
        )
        existing_canonical = json.dumps(
            existing_signature, sort_keys=True, separators=(",", ":")
        )
        existing_hash = hashlib.sha256(existing_canonical.encode("utf-8")).hexdigest()
        if existing.get("stage") != stage_name or existing_hash != manifest["config_sha256"]:
            raise ValueError(
                f"Cannot resume {stage_name} in {directory}: its saved configuration differs. "
                "Use the original settings or start a new batch run."
            )
        # Migrate compatible older manifests and record the latest mutable value.
        if existing.get("config_sha256") != manifest["config_sha256"] or existing.get("config") != config_value:
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, path)
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


def replace_file_with_retry(temporary, destination):
    """Replace an output file, retrying transient Windows file-lock errors."""
    destination = Path(destination)
    for attempt in range(5):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 4:
                raise PermissionError(
                    f"Unable to replace output file '{destination}'. Close any application "
                    "using this file and retry."
                ) from error
            time.sleep(0.1 * (2 ** attempt))


def save_progress_csv(rows, path):
    """Atomically persist completed candidate rows during a long-running stage."""
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        pd.DataFrame(rows).to_csv(temporary, index=False)
        replace_file_with_retry(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

def upsert_record(rows, row, keys):
    """Replace a candidate's previous progress record, avoiding duplicate rows."""
    return [item for item in rows if not all(item.get(key) == row.get(key) for key in keys)] + [row]
