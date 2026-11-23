"""Chú thích biến thể bằng Ensembl VEP offline chạy trong Docker (`docker exec <container> vep ...`)."""
from __future__ import annotations

import logging
import math
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.core.config import settings
from app.core.constants import BIO_CORE_COLUMNS, VariantKey
from app.core.errors import NonMissenseVariantError, VepUnavailableError
from app.core.log import timed_stage

logger = logging.getLogger("app.services.vep_runner")

DBNSFP_FIELDS = [
    "GERP++_NR", "GERP++_RS", "GERP_92_mammals",
    "phastCons100way_vertebrate", "phastCons17way_primate", "phastCons470way_mammalian",
    "phyloP100way_vertebrate", "phyloP17way_primate", "phyloP470way_mammalian",
    "AlphaMissense_score", "REVEL_score", "clinvar_clnsig",
]


@dataclass
class VepRecord:
    variant_id: str
    consequence: str
    gene: str = "UNKNOWN"
    transcript: Optional[str] = None
    ensp: Optional[str] = None
    protein_position: Optional[int] = None
    ref_aa: Optional[str] = None
    alt_aa: Optional[str] = None
    hgvsp: Optional[str] = None
    clinvar_raw: Optional[str] = None
    alphamissense: Optional[float] = None
    revel: Optional[float] = None
    bio_core: Dict[str, float] = field(default_factory=dict)  # 11 giá trị thô, thiếu = NaN

    @property
    def is_missense(self) -> bool:
        return "missense_variant" in self.consequence


def _numbers(raw: Optional[str]) -> List[float]:
    values = []
    for token in (raw or "").replace(";", ",").replace("&", ",").split(","):
        try:
            value = float(token)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _first(raw: Optional[str]) -> float:
    values = _numbers(raw)
    return values[0] if values else float("nan")


def _max(raw: Optional[str]) -> Optional[float]:
    values = _numbers(raw)
    return max(values) if values else None


def _clean(raw: Optional[str]) -> Optional[str]:
    return None if raw in (None, "", "-", ".") else raw


def ensure_missense(record: VepRecord) -> VepRecord:
    """Missense Gatekeeper: chặn mọi biến thể mà VEP (--pick, MANE Select/Canonical) không gán missense_variant."""
    if not record.is_missense:
        raise NonMissenseVariantError(record.variant_id, record.consequence)
    return record


def _build_command(container_in: str, container_out: str, n_variants: int = 1) -> List[str]:
    plugin = ",".join(["dbNSFP", settings.DBNSFP_CONTAINER_PATH, *DBNSFP_FIELDS])
    # Chỉ song song hoá khi lô đủ lớn; cấu hình chú thích giữ nguyên để đặc trưng khớp lúc huấn luyện.
    fork = (["--fork", str(settings.VEP_FORK_WORKERS)]
            if n_variants > settings.VEP_FORK_THRESHOLD and settings.VEP_FORK_WORKERS > 1 else [])
    return [
        "docker", "exec", settings.VEP_CONTAINER_NAME, "vep",
        "-i", container_in, "-o", container_out,
        "--format", "vcf", "--tab", "--force_overwrite", "--no_stats",
        "--offline", "--cache", "--cache_version", str(settings.VEP_CACHE_VERSION),
        "--assembly", "GRCh38",
        "--dir_cache", settings.VEP_CONTAINER_CACHE_DIR,
        "--fasta", settings.VEP_CONTAINER_FASTA,
        "--dir_plugins", settings.VEP_CONTAINER_PLUGIN_DIR,
        "--mane_select", "--canonical", "--pick",
        "--symbol", "--protein", "--hgvs",
        "--af", "--af_gnomade", "--check_existing",
        "--plugin", plugin,
        *fork,
    ]


def _parse_row(row: Dict[str, str]) -> VepRecord:
    record = VepRecord(
        variant_id=row["#Uploaded_variation"],
        consequence=_clean(row.get("Consequence")) or "unknown",
        gene=_clean(row.get("SYMBOL")) or "UNKNOWN",
        transcript=_clean(row.get("Feature")),
        ensp=_clean(row.get("ENSP")),
        clinvar_raw=_clean(row.get("clinvar_clnsig")) or _clean(row.get("CLIN_SIG")),
        alphamissense=_max(row.get("AlphaMissense_score")),
        revel=_max(row.get("REVEL_score")),
    )
    position = _clean(row.get("Protein_position"))
    if position and position.isdigit():
        record.protein_position = int(position)
    amino = (_clean(row.get("Amino_acids")) or "").split("/")
    if len(amino) == 2 and len(amino[0]) == 1 and len(amino[1]) == 1:
        record.ref_aa, record.alt_aa = amino
    hgvsp = _clean(row.get("HGVSp"))
    if hgvsp:
        record.hgvsp = hgvsp.split(":", 1)[-1].replace("%3D", "=")
    record.bio_core = {column: _first(row.get(column)) for column in BIO_CORE_COLUMNS}
    return record


def annotate_variants(variants: Sequence[VariantKey]) -> Dict[str, VepRecord]:
    """Chạy VEP một lần cho cả danh sách; trả về {variant_id: VepRecord}."""
    if not variants:
        return {}
    tmp_dir = Path(settings.VEP_TMP_DIR)
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VepUnavailableError(f"Cannot access VEP_TMP_DIR '{tmp_dir}': {exc}") from exc

    job = f"rbnet_{uuid.uuid4().hex}"
    host_in, host_out = tmp_dir / f"{job}.vcf", tmp_dir / f"{job}.tsv"
    container_dir = settings.VEP_TMP_CONTAINER_DIR.rstrip("/")

    unique = {v.variant_id: v for v in variants}
    lines = ["##fileformat=VCFv4.2", "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"])]
    for variant_id, v in unique.items():
        lines.append("\t".join([v.chrom, str(v.pos), variant_id, v.ref, v.alt, ".", ".", "."]))
    host_in.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")

    command = _build_command(f"{container_dir}/{job}.vcf", f"{container_dir}/{job}.tsv", len(unique))
    started = time.perf_counter()
    try:
        try:
            with timed_stage("vep"):
                proc = subprocess.run(command, capture_output=True, text=True, timeout=settings.VEP_TIMEOUT_SECONDS)
        except FileNotFoundError as exc:
            raise VepUnavailableError("Docker CLI not found on PATH; cannot run Ensembl VEP.") from exc
        except subprocess.TimeoutExpired as exc:
            raise VepUnavailableError(f"VEP timed out after {settings.VEP_TIMEOUT_SECONDS}s.") from exc
        if proc.returncode != 0 or not host_out.is_file():
            tail = (proc.stderr or proc.stdout or "").strip()[-600:]
            raise VepUnavailableError(
                f"VEP container '{settings.VEP_CONTAINER_NAME}' failed (exit {proc.returncode}). "
                f"Is Docker Desktop running and the container started? {tail}"
            )

        records: Dict[str, VepRecord] = {}
        header: Optional[List[str]] = None
        for line in host_out.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("##") or not line:
                continue
            cells = line.split("\t")
            if line.startswith("#"):
                header = cells
                continue
            if header is None:
                continue
            record = _parse_row(dict(zip(header, cells)))
            records.setdefault(record.variant_id, record)
    finally:
        for path in (host_in, host_out, Path(f"{host_out}_warnings.txt")):
            path.unlink(missing_ok=True)

    for variant_id in unique:
        records.setdefault(variant_id, VepRecord(variant_id=variant_id, consequence="not_annotated_by_vep"))
    logger.info("VEP annotated %d variant(s) in %d ms%s", len(unique),
                (time.perf_counter() - started) * 1000, " (--fork)" if "--fork" in command else "")
    return records
