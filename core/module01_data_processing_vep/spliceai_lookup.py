"""Direct, indexed lookup of pre-computed SpliceAI scores for GRCh38 SNVs.

This module deliberately has no VEP or external bioinformatics dependency. It
reads a bgzip-compressed VCF through its tabix index, so the complete 28 GB
score file is never loaded into memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import struct
from typing import Iterable

import numpy as np
import pandas as pd


SPLICEAI_FEATURE_COLUMNS = [
    "SpliceAI_pred_DS_AG",
    "SpliceAI_pred_DS_AL",
    "SpliceAI_pred_DS_DG",
    "SpliceAI_pred_DS_DL",
    "SpliceAI_pred_DP_AG",
    "SpliceAI_pred_DP_AL",
    "SpliceAI_pred_DP_DG",
    "SpliceAI_pred_DP_DL",
    "SpliceAI_pred_DS_max",
]

_BASES = frozenset({"A", "C", "G", "T"})
_SCORE_COLUMNS = SPLICEAI_FEATURE_COLUMNS[:4]
_POSITION_COLUMNS = SPLICEAI_FEATURE_COLUMNS[4:8]


def canonical_chromosome(value: object) -> str | None:
    """Return a stable chromosome representation for matching VCF contigs."""
    if pd.isna(value):
        return None
    chrom = str(value).strip().upper()
    if chrom.startswith("CHR"):
        chrom = chrom[3:]
    if chrom == "M":
        return "MT"
    return chrom or None


def _as_float(value: str) -> float:
    if value in {"", "."}:
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def _parse_spliceai_annotation(annotation: str) -> np.ndarray | None:
    """Parse one pipe-delimited SpliceAI annotation into 8 numeric values.

    Public SpliceAI releases use either ``GENE|DS...|DP...`` or
    ``ALLELE|GENE|DS...|DP...``. Searching for the first contiguous numeric
    run makes the parser compatible with both layouts.
    """
    fields = annotation.split("|")
    for start in range(len(fields) - 7):
        values = np.asarray([_as_float(item) for item in fields[start : start + 8]], dtype=float)
        if np.isfinite(values[:4]).all():
            return values
    return None


def parse_spliceai_info(info: str, alt: str) -> np.ndarray | None:
    """Return the annotation with the largest delta score for one ALT allele.

    A locus can have several transcript/gene annotations. Keeping all four
    delta scores and four delta positions from the same winning annotation
    preserves their biological relationship. Ties retain VCF order.
    """
    info_fields = {}
    for item in info.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        info_fields[key.lower()] = value

    raw = next(
        (info_fields[key] for key in ("spliceai", "spliceai_pred", "spliceai_scores") if key in info_fields),
        None,
    )
    if not raw:
        return None

    candidates: list[np.ndarray] = []
    for annotation in raw.split(","):
        fields = annotation.split("|")
        # In the allele-prefixed schema, ignore annotations belonging to a
        # different ALT allele at a multiallelic site.
        if len(fields) >= 10 and fields[0].upper() in _BASES and fields[0].upper() != alt:
            continue
        values = _parse_spliceai_annotation(annotation)
        if values is not None:
            candidates.append(values)

    if not candidates:
        return None
    return max(candidates, key=lambda values: float(np.nanmax(values[:4])))


def _vcf_contig_aliases(contigs: Iterable[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for contig in contigs:
        canonical = canonical_chromosome(contig)
        if canonical and canonical not in aliases:
            aliases[canonical] = contig
    return aliases


class NativeTabixReader:
    """Minimal read-only TBI/BGZF reader for coordinate-sorted VCF files.

    ``pysam`` has no supported Windows wheel in the current environment. This
    reader implements only the indexed range lookup required here: it parses
    the compact TBI index once, then decompresses just the BGZF blocks covered
    by each query chunk with Python's standard library.
    """

    _MIN_SHIFT = 14
    _DEPTH = 5

    def __init__(self, vcf_path: Path):
        self.vcf_path = vcf_path
        self._vcf_handle = None
        self._references = self._load_index(Path(f"{vcf_path}.tbi"))
        self.contigs = tuple(reference["name"] for reference in self._references)
        self._contig_to_index = {contig: index for index, contig in enumerate(self.contigs)}

    @staticmethod
    def _load_index(index_path: Path) -> list[dict[str, object]]:
        with gzip.open(index_path, "rb") as handle:
            raw = handle.read()
        view = memoryview(raw)
        offset = 0

        if bytes(view[:4]) != b"TBI\x01":
            raise ValueError(f"Tabix index không đúng định dạng TBI: {index_path}")
        offset += 4
        n_ref = struct.unpack_from("<i", view, offset)[0]
        offset += 4
        # format, sequence/start/end columns, meta char, skipped lines, names length
        _, _, _, _, _, _, names_length = struct.unpack_from("<7i", view, offset)
        offset += struct.calcsize("<7i")
        raw_names = bytes(view[offset : offset + names_length]).decode("utf-8").split("\x00")
        names = raw_names[:-1]
        offset += names_length
        if len(names) != n_ref:
            raise ValueError(f"Tabix index có {n_ref} references nhưng chỉ đọc được {len(names)} tên contig.")

        references = []
        for name in names:
            n_bins = struct.unpack_from("<i", view, offset)[0]
            offset += 4
            bins: dict[int, list[tuple[int, int]]] = {}
            for _ in range(n_bins):
                bin_id, n_chunks = struct.unpack_from("<Ii", view, offset)
                offset += struct.calcsize("<Ii")
                chunks = []
                for _ in range(n_chunks):
                    chunk_start, chunk_end = struct.unpack_from("<QQ", view, offset)
                    offset += struct.calcsize("<QQ")
                    chunks.append((chunk_start, chunk_end))
                bins[bin_id] = chunks

            n_intervals = struct.unpack_from("<i", view, offset)[0]
            offset += 4
            intervals = list(struct.unpack_from(f"<{n_intervals}Q", view, offset)) if n_intervals else []
            offset += n_intervals * struct.calcsize("<Q")
            references.append({"name": name, "bins": bins, "intervals": intervals})
        return references

    @classmethod
    def _region_bins(cls, start: int, end: int) -> list[int]:
        """Return TBI bins intersecting the zero-based half-open query range."""
        if end <= start:
            return []
        end -= 1
        bins = [0]
        for level in range(1, cls._DEPTH + 1):
            shift = cls._MIN_SHIFT + 3 * (cls._DEPTH - level)
            first = ((1 << (3 * level)) - 1) // 7 + (start >> shift)
            last = ((1 << (3 * level)) - 1) // 7 + (end >> shift)
            bins.extend(range(first, last + 1))
        return bins

    def _merged_chunks(self, contig: str, start: int, end: int) -> list[tuple[int, int]]:
        ref_index = self._contig_to_index.get(contig)
        if ref_index is None:
            return []
        reference = self._references[ref_index]
        intervals = reference["intervals"]
        linear_index = start >> self._MIN_SHIFT
        minimum_offset = intervals[linear_index] if linear_index < len(intervals) else 0
        bins = reference["bins"]
        chunks = [
            chunk
            for bin_id in self._region_bins(start, end)
            for chunk in bins.get(bin_id, [])
            if chunk[1] > minimum_offset
        ]
        if not chunks:
            return []
        chunks.sort()

        merged = [chunks[0]]
        for chunk_start, chunk_end in chunks[1:]:
            prev_start, prev_end = merged[-1]
            if chunk_start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, chunk_end))
            else:
                merged.append((chunk_start, chunk_end))
        return merged

    def _handle(self):
        if self._vcf_handle is None:
            self._vcf_handle = self.vcf_path.open("rb")
        return self._vcf_handle

    @staticmethod
    def _bgzf_block_size(handle, block_offset: int) -> int:
        handle.seek(block_offset)
        fixed_header = handle.read(12)
        if len(fixed_header) != 12 or fixed_header[:3] != b"\x1f\x8b\x08":
            raise ValueError(f"BGZF block không hợp lệ tại byte offset {block_offset}.")
        extra_length = struct.unpack_from("<H", fixed_header, 10)[0]
        extra = handle.read(extra_length)
        cursor = 0
        while cursor + 4 <= len(extra):
            subfield_id = extra[cursor : cursor + 2]
            subfield_length = struct.unpack_from("<H", extra, cursor + 2)[0]
            cursor += 4
            subfield = extra[cursor : cursor + subfield_length]
            cursor += subfield_length
            if subfield_id == b"BC" and len(subfield) == 2:
                return struct.unpack("<H", subfield)[0] + 1
        raise ValueError(f"Không tìm thấy BGZF BC subfield tại byte offset {block_offset}.")

    def _read_virtual_chunk(self, chunk_start: int, chunk_end: int) -> bytes:
        if chunk_end <= chunk_start:
            return b""
        handle = self._handle()
        block_offset = chunk_start >> 16
        end_block_offset = chunk_end >> 16
        start_in_block = chunk_start & 0xFFFF
        end_in_block = chunk_end & 0xFFFF
        pieces = []

        while block_offset <= end_block_offset:
            size = self._bgzf_block_size(handle, block_offset)
            handle.seek(block_offset)
            compressed = handle.read(size)
            decompressed = gzip.decompress(compressed)
            lower = start_in_block if block_offset == (chunk_start >> 16) else 0
            upper = end_in_block if block_offset == end_block_offset else len(decompressed)
            if upper > lower:
                pieces.append(decompressed[lower:upper])
            block_offset += size
        return b"".join(pieces)

    def fetch(self, contig: str, start: int, end: int) -> Iterable[str]:
        """Yield VCF records in a zero-based, half-open genomic interval."""
        for chunk_start, chunk_end in self._merged_chunks(contig, start, end):
            for line in self._read_virtual_chunk(chunk_start, chunk_end).splitlines():
                if line and not line.startswith(b"#"):
                    yield line.decode("utf-8")

    def close(self) -> None:
        if self._vcf_handle is not None:
            self._vcf_handle.close()
            self._vcf_handle = None


@dataclass(frozen=True)
class SpliceAILookupReport:
    total_rows: int
    eligible_snv_rows: int
    unique_eligible_snvs: int
    matched_unique_snvs: int
    matched_rows: int
    unavailable_rows: int

    def as_dict(self) -> dict[str, int | float]:
        coverage = self.matched_rows / self.eligible_snv_rows if self.eligible_snv_rows else 0.0
        return {
            "total_rows": self.total_rows,
            "eligible_snv_rows": self.eligible_snv_rows,
            "unique_eligible_snvs": self.unique_eligible_snvs,
            "matched_unique_snvs": self.matched_unique_snvs,
            "matched_rows": self.matched_rows,
            "unavailable_rows": self.unavailable_rows,
            "eligible_row_coverage": round(coverage, 6),
        }


class IndexedSpliceAILookup:
    """Map a DataFrame of GRCh38 SNVs to a tabix-indexed SpliceAI VCF."""

    def __init__(self, vcf_path: str | Path, window_size: int = 1_000_000):
        self.vcf_path = Path(vcf_path)
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError("window_size phải là số nguyên dương.")
        if not self.vcf_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy SpliceAI VCF: {self.vcf_path}")
        if not Path(f"{self.vcf_path}.tbi").is_file():
            raise FileNotFoundError(f"Không tìm thấy tabix index: {self.vcf_path}.tbi")

    @staticmethod
    def _validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
        required = {"CHROM", "POS", "REF", "ALT"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise KeyError(f"Parquet thiếu cột bắt buộc để map SpliceAI: {missing}")

        prepared = pd.DataFrame({"_row": np.arange(len(df), dtype=np.int64)})
        prepared["_chrom"] = df["CHROM"].map(canonical_chromosome)
        prepared["_pos"] = pd.to_numeric(df["POS"], errors="coerce")
        prepared["_ref"] = df["REF"].astype("string").str.strip().str.upper()
        prepared["_alt"] = df["ALT"].astype("string").str.strip().str.upper()

        prepared["_valid_snv"] = (
            prepared["_chrom"].notna()
            & prepared["_pos"].notna()
            & (prepared["_pos"] > 0)
            & (prepared["_pos"] % 1 == 0)
            & prepared["_ref"].isin(_BASES)
            & prepared["_alt"].isin(_BASES)
        )
        prepared = prepared.loc[prepared["_valid_snv"], ["_row", "_chrom", "_pos", "_ref", "_alt"]].copy()
        prepared["_pos"] = prepared["_pos"].astype(np.int64)
        return prepared

    def _lookup_unique_variants(self, unique_variants: pd.DataFrame) -> pd.DataFrame:
        matched: dict[tuple[str, int, str, str], np.ndarray] = {}

        tabix = NativeTabixReader(self.vcf_path)
        try:
            contig_aliases = _vcf_contig_aliases(tabix.contigs)
            query = unique_variants.copy()
            query["_vcf_contig"] = query["_chrom"].map(contig_aliases)
            query = query.dropna(subset=["_vcf_contig"])
            query["_window"] = (query["_pos"] - 1) // self.window_size

            for (contig, window), group in query.groupby(["_vcf_contig", "_window"], sort=False):
                requested = {
                    (chrom, int(pos), ref, alt)
                    for chrom, pos, ref, alt in group[["_chrom", "_pos", "_ref", "_alt"]].itertuples(
                        index=False, name=None
                    )
                }
                start = int(window) * self.window_size
                end = start + self.window_size

                for line in tabix.fetch(contig, start, end):
                    fields = line.split("\t")
                    if len(fields) < 8:
                        continue
                    chrom = canonical_chromosome(fields[0])
                    try:
                        pos = int(fields[1])
                    except ValueError:
                        continue
                    ref = fields[3].upper()
                    for alt in fields[4].upper().split(","):
                        key = (chrom, pos, ref, alt)
                        if key not in requested:
                            continue
                        values = parse_spliceai_info(fields[7], alt)
                        if values is None:
                            continue
                        current = matched.get(key)
                        if current is None or np.nanmax(values[:4]) > np.nanmax(current[:4]):
                            matched[key] = values
        finally:
            tabix.close()

        rows = []
        for (chrom, pos, ref, alt), values in matched.items():
            rows.append(
                {
                    "_chrom": chrom,
                    "_pos": pos,
                    "_ref": ref,
                    "_alt": alt,
                    **dict(zip(_SCORE_COLUMNS + _POSITION_COLUMNS, values, strict=True)),
                    "SpliceAI_pred_DS_max": float(np.nanmax(values[:4])),
                }
            )
        return pd.DataFrame(rows, columns=["_chrom", "_pos", "_ref", "_alt", *SPLICEAI_FEATURE_COLUMNS])

    def enrich(self, df: pd.DataFrame) -> tuple[pd.DataFrame, SpliceAILookupReport]:
        prepared = self._validate_and_prepare(df)
        unique_variants = prepared.drop_duplicates(["_chrom", "_pos", "_ref", "_alt"])
        matches = self._lookup_unique_variants(unique_variants)

        out = df.copy()
        out = out.drop(columns=SPLICEAI_FEATURE_COLUMNS, errors="ignore")
        for column in SPLICEAI_FEATURE_COLUMNS:
            out[column] = np.nan

        if not prepared.empty and not matches.empty:
            mapped = prepared.merge(matches, on=["_chrom", "_pos", "_ref", "_alt"], how="left", sort=False)
            mapped_values = mapped[SPLICEAI_FEATURE_COLUMNS].to_numpy(dtype=float)
            out.loc[mapped["_row"].to_numpy(), SPLICEAI_FEATURE_COLUMNS] = mapped_values

        matched_rows = int(out["SpliceAI_pred_DS_max"].notna().sum())
        report = SpliceAILookupReport(
            total_rows=len(df),
            eligible_snv_rows=len(prepared),
            unique_eligible_snvs=len(unique_variants),
            matched_unique_snvs=len(matches),
            matched_rows=matched_rows,
            unavailable_rows=len(df) - matched_rows,
        )
        return out, report
