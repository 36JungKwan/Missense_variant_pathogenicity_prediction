"""Đọc danh sách biến thể từ file upload của Batch Screening.

Định dạng hỗ trợ:
  * .vcf / .txt : dòng VCF (#CHROM POS ID REF ALT ...) hoặc mỗi dòng một mã chrN:POS:REF>ALT
  * .csv        : cột CHROM/POS/REF/ALT, hoặc ô chứa mã chrN:POS:REF>ALT
  * .parquet / .xlsx : bảng có cột CHROM, POS, REF, ALT (không phân biệt hoa thường; chấp nhận các tên
                       thường gặp như chr/chromosome, position/start), hoặc một cột chứa mã chrN:POS:REF>ALT
Kết quả đã chuẩn hoá và khử trùng lặp, giữ nguyên thứ tự xuất hiện.
"""
from __future__ import annotations

import csv
import io
import math
from typing import Dict, Iterable, List, Optional

import pandas as pd

from app.core.constants import VARIANT_ID_PATTERN, parse_variant_id
from app.core.errors import InvalidUploadError

SUPPORTED_EXTENSIONS = (".vcf", ".txt", ".csv", ".parquet", ".xlsx")
_COLUMN_ALIASES: Dict[str, tuple] = {
    "chrom": ("chrom", "#chrom", "chr", "chromosome", "contig", "seqname"),
    "pos": ("pos", "position", "start", "pos_hg38", "pos_grch38"),
    "ref": ("ref", "ref_allele", "reference"),
    "alt": ("alt", "alt_allele", "alternate"),
}


class _Collector:
    def __init__(self) -> None:
        self.seen: set = set()
        self.ids: List[str] = []

    def add(self, candidate: str) -> None:
        candidate = candidate.strip()
        if not VARIANT_ID_PATTERN.match(candidate):
            return
        norm = parse_variant_id(candidate).variant_id
        if norm not in self.seen:
            self.seen.add(norm)
            self.ids.append(norm)

    def add_fields(self, chrom: object, pos: object, ref: object, alt: object) -> None:
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (chrom, pos, ref, alt)):
            return
        try:
            pos_int = int(float(pos))
        except (TypeError, ValueError):
            return
        chrom_s = str(chrom).strip()
        if chrom_s.endswith(".0"):  # CHROM số bị pandas đọc thành float (7.0)
            chrom_s = chrom_s[:-2]
        alt_s = str(alt).split(",")[0].strip().upper()
        if alt_s and alt_s != ".":
            self.add(f"{chrom_s}:{pos_int}:{str(ref).strip().upper()}>{alt_s}")


def _find_columns(columns: Iterable[object]) -> Optional[Dict[str, str]]:
    by_lower = {str(c).strip().lower(): c for c in columns}
    found = {}
    for key, aliases in _COLUMN_ALIASES.items():
        match = next((by_lower[a] for a in aliases if a in by_lower), None)
        if match is None:
            return None
        found[key] = match
    return found


def _from_frame(frame: pd.DataFrame, collector: _Collector) -> None:
    columns = _find_columns(frame.columns)
    if columns is not None:
        subset = frame[[columns["chrom"], columns["pos"], columns["ref"], columns["alt"]]]
        for chrom, pos, ref, alt in subset.itertuples(index=False, name=None):
            collector.add_fields(chrom, pos, ref, alt)
        return
    # Không có 4 cột toạ độ: tìm cột chứa mã biến thể (vd Variant_ID = chr7:140753336:A>T).
    for column in frame.columns:
        values = frame[column].dropna().astype(str)
        if len(values) and values.head(20).map(lambda v: bool(VARIANT_ID_PATTERN.match(v.strip()))).any():
            for value in values:
                collector.add(value)
            return
    raise InvalidUploadError(
        "Table has no CHROM/POS/REF/ALT columns (case-insensitive) and no column of chrN:POS:REF>ALT ids. "
        f"Columns found: {', '.join(map(str, list(frame.columns)[:20]))}"
    )


def _read_table(raw_bytes: bytes, lower_name: str) -> pd.DataFrame:
    buffer = io.BytesIO(raw_bytes)
    try:
        if lower_name.endswith(".parquet"):
            return pd.read_parquet(buffer)
        return pd.read_excel(buffer, sheet_name=0, engine="openpyxl")
    except ImportError as exc:
        raise InvalidUploadError(f"Server is missing the reader for this format: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - file hỏng / sai định dạng
        raise InvalidUploadError(f"Cannot read '{lower_name}': {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}") from exc


def _from_csv(text: str, collector: _Collector) -> None:
    rows = [row for row in csv.reader(io.StringIO(text)) if row]
    if rows and _find_columns(rows[0]) is not None:
        _from_frame(pd.DataFrame(rows[1:], columns=[c.strip() for c in rows[0]]), collector)
        return
    for row in rows:
        cell = next((c.strip() for c in row if VARIANT_ID_PATTERN.match(c.strip())), None)
        if cell is not None:
            collector.add(cell)
        elif len(row) >= 4 and row[1].strip().isdigit():
            collector.add_fields(row[0], row[1], row[2], row[3])


def _from_vcf_or_text(text: str, collector: _Collector) -> None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cols = stripped.split("\t") if "\t" in stripped else stripped.split()
        if len(cols) >= 5 and cols[1].isdigit():
            collector.add_fields(cols[0], cols[1], cols[3], cols[4])
        else:
            collector.add(stripped)


def parse_uploaded_variants(raw_bytes: bytes, filename: str = "") -> List[str]:
    lower_name = (filename or "").lower()
    if lower_name and not lower_name.endswith(SUPPORTED_EXTENSIONS):
        raise InvalidUploadError(
            f"Unsupported file type '{filename}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    collector = _Collector()
    if lower_name.endswith((".parquet", ".xlsx")):
        _from_frame(_read_table(raw_bytes, lower_name), collector)
    elif lower_name.endswith(".csv"):
        _from_csv(raw_bytes.decode("utf-8", errors="ignore"), collector)
    else:
        _from_vcf_or_text(raw_bytes.decode("utf-8", errors="ignore"), collector)
    return collector.ids
