"""Enrich all current fusion inputs with pre-computed SpliceAI scores.

Run this with the project's regular Python environment:
    python tools/enrich_spliceai_scores.py

The command writes sibling files with ``_spliceai.parquet`` in their names and
never overwrites source datasets unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.module01_data_processing_vep.spliceai_lookup import IndexedSpliceAILookup


DEFAULT_VCF = Path(r"D:\vep_resources\spliceai_scores.raw.snv.ensembl_mane_v1.4.grch38.vcf.gz")
DEFAULT_INPUTS = [
    Path(r"D:\variant_data\train1_full_seq_final.parquet"),
    Path(r"D:\variant_data\train2_full_seq_final.parquet"),
    Path(r"D:\variant_data\train3_full_seq_final.parquet"),
    Path(r"D:\variant_data\val_full_seq_final.parquet"),
    Path(r"D:\variant_data\test_full_seq_after_vep_final.parquet"),
    Path(r"D:\variant_data\clinvarhq_full_seq_after_vep_final.parquet"),
    Path(r"D:\variant_data\uniprot_full_seq_after_vep_final.parquet"),
    Path(r"D:\variant_data\proteingym_full_seq_after_vep_final.parquet"),
]


def spliceai_output_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_spliceai{source.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map indexed SpliceAI GRCh38 scores to fusion Parquets.")
    parser.add_argument("--vcf", type=Path, default=DEFAULT_VCF, help="SpliceAI VCF.gz with adjacent .tbi index.")
    parser.add_argument("--input", type=Path, action="append", help="Input Parquet; repeat to override default inputs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing _spliceai.parquet outputs.")
    parser.add_argument("--window-size", type=int, default=1_000_000, help="Tabix query window in bp.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(r"D:\variant_data\spliceai_mapping_report.json"),
        help="Coverage report written after all files complete.",
    )
    return parser.parse_args()


def write_parquet_atomically(df: pd.DataFrame, target: Path) -> None:
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    try:
        df.to_parquet(temporary, index=False)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    inputs = args.input or DEFAULT_INPUTS
    lookup = IndexedSpliceAILookup(args.vcf, window_size=args.window_size)
    summary = {
        "spliceai_vcf": str(args.vcf),
        "genome_build": "GRCh38",
        "aggregation": "annotation with largest DS_max; scores and positions retained from that annotation",
        "datasets": {},
    }

    for source in inputs:
        target = spliceai_output_path(source)
        if not source.is_file():
            raise FileNotFoundError(f"Không tìm thấy input Parquet: {source}")
        if target.exists() and not args.overwrite:
            print(f"[SKIP] Đã có output: {target}")
            continue

        print(f"[*] Đang map SpliceAI: {source.name}")
        df = pd.read_parquet(source)
        enriched, report = lookup.enrich(df)
        write_parquet_atomically(enriched, target)
        dataset_report = report.as_dict() | {"input": str(source), "output": str(target)}
        summary["datasets"][source.name] = dataset_report
        print(
            "    -> match "
            f"{dataset_report['matched_rows']:,}/{dataset_report['eligible_snv_rows']:,} "
            f"({dataset_report['eligible_row_coverage']:.2%})"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[+] Đã lưu coverage report: {args.report}")


if __name__ == "__main__":
    main()
