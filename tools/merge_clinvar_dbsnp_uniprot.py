"""Append UniProt VEP+SpliceAI variants to the ClinVar/dbSNP dataset.

This is a row-wise dataset merge keyed and deduplicated by Variant_ID. When an
ID exists in both files, the ClinVar/dbSNP row is retained. Columns present in
only one input are preserved, with null values for the other source.

Example:
    python tools/merge_clinvar_dbsnp_uniprot.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


DEFAULT_CLINVAR = Path(r"D:\variant_data\train1_full_seq_final_spliceai.parquet")
DEFAULT_UNIPROT = Path(r"D:\variant_data\uniprot_all_after_vep_final_spliceai_full.parquet")
DEFAULT_OUTPUT = Path(r"D:\variant_data\train1_uniprot_full_seq_final_spliceai.parquet")
SPLICEAI_COLUMNS = (
    "SpliceAI_pred_DS_AG",
    "SpliceAI_pred_DS_AL",
    "SpliceAI_pred_DS_DG",
    "SpliceAI_pred_DS_DL",
    "SpliceAI_pred_DP_AG",
    "SpliceAI_pred_DP_AL",
    "SpliceAI_pred_DP_DG",
    "SpliceAI_pred_DP_DL",
    "SpliceAI_pred_DS_max",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinvar", type=Path, default=DEFAULT_CLINVAR)
    parser.add_argument("--uniprot", type=Path, default=DEFAULT_UNIPROT,
                        help="UniProt after VEP and SpliceAI")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for path in (args.clinvar, args.uniprot):
        if not path.is_file():
            raise FileNotFoundError(f"Input Parquet not found: {path}")

    clinvar = pd.read_parquet(args.clinvar)
    uniprot = pd.read_parquet(args.uniprot)
    for name, df in (("ClinVar/dbSNP", clinvar), ("UniProt", uniprot)):
        if "Variant_ID" not in df.columns:
            raise ValueError(f"{name} input has no Variant_ID column")
        if df["Variant_ID"].isna().any():
            raise ValueError(f"{name} input has missing Variant_ID values")
        duplicates = df["Variant_ID"].duplicated().sum()
        if duplicates:
            raise ValueError(f"{name} input has {duplicates:,} duplicate Variant_ID rows")

    present_spliceai = [column for column in SPLICEAI_COLUMNS if column in uniprot.columns]
    if not present_spliceai:
        raise ValueError("UniProt input has no SpliceAI score columns; run enrich_spliceai_scores.py first")

    # Keep the existing source tags and identify UniProt rows in the combined file.
    if "Source" in clinvar.columns or "Source" in uniprot.columns:
        if "Source" not in clinvar.columns:
            clinvar["Source"] = "ClinVar/dbSNP"
        if "Source" not in uniprot.columns:
            uniprot["Source"] = "UniProt"
        else:
            uniprot["Source"] = uniprot["Source"].fillna("UniProt")

    overlap = set(clinvar["Variant_ID"].astype(str)) & set(uniprot["Variant_ID"].astype(str))
    if overlap:
        print(f"Overlapping Variant_IDs: {len(overlap):,}; keeping ClinVar/dbSNP rows for these IDs")
        uniprot = uniprot.loc[~uniprot["Variant_ID"].astype(str).isin(overlap)].copy()

    combined = pd.concat([clinvar, uniprot], ignore_index=True, sort=False)
    if "CHROM" in combined.columns and "POS" in combined.columns:
        chrom = combined["CHROM"].astype("string").str.upper().str.replace(r"^CHR", "", regex=True)
        chrom = chrom.replace({"M": "MT"})
        chrom_number = pd.to_numeric(chrom, errors="coerce")
        special_order = chrom.map({"X": 23, "Y": 24, "MT": 25})
        combined["_chrom_sort"] = chrom_number.fillna(special_order).fillna(1000)
        combined["_chrom_text"] = chrom.where(combined["_chrom_sort"] >= 1000, "")
        combined["_pos_sort"] = pd.to_numeric(combined["POS"], errors="coerce")
        combined = combined.sort_values(
            ["_chrom_sort", "_chrom_text", "_pos_sort", "Variant_ID"], kind="stable", na_position="last"
        ).drop(columns=["_chrom_sort", "_chrom_text", "_pos_sort"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        combined.to_parquet(temporary, index=False)
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"ClinVar/dbSNP input rows: {len(clinvar):,}")
    print(f"Unique UniProt rows added: {len(uniprot):,}")
    print(f"Combined rows: {len(combined):,}")
    print(f"SpliceAI columns in UniProt input: {', '.join(present_spliceai)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
