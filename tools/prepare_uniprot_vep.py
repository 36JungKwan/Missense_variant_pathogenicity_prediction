r"""Normalize UniProt variants and write chromosome-sorted VEP inputs.

The script writes:
  * uniprot_vep_input.parquet: normalized UniProt rows absent from the merge target
  * uniprot_vep_input.vcf: VCF 4.2 input containing those same novel rows
  * uniprot_variants_already_in_clinvar_dbsnp.parquet: excluded overlap for audit

Examples (PowerShell):
  python tools/prepare_uniprot_vep.py
  python tools/prepare_uniprot_vep.py --input D:\variant_data\uniprot.parquet `
      --output-dir D:\variant_data\uniprot_vep
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


DEFAULT_INPUT = Path(r"D:\variant_data\uniprot.parquet")
DEFAULT_MERGED = Path(r"D:\variant_data\clinvar_dbsnp_vep_merged_final.parquet")


def chromosome_key(chrom: str) -> tuple[int, str]:
    """Natural chromosome order: 1-22, X, Y, MT, then other contigs."""
    value = chrom.upper()
    if value.isdigit():
        number = int(value)
        if 1 <= number <= 22:
            return number, ""
        return 100 + number, ""
    special = {"X": 23, "Y": 24, "MT": 25}
    if value in special:
        return special[value], ""
    return 1000, value


def normalize_label(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Label contains a missing value")
    text = str(value).strip().lower()
    mapping = {"benign": 0, "0": 0, "0.0": 0, "patho": 1, "pathogenic": 1, "1": 1, "1.0": 1}
    if text not in mapping:
        raise ValueError(f"Unsupported Label value: {value!r}; expected benign/patho or 0/1")
    return mapping[text]


def find_column(df: pd.DataFrame, candidates: tuple[str, ...], required: bool = True) -> str | None:
    by_lower = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    if required:
        raise ValueError(f"Missing required column. Expected one of: {', '.join(candidates)}")
    return None


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "CHROM": find_column(df, ("CHROM", "chrom")),
        "POS": find_column(df, ("POS", "pos")),
        "REF": find_column(df, ("REF", "ref")),
        "ALT": find_column(df, ("ALT", "alt")),
        "Label": find_column(df, ("Label", "label")),
    }
    result = df.copy()
    # Drop source versions of normalized fields before adding canonical fields,
    # avoiding duplicate columns when the input is already partly normalized.
    source_names = list(dict.fromkeys(columns.values()))
    result = result.drop(columns=source_names)

    chrom = df[columns["CHROM"]].astype("string").str.strip().str.upper()
    chrom = chrom.str.replace(r"^CHR", "", regex=True).replace({"M": "MT"})
    pos_numeric = pd.to_numeric(df[columns["POS"]], errors="raise")
    if pos_numeric.isna().any() or (pos_numeric % 1 != 0).any() or (pos_numeric < 1).any():
        raise ValueError("POS must contain positive whole-number genomic positions")
    ref = df[columns["REF"]].astype("string").str.strip().str.upper()
    alt = df[columns["ALT"]].astype("string").str.strip().str.upper()
    if chrom.isna().any() or (chrom == "").any() or ref.isna().any() or alt.isna().any():
        raise ValueError("CHROM, REF, and ALT must not be empty")
    if (ref == "").any() or (alt == "").any():
        raise ValueError("REF and ALT must not be empty")

    result["CHROM"] = chrom.astype(str)
    result["POS"] = pos_numeric.astype("int64")
    result["REF"] = ref.astype(str)
    result["ALT"] = alt.astype(str)
    result["Label"] = df[columns["Label"]].map(normalize_label).astype("int8")
    result["Variant_ID"] = (
        result["CHROM"] + "_" + result["POS"].astype(str) + "_" + result["REF"] + "_" + result["ALT"]
    )

    if result["Variant_ID"].duplicated().any():
        duplicates = result.loc[result["Variant_ID"].duplicated(keep=False), "Variant_ID"].head(10).tolist()
        raise ValueError(f"Duplicate Variant_ID values found (examples: {duplicates})")

    result["_chrom_order"] = result["CHROM"].map(chromosome_key)
    result = result.sort_values(
        by=["_chrom_order", "POS", "REF", "ALT"], kind="stable"
    ).drop(columns="_chrom_order").reset_index(drop=True)
    return result


def write_vcf(df: pd.DataFrame, path: Path) -> None:
    """Write VCF 4.2 accepted by VEP, preserving join key and binary label."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##source=uniprot_variant_preparation\n")
        handle.write('##INFO=<ID=LABEL,Number=1,Type=Integer,Description="UniProt class: benign=0, patho=1">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for row in df[["CHROM", "POS", "Variant_ID", "REF", "ALT", "Label"]].itertuples(index=False, name=None):
            chrom, pos, variant_id, ref, alt, label = row
            if any("\t" in str(value) or "\n" in str(value) for value in (chrom, variant_id, ref, alt)):
                raise ValueError(f"VCF field contains a tab/newline for Variant_ID {variant_id}")
            handle.write(f"{chrom}\t{pos}\t{variant_id}\t{ref}\t{alt}\t.\tPASS\tLABEL={label}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input UniProt Parquet")
    parser.add_argument("--merged-parquet", type=Path, default=DEFAULT_MERGED,
                        help="Existing ClinVar/dbSNP/VEP Parquet to inspect and compare Variant_IDs")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: same directory as --input)")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading UniProt: {args.input}")
    source = pd.read_parquet(args.input)
    print(f"  rows={len(source):,}; columns={len(source.columns)}")
    print(f"  columns: {', '.join(map(str, source.columns))}")

    merged_ids: set[str] | None = None
    if args.merged_parquet.is_file():
        print(f"Inspecting merge target: {args.merged_parquet}")
        merged = pd.read_parquet(args.merged_parquet, columns=["Variant_ID"])
        merged_ids = set(merged["Variant_ID"].dropna().astype(str))
        print(f"  rows={len(merged):,}; unique Variant_ID={len(merged_ids):,}")
    else:
        print(f"Merge target not found; skipping ID overlap audit: {args.merged_parquet}")

    prepared = normalize(source)
    excluded_path = out_dir / "uniprot_variants_already_in_clinvar_dbsnp.parquet"
    if merged_ids is not None:
        overlap = prepared["Variant_ID"].isin(merged_ids)
        excluded = prepared.loc[overlap].copy()
        prepared = prepared.loc[~overlap].copy().reset_index(drop=True)
        excluded.to_parquet(excluded_path, index=False)
        print(f"Excluded existing Variant_IDs: {len(excluded):,}")
        print(f"Saved excluded rows: {excluded_path}")

    parquet_path = out_dir / "uniprot_vep_input.parquet"
    vcf_path = out_dir / "uniprot_vep_input.vcf"
    prepared.to_parquet(parquet_path, index=False)
    write_vcf(prepared, vcf_path)

    print(f"Prepared rows: {len(prepared):,}")
    print(f"Label counts: {prepared['Label'].value_counts().sort_index().to_dict()} (0=benign, 1=patho)")
    print(f"Chromosomes: {', '.join(prepared['CHROM'].drop_duplicates().tolist())}")
    if merged_ids is not None:
        print(f"Remaining Variant_ID overlap with merge target: {int(prepared['Variant_ID'].isin(merged_ids).sum()):,}")
    print(f"Wrote Parquet: {parquet_path}")
    print(f"Wrote VCF:     {vcf_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
