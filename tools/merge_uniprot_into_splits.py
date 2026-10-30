"""Add chromosome-assigned UniProt rows to train/validation/test Parquets.

Writes new ``*_with_uniprot.parquet`` files and a JSON summary, leaving input
files unchanged. Train2 only receives rows that preserve two classes per gene.
Train3 receives equal benign/patho additions within each chromosome+gene group,
preserving its existing exact per-gene and per-chromosome class balance.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


DATA_DIR = Path(r"D:\variant_data")
UNIPROT_INPUT = DATA_DIR / "uniprot_all_after_vep_final_spliceai_full.parquet"
SPLITS = {
    "train1": DATA_DIR / "train1_full_seq_final_spliceai.parquet",
    "train2": DATA_DIR / "train2_full_seq_final_spliceai.parquet",
    "train3": DATA_DIR / "train3_full_seq_final_spliceai.parquet",
    "val": DATA_DIR / "val_full_seq_final_spliceai.parquet",
    "test": DATA_DIR / "test_full_seq_after_vep_final_spliceai.parquet",
}
TRAIN_CHROMS = {"1", "2", "X", "17", "7", "3", "12", "5", "11", "9", "16", "6", "14", "8", "13", "21"}
VAL_CHROMS = {"19", "4", "22", "18"}
TEST_CHROMS = {"15", "10", "20", "Y"}


def canonical_chrom(series: pd.Series) -> pd.Series:
    chrom = series.astype("string").str.strip().str.upper().str.replace(r"^CHR", "", regex=True)
    return chrom.replace({"M": "MT"})


def gene_class_summary(df: pd.DataFrame) -> tuple[int, int]:
    grouped = df.groupby("SYMBOL", dropna=True)["Label"].nunique()
    return int(df["SYMBOL"].nunique(dropna=True)), int((grouped < 2).sum())


def validate_frame(df: pd.DataFrame, name: str) -> None:
    required = {"Variant_ID", "CHROM", "Label", "SYMBOL"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    if df["Variant_ID"].isna().any():
        raise ValueError(f"{name} contains missing Variant_ID")
    labels = set(pd.to_numeric(df["Label"], errors="coerce").dropna().unique())
    if not labels.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"{name} contains labels outside 0/1: {sorted(labels)}")
    if df["SYMBOL"].isna().any():
        raise ValueError(f"{name} contains missing SYMBOL values; gene constraints cannot be checked")


def select_train3_rows(source: pd.DataFrame) -> pd.DataFrame:
    """Choose a deterministic maximal paired subset within chromosome+gene."""
    counts = source.groupby(["CHROM_KEY", "SYMBOL", "Label"]).size().unstack(fill_value=0)
    for label in (0, 1):
        if label not in counts.columns:
            counts[label] = 0
    pair_limit = counts[[0, 1]].min(axis=1)
    rank = source.groupby(["CHROM_KEY", "SYMBOL", "Label"], sort=False).cumcount()
    row_limit = pd.MultiIndex.from_frame(source[["CHROM_KEY", "SYMBOL"]]).map(pair_limit.to_dict())
    return source.loc[rank.to_numpy() < row_limit].copy()


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def merged_report(name: str, df: pd.DataFrame, added: pd.DataFrame, output: Path) -> dict[str, object]:
    labels = pd.to_numeric(df["Label"], errors="coerce")
    patho = int((labels == 1).sum())
    benign = int((labels == 0).sum())
    genes, singleton_genes = gene_class_summary(df)
    return {
        "dataset": name,
        "output": str(output),
        "added_uniprot_variants": int(len(added)),
        "total_variants": int(len(df)),
        "patho": patho,
        "benign": benign,
        "patho_percent": round(100 * patho / len(df), 4) if len(df) else 0.0,
        "genes": genes,
        "chromosomes": int(df["CHROM"].nunique(dropna=True)),
        "genes_with_one_class": singleton_genes,
        "class_counts_by_chromosome": {
            str(chrom): {str(label): int(count) for label, count in group.items()}
            for chrom, group in df.groupby("CHROM")["Label"].value_counts().unstack(fill_value=0).iterrows()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniprot", type=Path, default=UNIPROT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR,
                        help="Directory for new merged Parquets and report")
    args = parser.parse_args()
    if not args.uniprot.is_file():
        raise FileNotFoundError(f"UniProt input not found: {args.uniprot}")
    for path in SPLITS.values():
        if not path.is_file():
            raise FileNotFoundError(f"Split input not found: {path}")

    source = pd.read_parquet(args.uniprot)
    validate_frame(source, "UniProt")
    source["CHROM_KEY"] = canonical_chrom(source["CHROM"])
    source["Label"] = pd.to_numeric(source["Label"], errors="raise").astype("int8")
    train_source = source[source["CHROM_KEY"].isin(TRAIN_CHROMS)].copy()
    val_source = source[source["CHROM_KEY"].isin(VAL_CHROMS)].copy()
    test_source = source[source["CHROM_KEY"].isin(TEST_CHROMS)].copy()
    unassigned = source[~source["CHROM_KEY"].isin(TRAIN_CHROMS | VAL_CHROMS | TEST_CHROMS)]
    if len(unassigned):
        raise ValueError(f"{len(unassigned):,} UniProt variants have chromosomes outside the requested split map")

    base_frames: dict[str, pd.DataFrame] = {}
    for name, path in SPLITS.items():
        df = pd.read_parquet(path)
        validate_frame(df, name)
        df["CHROM_KEY"] = canonical_chrom(df["CHROM"])
        base_frames[name] = df

    # Check and remove any unexpected overlap, retaining the existing split row.
    additions: dict[str, pd.DataFrame] = {"train1": train_source.copy(), "val": val_source, "test": test_source}
    train2 = base_frames["train2"]
    train2_gene_classes = train2.groupby("SYMBOL")["Label"].nunique()
    if (train2_gene_classes < 2).any():
        raise ValueError("Train2 already contains genes with only one class")
    source_gene_classes = train_source.groupby("SYMBOL")["Label"].nunique()
    train2_existing_genes = set(train2["SYMBOL"])
    train2_new_two_class_genes = set(source_gene_classes[source_gene_classes == 2].index)
    additions["train2"] = train_source[
        train_source["SYMBOL"].isin(train2_existing_genes | train2_new_two_class_genes)
    ].copy()

    train3 = base_frames["train3"]
    t3_gene = train3.groupby(["SYMBOL", "Label"]).size().unstack(fill_value=0)
    t3_chrom = train3.groupby(["CHROM_KEY", "Label"]).size().unstack(fill_value=0)
    for table, scope in ((t3_gene, "gene"), (t3_chrom, "chromosome")):
        if 0 not in table.columns or 1 not in table.columns or not (table[0] == table[1]).all():
            raise ValueError(f"Train3 is not exactly class-balanced by {scope} before merge")
    additions["train3"] = select_train3_rows(train_source)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, object] = {
        "source": str(args.uniprot),
        "source_rows": int(len(source)),
        "chromosome_assignment": {
            "train": sorted(TRAIN_CHROMS), "val": sorted(VAL_CHROMS), "test": sorted(TEST_CHROMS)
        },
        "train3_rule": "add matched benign/patho pairs within each chromosome+SYMBOL group",
        "train2_rule": "add source rows for existing two-class genes, plus new genes having both classes in UniProt",
        "source_missing_sequence_columns": {},
        "datasets": {},
    }

    for name, base_df in base_frames.items():
        add = additions[name].copy()
        overlap = set(base_df["Variant_ID"].astype(str)) & set(add["Variant_ID"].astype(str))
        if overlap:
            add = add.loc[~add["Variant_ID"].astype(str).isin(overlap)].copy()
        # Annotate provenance where the target schema provides a Source field.
        if "Source" in base_df.columns and "Source" not in add.columns:
            add["Source"] = "UniProt"
        # Carry the gene symbol into the ClinVar-style synonym when possible.
        if "GeneSymbol" in base_df.columns and "GeneSymbol" not in add.columns:
            add["GeneSymbol"] = add["SYMBOL"]
        target_label_dtype = base_df["Label"].dtype
        try:
            add["Label"] = add["Label"].astype(target_label_dtype)
        except (TypeError, ValueError):
            pass
        merged = pd.concat([base_df.drop(columns="CHROM_KEY"), add.drop(columns="CHROM_KEY")], ignore_index=True, sort=False)
        if merged["Variant_ID"].duplicated().any():
            raise ValueError(f"Duplicate Variant_ID remains after merge for {name}")
        if name == "train2" and (merged.groupby("SYMBOL")["Label"].nunique() < 2).any():
            raise ValueError("Train2 merge would produce a gene with one class")
        if name == "train3":
            by_gene = merged.groupby(["SYMBOL", "Label"]).size().unstack(fill_value=0)
            by_chrom = merged.groupby(["CHROM", "Label"]).size().unstack(fill_value=0)
            if not (by_gene[0] == by_gene[1]).all() or not (by_chrom[0] == by_chrom[1]).all():
                raise ValueError("Train3 merge failed per-gene or per-chromosome class balance")

        source_only = sorted(set(add.columns) - set(base_df.columns))
        if name == "train1":
            reports["source_missing_sequence_columns"] = sorted(
                column for column in base_df.columns
                if "seq_" in column and column not in source.columns
            )
        output = args.output_dir / f"{SPLITS[name].stem}_with_uniprot.parquet"
        atomic_parquet(merged, output)
        item = merged_report(name, merged, add, output)
        item["overlap_rows_skipped"] = int(len(overlap))
        item["uniprot_only_columns"] = source_only
        reports["datasets"][name] = item
        print(
            f"{name}: added={len(add):,}, total={item['total_variants']:,}, "
            f"patho={item['patho']:,}, benign={item['benign']:,}, "
            f"patho%={item['patho_percent']:.2f}, genes={item['genes']:,}, "
            f"chromosomes={item['chromosomes']}, one-class genes={item['genes_with_one_class']:,}"
        )

    report_path = args.output_dir / "uniprot_split_merge_report.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    if reports["source_missing_sequence_columns"]:
        print("WARNING: UniProt rows will have nulls in target sequence columns: "
              + ", ".join(reports["source_missing_sequence_columns"]))


if __name__ == "__main__":
    main()
