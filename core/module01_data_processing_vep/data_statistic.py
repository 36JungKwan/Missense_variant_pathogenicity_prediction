import pandas as pd

DATASETS = {
    "train1": r"D:\variant_data\train1_full_seq_final_spliceai.parquet",
    "train2": r"D:\variant_data\train2_full_seq_final_spliceai.parquet",
    "train3": r"D:\variant_data\train3_full_seq_final_spliceai.parquet",
    "val": r"D:\variant_data\val_full_seq_final_spliceai.parquet",
    "clinvarhq": r"D:\variant_data\clinvarhq_full_seq_after_vep_final_spliceai.parquet",
    "proteingym": r"D:\variant_data\proteingym_full_seq_after_vep_final_spliceai.parquet",
    "test": r"D:\variant_data\test_full_seq_after_vep_final_spliceai.parquet",
}


def find_column(df, candidates, purpose):
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(
        f"Không tìm thấy cột {purpose}. Các cột hiện có: {df.columns.tolist()}"
    )


def count_label(labels, kind):
    labels = (
        labels.astype("string")
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
    )

    if kind == "pathogenic":
        mask = labels.isin(
            {"1", "1.0", "pathogenic", "likely pathogenic", "patho"}
        )
        mask |= labels.str.startswith("pathogenic/", na=False)
    else:
        mask = labels.isin({"0", "0.0", "benign", "likely benign"})
        mask |= labels.str.startswith("benign/", na=False)

    return int(mask.sum())


def summarize_datasets():
    summary_rows = []

    for dataset_name, file_path in DATASETS.items():
        df = pd.read_parquet(file_path)

        label_col = find_column(df, ("Label", "label"), "nhãn")
        gene_col = find_column(
            df, ("SYMBOL", "GeneInfo", "gene", "gene_info"), "gene"
        )
        chromosome_col = find_column(
            df, ("CHROM", "chromosome", "chrom"), "chromosome"
        )

        variants = len(df)
        pathogenic = count_label(df[label_col], "pathogenic")
        benign = count_label(df[label_col], "benign")

        summary_rows.append({
            "dataset": dataset_name,
            "variants": variants,
            "pathogenic": pathogenic,
            "benign": benign,
            "%pathogenic": (
                round(pathogenic / variants * 100, 2) if variants else 0.0
            ),
            "genes": df[gene_col].nunique(dropna=True),
            "chromosomes": df[chromosome_col].nunique(dropna=True),
        })

    return pd.DataFrame(
        summary_rows,
        columns=[
            "dataset", "variants", "pathogenic", "benign",
            "%pathogenic", "genes", "chromosomes",
        ],
    )


if __name__ == "__main__":
    summary_df = summarize_datasets()
    print(summary_df.to_string(index=False))