"""Fill SOTA prediction columns in the merged test set from UniProt by Variant_ID.

Missing SOTA cells are filled from the UniProt reference. Rows that still have
any missing value among the requested SOTA columns are removed. The target is
updated atomically after a one-time backup is made beside it.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd


SOURCE = Path(r"D:\variant_data\uniprot_full_seq_after_vep_final_spliceai.parquet")
TARGET = Path(r"D:\variant_data\test_full_seq_after_vep_final_spliceai_with_uniprot.parquet")
SOTA_COLUMNS = [
    "SIFT_score", "SIFT_converted_rankscore", "SIFT_pred",
    "SIFT4G_score", "SIFT4G_converted_rankscore", "SIFT4G_pred",
    "Polyphen2_HDIV_score", "Polyphen2_HDIV_rankscore", "Polyphen2_HDIV_pred",
    "Polyphen2_HVAR_score", "Polyphen2_HVAR_rankscore", "Polyphen2_HVAR_pred",
    "MutationTaster_score", "MutationTaster_rankscore", "MutationTaster_pred",
    "MetaSVM_score", "MetaSVM_rankscore", "MetaSVM_pred",
    "MetaLR_score", "MetaLR_rankscore", "MetaLR_pred",
    "MetaRNN_score", "MetaRNN_rankscore", "MetaRNN_pred",
    "M-CAP_score", "M-CAP_rankscore", "M-CAP_pred",
    "REVEL_score", "REVEL_rankscore", "REVEL_pred",
    "MutPred2_score", "MutPred2_rankscore", "MutPred2_pred",
    "MVP_score", "MVP_rankscore", "MVP_pred",
    "gMVP_score", "gMVP_rankscore", "gMVP_pred",
    "MisFit_D_score", "MisFit_D_rankscore", "MisFit_D_pred_lenient",
    "MPC_score", "MPC_rankscore", "MPC_pred",
    "PrimateAI_score", "PrimateAI_rankscore", "PrimateAI_pred",
    "BayesDel_addAF_score", "BayesDel_addAF_rankscore", "BayesDel_addAF_pred",
    "BayesDel_noAF_score", "BayesDel_noAF_rankscore", "BayesDel_noAF_pred",
    "ClinPred_score", "ClinPred_rankscore", "ClinPred_pred",
    "LIST-S2_score", "LIST-S2_rankscore", "LIST-S2_pred",
    "VARITY_R_score", "VARITY_R_rankscore", "VARITY_R_pred",
    "VARITY_ER_score", "VARITY_ER_rankscore", "VARITY_ER_pred",
    "AlphaMissense_score", "AlphaMissense_rankscore", "AlphaMissense_pred",
    "PHACTboost_score", "PHACTboost_rankscore", "PHACTboost_pred",
    "MutFormer_score", "MutFormer_rankscore", "MutFormer_pred",
    "popEVE_score", "popEVE_converted_rankscore", "popEVE_pred",
    "CADD_raw", "CADD_raw_rankscore", "CADD_phred", "CADD_pred",
    "DANN_score", "DANN_rankscore", "DANN_pred",
]


def main() -> None:
    if not SOURCE.is_file() or not TARGET.is_file():
        raise FileNotFoundError(f"Input not found: source={SOURCE.is_file()}, target={TARGET.is_file()}")

    source_columns = ["Variant_ID", *SOTA_COLUMNS]
    source = pd.read_parquet(SOURCE, columns=source_columns)
    target = pd.read_parquet(TARGET)
    if target["Variant_ID"].isna().any() or source["Variant_ID"].isna().any():
        raise ValueError("Variant_ID contains missing values")
    if source["Variant_ID"].duplicated().any():
        raise ValueError("Source UniProt contains duplicate Variant_ID values")
    missing_in_target = sorted(set(SOTA_COLUMNS) - set(target.columns))
    if missing_in_target:
        raise ValueError(f"Target is missing SOTA columns: {missing_in_target}")

    source_ids = source["Variant_ID"].astype("string")
    if source[SOTA_COLUMNS].isna().any().any():
        missing_counts = source[SOTA_COLUMNS].isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Source has missing SOTA values, cannot guarantee completeness: {missing_counts}")
    source_indexed = source.assign(Variant_ID=source_ids).set_index("Variant_ID")
    target_ids = target["Variant_ID"].astype("string")

    before_missing = target[SOTA_COLUMNS].isna()
    missing_cells_before = int(before_missing.to_numpy().sum())
    target_rows_before = len(target)
    matched_mask = target_ids.isin(source_indexed.index)
    matched_rows = int(matched_mask.sum())
    filled_cells = 0
    for column in SOTA_COLUMNS:
        mapped = target_ids.map(source_indexed[column])
        fill_mask = target[column].isna() & mapped.notna()
        filled_cells += int(fill_mask.sum())
        target[column] = target[column].where(~fill_mask, mapped)

    incomplete = target[SOTA_COLUMNS].isna().any(axis=1)
    removed_rows = int(incomplete.sum())
    target = target.loc[~incomplete].copy().reset_index(drop=True)
    if target[SOTA_COLUMNS].isna().any().any():
        raise RuntimeError("Postcondition failed: remaining rows still contain missing SOTA values")
    if target["Variant_ID"].duplicated().any():
        raise ValueError("Target contains duplicate Variant_ID values")

    backup = TARGET.with_name(f"{TARGET.stem}_before_sota_mapping{TARGET.suffix}")
    if not backup.exists():
        shutil.copy2(TARGET, backup)
    temporary = TARGET.with_suffix(TARGET.suffix + ".tmp")
    try:
        target.to_parquet(temporary, index=False)
        os.replace(temporary, TARGET)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"SOTA columns checked: {len(SOTA_COLUMNS)}")
    print(f"Rows before: {target_rows_before:,}")
    print(f"Rows matched to UniProt source: {matched_rows:,}")
    print(f"Missing SOTA cells before mapping: {missing_cells_before:,}")
    print(f"SOTA cells filled: {filled_cells:,}")
    print(f"Rows removed for remaining missing SOTA: {removed_rows:,}")
    print(f"Rows remaining: {len(target):,}")
    print(f"Backup: {backup}")
    print(f"Updated target: {TARGET}")


if __name__ == "__main__":
    main()
