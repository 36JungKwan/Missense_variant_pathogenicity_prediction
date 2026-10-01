# Geometry Feature Selection By Validation - Batch Run 4

Validation metrics were backfilled from the saved `Pure_XGBoost_Concat` checkpoints and added to `global_leaderboard_metrics.csv`.

- Rows updated: 4092
- Geometry subsets: 1023
- Validation split: `val` from `train3__val`
- Ranking metric: `Val_MCC`

## Top 10 By Validation MCC

| Rank | Geometry features | Val MCC | Val AUROC | Val AUPRC | Mean test MCC |
|---:|---|---:|---:|---:|---:|
| 1 | `dna_LVD_L2+dna_LVD_Relative+prot_LLR+prot_LVD_Cosine+prot_LVD_Relative` | 0.7911 | 0.9724 | 0.9103 | 0.6769 |
| 2 | `dna_LVD_L2+dna_LID+dna_LVD_Relative+prot_LLR+prot_LVD_Cosine+prot_LVD_Relative` | 0.7910 | 0.9722 | 0.9092 | 0.6735 |
| 3 | `dna_LLR+dna_LVD_L2+prot_LLR+prot_LVD_L2+prot_LVD_Cosine+prot_LID` | 0.7907 | 0.9727 | 0.9119 | 0.6770 |
| 4 | `dna_LVD_Cosine+dna_LVD_Relative+prot_LLR+prot_LVD_L2+prot_LID+prot_LVD_Relative` | 0.7907 | 0.9725 | 0.9107 | 0.6719 |
| 5 | `dna_LVD_Relative+prot_LLR+prot_LVD_L2+prot_LID+prot_LVD_Relative` | 0.7906 | 0.9726 | 0.9094 | 0.6757 |
| 6 | `dna_LLR+dna_LVD_L2+prot_LLR+prot_LVD_L2+prot_LVD_Relative` | 0.7902 | 0.9722 | 0.9103 | 0.6806 |
| 7 | `dna_LVD_Cosine+dna_LID+dna_LVD_Relative+prot_LLR+prot_LID+prot_LVD_Relative` | 0.7901 | 0.9727 | 0.9095 | 0.6858 |
| 8 | `dna_LID+prot_LLR+prot_LVD_L2+prot_LVD_Relative` | 0.7901 | 0.9723 | 0.9104 | 0.6800 |
| 9 | `dna_LVD_L2+dna_LID+dna_LVD_Relative+prot_LLR+prot_LID+prot_LVD_Relative` | 0.7901 | 0.9723 | 0.9095 | 0.6696 |
| 10 | `dna_LVD_L2+dna_LVD_Cosine+dna_LID+prot_LLR+prot_LVD_L2+prot_LVD_Cosine` | 0.7898 | 0.9723 | 0.9082 | 0.6658 |

## Selection Guidance

Use rank 1 as the validation-selected candidate for the next fixed-geometry benchmark. Rank 7 is a useful stability candidate because it has slightly lower validation MCC but the strongest mean test MCC among this validation top 10.

The test columns remain available for final reporting only. They should not be used to select the final geometry configuration.
