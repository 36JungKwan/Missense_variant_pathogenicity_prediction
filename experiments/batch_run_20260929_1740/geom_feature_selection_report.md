# Geometry Feature Selection - Batch Run 4

## Scope

- Source: `global_leaderboard_metrics.csv`
- Rows: 4092
- Geometry subsets: 1023
- Test datasets: 4
- Primary metric: mean MCC across the four datasets
- Tie-breakers: median MCC, then minimum MCC

The ranking is exploratory because the same test datasets were used for selecting the candidates. The shortlisted configurations should be re-evaluated on a held-out validation protocol before being treated as the final geometry choice.

## Top 10 By Mean MCC

| Rank | Geometry features | N | Mean MCC | Median MCC | Min MCC | Mean AUROC | Mean AUPRC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `dna_LLR+dna_LVD_L2+dna_LID+prot_LLR+prot_LVD_Relative` | 5 | 0.6913 | 0.7088 | 0.4805 | 0.9143 | 0.8968 |
| 2 | `dna_LLR+dna_LVD_Cosine+dna_LID+prot_LLR+prot_LID+prot_LVD_Relative` | 6 | 0.6867 | 0.7058 | 0.4735 | 0.9131 | 0.8942 |
| 3 | `dna_LLR+dna_LVD_L2+dna_LID+dna_LVD_Relative+prot_LLR+prot_LVD_L2+prot_LVD_Cosine` | 7 | 0.6866 | 0.7079 | 0.4773 | 0.9115 | 0.8930 |
| 4 | `dna_LVD_Cosine+dna_LID+dna_LVD_Relative+prot_LLR+prot_LID+prot_LVD_Relative` | 6 | 0.6858 | 0.7038 | 0.4819 | 0.9131 | 0.8952 |
| 5 | `dna_LLR+dna_LVD_L2+dna_LID+dna_LVD_Relative+prot_LLR+prot_LVD_L2` | 6 | 0.6857 | 0.7075 | 0.4658 | 0.9141 | 0.8963 |
| 6 | `dna_LVD_Cosine+dna_LID+prot_LLR+prot_LVD_L2` | 4 | 0.6856 | 0.7020 | 0.4713 | 0.9136 | 0.8957 |
| 7 | `dna_LLR+dna_LID+dna_LVD_Relative+prot_LLR+prot_LVD_L2+prot_LVD_Relative` | 6 | 0.6856 | 0.7041 | 0.4640 | 0.9152 | 0.8978 |
| 8 | `dna_LLR+dna_LVD_Cosine+dna_LID+prot_LLR+prot_LVD_L2+prot_LVD_Relative` | 6 | 0.6855 | 0.7002 | 0.4743 | 0.9142 | 0.8957 |
| 9 | `dna_LLR+dna_LVD_Cosine+dna_LID+dna_LVD_Relative+prot_LLR` | 5 | 0.6851 | 0.7033 | 0.4809 | 0.9129 | 0.8955 |
| 10 | `dna_LLR+dna_LVD_L2+dna_LVD_Relative+prot_LLR` | 4 | 0.6849 | 0.7036 | 0.4678 | 0.9135 | 0.8966 |

## Recommendation

Use rank 1 as the primary candidate for the next full-model benchmark:

```text
dna_LLR + dna_LVD_L2 + dna_LID + prot_LLR + prot_LVD_Relative
```

Keep these as secondary candidates:

- Rank 4: best minimum MCC among the top-10 list and only 6 features.
- Rank 6: compact 4-feature configuration with nearly the same mean MCC as ranks 2-5.
- Rank 9: good balance between mean MCC and minimum MCC with 5 features.

For a more conservative cross-dataset choice, the highest minimum-MCC configuration over all 1023 subsets was:

```text
dna_LLR + dna_LID + dna_LVD_Relative + prot_LLR + prot_LID + prot_LVD_Relative
```

It has mean MCC `0.6830` and minimum MCC `0.4882`, so it is slightly weaker on average but more stable across the four datasets.

## Dataset-Specific Note

The leading configuration by average MCC is not the winner on every dataset. The MCC values for rank 1 are:

| Dataset | MCC |
|---|---:|
| `Train3Val_ClinVarHQ` | 0.8672 |
| `Train3Val_ProteinGym` | 0.6103 |
| `Train3Val_Test` | 0.8072 |
| `Train3Val_UniProt` | 0.4805 |

This spread is why the report retains both average and worst-case MCC instead of selecting solely by the ClinVarHQ score.
