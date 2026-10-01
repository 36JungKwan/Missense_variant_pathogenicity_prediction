# LID-k and Architecture Ablation

## Scope

- Train/validation context: `train3` / `val`
- DNA model: `nt_v2_500m`
- Protein model: `esm1b_650m`
- Pooling: `mean`
- Test datasets: 4
- Metrics file: `global_leaderboard_metrics.csv`
- Completed rows: 88

All rows include `Val_*` metrics and test metrics.

## Stage 1: LID and k Ablation

The fixed geometry control was:

```text
dna_LVD_L2 + dna_LVD_Relative + prot_LLR
+ prot_LVD_Cosine + prot_LVD_Relative
```

The LID variants added DNA LID, protein LID, or both for `k = 8, 16, 32, 64`.

| Configuration | Val MCC | Mean test MCC | Minimum test MCC |
|---|---:|---:|---:|
| `lid_dna_k32` | 0.7878 | 0.6744 | 0.4649 |
| `lid_both_k16` | 0.7873 | 0.6708 | 0.4585 |
| `lid_control` | 0.7866 | 0.6772 | 0.4662 |
| `lid_prot_k64` | 0.7865 | 0.6787 | 0.4652 |
| `lid_prot_k32` | 0.7864 | 0.6607 | 0.4348 |
| `lid_dna_k64` | 0.7857 | 0.6740 | 0.4498 |
| `lid_dna_k8` | 0.7855 | 0.6785 | 0.4747 |
| `lid_both_k8` | 0.7839 | 0.6758 | 0.4665 |
| `lid_prot_k16` | 0.7833 | 0.6624 | 0.4434 |
| `lid_dna_k16` | 0.7830 | 0.6715 | 0.4654 |
| `lid_both_k32` | 0.7830 | 0.6665 | 0.4430 |
| `lid_both_k64` | 0.7820 | 0.6830 | 0.4672 |
| `lid_prot_k8` | 0.7786 | 0.6746 | 0.4649 |

Validation-selected geometry for Stage 2:

```text
dna_LVD_L2 + dna_LVD_Relative + prot_LLR
+ prot_LVD_Cosine + prot_LVD_Relative + dna_LID
```

with `k=32`.

## Stage 2: Architecture Ablation

The Stage 2 geometry was selected using validation only.

| Architecture | Val MCC | Val AUROC | Mean test MCC | Minimum test MCC |
|---|---:|---:|---:|---:|
| `Pure_XGBoost_Concat` | 0.7867 | 0.9729 | 0.6798 | 0.4757 |
| `PyTorch_Concat` | 0.7833 | 0.9667 | 0.6689 | 0.4674 |
| `PyTorch_Transformer` | 0.7818 | 0.9666 | 0.6739 | 0.4794 |
| `PyTorch_CrossAttn` | 0.7762 | 0.9652 | 0.6554 | 0.4244 |
| `PyTorch_Gating` | 0.7709 | 0.9660 | 0.6610 | 0.4530 |
| `Hybrid_Concat_XGBoost` | 0.7661 | 0.9666 | 0.6641 | 0.4542 |
| `Hybrid_Gating_XGBoost` | 0.7657 | 0.9658 | 0.6600 | 0.4363 |
| `Hybrid_Transformer_XGBoost` | 0.7635 | 0.9664 | 0.6729 | 0.4629 |
| `Hybrid_CrossAttn_XGBoost` | 0.7597 | 0.9670 | 0.6522 | 0.4398 |

## Conclusion

- Validation-selected LID variant: `dna_LID` with `k=32`.
- The LID gain over the no-LID control is small: `0.7878` versus `0.7866` Val MCC.
- `Pure_XGBoost_Concat` remains the strongest architecture for this fixed geometry on validation.
- `PyTorch_Transformer` has the best minimum test MCC among the architecture rows, despite lower mean MCC than Pure XGBoost.
- The LID result should be treated as a weak, architecture-dependent signal rather than a universally useful feature.
