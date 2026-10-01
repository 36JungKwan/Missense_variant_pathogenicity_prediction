# MCC config consistency: `batch_run_20260904_1800`

## Phạm vi và quy ước

- Chỉ dùng các dòng có `Source=OURS`; không dùng các model SOTA để xác định top 1.
- Metric duy nhất: `MCC`.
- Top 1 được tính riêng theo từng dataset; ngưỡng đạt là `top1_MCC - MCC <= 0.02`.
- Một config được coi là giống nhau khi toàn bộ 6 trường sau giống nhau: `Pooling`, `Ablation`, `DNA_Model`, `Prot_Model`, `Network`, `Geom_Features`.
- Các dòng trùng dataset/config được gộp bằng MCC lớn nhất; giá trị rỗng được chuẩn hóa thành `None`.

## Tổng quan

- OURS rows ban đầu: **6,972**.
- Dataset/config duy nhất sau khi gộp: **6,468**.
- Config duy nhất xuất hiện trên toàn bộ 4 dataset: **1,617**.
- Config đạt ngưỡng trên ít nhất một dataset: **129**.
- Config đạt ngưỡng trên cả 4 dataset: **0**.

## Top 1 OURS theo dataset

| Dataset              |   Top1_MCC | Top1_config                                                                                                                                          |
|:---------------------|-----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------|
| Train3Val_ClinVarHQ  |     0.901  | Pooling=mean | Ablation=bio_core_dna_prot_spliceai | DNA_Model=nt_v2_500m | Prot_Model=esm1b_650m | Network=Pure_XGBoost_Concat | Geom_Features=None |
| Train3Val_ProteinGym |     0.6837 | Pooling=center | Ablation=bio_geom_prot | DNA_Model=nt_v1_500m | Prot_Model=esmc_600m | Network=PyTorch_Concat | Geom_Features=None                  |
| Train3Val_Test       |     0.8141 | Pooling=center | Ablation=bio_geom_prot | DNA_Model=nt_v1_500m | Prot_Model=esmc_600m | Network=Pure_XGBoost_Concat | Geom_Features=None             |
| Train3Val_UniProt    |     0.493  | Pooling=center | Ablation=bio_geom_prot | DNA_Model=nt_v1_500m | Prot_Model=esmc_600m | Network=PyTorch_Concat | Geom_Features=None                  |

## Số config nằm trong ngưỡng 0.02

| Dataset              |   Configs_within_0.02 |
|:---------------------|----------------------:|
| Train3Val_ClinVarHQ  |                     8 |
| Train3Val_ProteinGym |                     1 |
| Train3Val_Test       |                   115 |
| Train3Val_UniProt    |                    15 |

## Kiểm tra tính ổn định xuyên dataset

**Không có config nào** đồng thời nằm trong ngưỡng 0.02 của top 1 trên cả 4 dataset.

Mức bao phủ tốt nhất chỉ là **2/4 dataset**.

### Ngưỡng nhỏ nhất để có config phủ đủ tất cả dataset

Cần nới ngưỡng tối thiểu lên **0.0608**. Đây là giá trị nhỏ nhất của `max(delta_to_top1)` trên một config xuất hiện ở toàn bộ dataset.

| Pooling   | Ablation               | DNA_Model   | Prot_Model   | Network        | Geom_Features   |   Train3Val_ClinVarHQ |   Train3Val_ProteinGym |   Train3Val_Test |   Train3Val_UniProt |   Required_tolerance |
|:----------|:-----------------------|:------------|:-------------|:---------------|:----------------|----------------------:|-----------------------:|-----------------:|--------------------:|---------------------:|
| center    | bio_core_prot_spliceai | None        | esmc_600m    | PyTorch_Concat | None            |                0.0608 |                 0.0594 |           0.0215 |              0.0413 |               0.0608 |

### Các config có mức bao phủ tốt nhất

| Pooling   | Ablation                        | DNA_Model   | Prot_Model   | Network             | Geom_Features                                                                           |   Datasets_present |   Datasets_within_0.02 |
|:----------|:--------------------------------|:------------|:-------------|:--------------------|:----------------------------------------------------------------------------------------|-------------------:|-----------------------:|
| center    | bio_geom_prot                   | nt_v1_500m  | esmc_600m    | PyTorch_Concat      | None                                                                                    |                  4 |                      2 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2+dna_LVD_Cosine+dna_LID+prot_LLR+prot_LVD_Cosine                      |                  4 |                      2 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2+dna_LVD_Cosine+dna_LID+prot_LLR+prot_LVD_L2+prot_LVD_Cosine          |                  4 |                      2 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2+dna_LVD_Cosine+dna_LID+prot_LLR+prot_LVD_L2+prot_LVD_Cosine+prot_LID |                  4 |                      2 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2+dna_LVD_Cosine+prot_LLR+prot_LVD_Cosine                              |                  4 |                      2 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2+dna_LVD_Cosine+prot_LLR+prot_LVD_Cosine+prot_LID                     |                  4 |                      2 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2+prot_LLR+prot_LVD_L2+prot_LVD_Cosine                                 |                  4 |                      2 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LVD_L2+dna_LID+prot_LLR+prot_LVD_L2+prot_LVD_Cosine                                 |                  4 |                      2 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | prot_LLR+prot_LVD_Cosine+prot_LID                                                       |                  4 |                      2 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | prot_LLR+prot_LVD_L2                                                                    |                  4 |                      2 |

## Top 10 config gần top 1: `Train3Val_ClinVarHQ`

| Pooling   | Ablation                        | DNA_Model   | Prot_Model   | Network             | Geom_Features                                      |    MCC |   Delta_to_top1 |
|:----------|:--------------------------------|:------------|:-------------|:--------------------|:---------------------------------------------------|-------:|----------------:|
| mean      | bio_core_dna_prot_spliceai      | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | None                                               | 0.901  |          0      |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LVD_Cosine+prot_LVD_L2                         | 0.8894 |          0.0116 |
| cls       | bio_core_dna_prot_spliceai      | nt_v3_650m  | esm1b_650m   | Pure_XGBoost_Concat | None                                               | 0.8879 |          0.0131 |
| cls       | bio_core_dna_prot_spliceai      | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | None                                               | 0.8852 |          0.0158 |
| mean      | bio_core_dna_prot_spliceai      | nt_v3_650m  | esm2_650m    | Pure_XGBoost_Concat | None                                               | 0.8819 |          0.0191 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_Cosine+prot_LVD_L2+prot_LVD_Cosine | 0.8814 |          0.0196 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2                                 | 0.8814 |          0.0196 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_Cosine+dna_LID+prot_LVD_Cosine     | 0.8812 |          0.0198 |

## Top 10 config gần top 1: `Train3Val_ProteinGym`

| Pooling   | Ablation      | DNA_Model   | Prot_Model   | Network        | Geom_Features   |    MCC |   Delta_to_top1 |
|:----------|:--------------|:------------|:-------------|:---------------|:----------------|-------:|----------------:|
| center    | bio_geom_prot | nt_v1_500m  | esmc_600m    | PyTorch_Concat | None            | 0.6837 |               0 |

## Top 10 config gần top 1: `Train3Val_Test`

| Pooling   | Ablation                        | DNA_Model   | Prot_Model   | Network             | Geom_Features                                                                   |    MCC |   Delta_to_top1 |
|:----------|:--------------------------------|:------------|:-------------|:--------------------|:--------------------------------------------------------------------------------|-------:|----------------:|
| center    | bio_geom_prot                   | nt_v1_500m  | esmc_600m    | Pure_XGBoost_Concat | None                                                                            | 0.8141 |          0      |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LID+prot_LLR+prot_LID                                                       | 0.8062 |          0.0079 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2+dna_LID+prot_LLR+prot_LVD_L2+prot_LVD_Cosine                 | 0.8057 |          0.0084 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | prot_LLR+prot_LVD_Cosine+prot_LID                                               | 0.8055 |          0.0086 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2+dna_LVD_Cosine+prot_LLR+prot_LVD_L2                          | 0.8046 |          0.0095 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | prot_LLR+prot_LVD_L2                                                            | 0.8036 |          0.0105 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2+dna_LVD_Cosine+prot_LLR+prot_LVD_L2+prot_LVD_Cosine+prot_LID | 0.8031 |          0.011  |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2+dna_LVD_Cosine+prot_LLR+prot_LVD_Cosine+prot_LID             | 0.803  |          0.0111 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LVD_Cosine+dna_LID+prot_LLR+prot_LVD_L2                                     | 0.803  |          0.0111 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LVD_L2+prot_LLR+prot_LVD_L2+prot_LVD_Cosine+prot_LID                        | 0.803  |          0.0111 |

## Top 10 config gần top 1: `Train3Val_UniProt`

| Pooling   | Ablation                        | DNA_Model   | Prot_Model   | Network               | Geom_Features                                                                  |    MCC |   Delta_to_top1 |
|:----------|:--------------------------------|:------------|:-------------|:----------------------|:-------------------------------------------------------------------------------|-------:|----------------:|
| center    | bio_geom_prot                   | nt_v1_500m  | esmc_600m    | PyTorch_Concat        | None                                                                           | 0.493  |          0      |
| center    | bio_dna_geom_prot               | nt_v1_500m  | esmc_600m    | Hybrid_Gating_XGBoost | None                                                                           | 0.4891 |          0.0039 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat   | dna_LLR+dna_LVD_L2+dna_LVD_Cosine+dna_LID+prot_LLR+prot_LVD_L2+prot_LVD_Cosine | 0.489  |          0.004  |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat   | dna_LVD_L2+dna_LID+prot_LLR+prot_LVD_L2+prot_LVD_Cosine                        | 0.485  |          0.008  |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat   | prot_LLR+prot_LVD_Cosine+prot_LID                                              | 0.4846 |          0.0084 |
| mean      | bio_core_dna_prot_spliceai      | nt_v3_650m  | esm1b_650m   | PyTorch_Transformer   | None                                                                           | 0.4826 |          0.0104 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat   | dna_LLR+dna_LVD_L2+prot_LLR+prot_LVD_L2+prot_LVD_Cosine                        | 0.4799 |          0.0131 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat   | dna_LLR+dna_LVD_L2+dna_LVD_Cosine+prot_LLR+prot_LVD_Cosine                     | 0.4787 |          0.0143 |
| center    | prot_spliceai                   | None        | esmc_600m    | PyTorch_Concat        | None                                                                           | 0.4786 |          0.0144 |
| mean      | bio_geom_prot                   | nt_v2_500m  | esm1b_650m   | PyTorch_Concat        | None                                                                           | 0.4782 |          0.0148 |
