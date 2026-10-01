# MCC config consistency: `batch_run_20260902_1626`

## Phạm vi và quy ước

- Chỉ dùng các dòng có `Source=OURS`; không dùng các model SOTA để xác định top 1.
- Metric duy nhất: `MCC`.
- Top 1 được tính riêng theo từng dataset; ngưỡng đạt là `top1_MCC - MCC <= 0.02`.
- Một config được coi là giống nhau khi toàn bộ 6 trường sau giống nhau: `Pooling`, `Ablation`, `DNA_Model`, `Prot_Model`, `Network`, `Geom_Features`.
- Các dòng trùng dataset/config được gộp bằng MCC lớn nhất; giá trị rỗng được chuẩn hóa thành `None`.

## Tổng quan

- OURS rows ban đầu: **6,804**.
- Dataset/config duy nhất sau khi gộp: **6,468**.
- Config duy nhất xuất hiện trên toàn bộ 4 dataset: **1,617**.
- Config đạt ngưỡng trên ít nhất một dataset: **290**.
- Config đạt ngưỡng trên cả 4 dataset: **0**.

## Top 1 OURS theo dataset

| Dataset              |   Top1_MCC | Top1_config                                                                                                                                                                                                      |
|:---------------------|-----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Train1Val_ClinVarHQ  |     0.8832 | Pooling=mean | Ablation=bio_core_dna_prot_spliceai | DNA_Model=nt_v1_500m | Prot_Model=esm2_650m | Network=Hybrid_Gating_XGBoost | Geom_Features=None                                                            |
| Train1Val_ProteinGym |     0.7253 | Pooling=mean | Ablation=bio_core_dna_prot_spliceai_geom | DNA_Model=nt_v2_500m | Prot_Model=esm1b_650m | Network=Pure_XGBoost_Concat | Geom_Features=dna_LLR+dna_LVD_L2+dna_LVD_Cosine+dna_LID+prot_LLR+prot_LID |
| Train1Val_Test       |     0.8344 | Pooling=center | Ablation=bio_prot | DNA_Model=None | Prot_Model=esmc_600m | Network=PyTorch_Concat | Geom_Features=None                                                                                         |
| Train1Val_UniProt    |     0.5247 | Pooling=center | Ablation=bio_dna_prot | DNA_Model=nt_v3_650m | Prot_Model=esmc_600m | Network=Hybrid_CrossAttn_XGBoost | Geom_Features=None                                                                     |

## Số config nằm trong ngưỡng 0.02

| Dataset              |   Configs_within_0.02 |
|:---------------------|----------------------:|
| Train1Val_ClinVarHQ  |                    12 |
| Train1Val_ProteinGym |                    99 |
| Train1Val_Test       |                   186 |
| Train1Val_UniProt    |                    21 |

## Kiểm tra tính ổn định xuyên dataset

**Không có config nào** đồng thời nằm trong ngưỡng 0.02 của top 1 trên cả 4 dataset.

Mức bao phủ tốt nhất chỉ là **3/4 dataset**.

### Ngưỡng nhỏ nhất để có config phủ đủ tất cả dataset

Cần nới ngưỡng tối thiểu lên **0.0500**. Đây là giá trị nhỏ nhất của `max(delta_to_top1)` trên một config xuất hiện ở toàn bộ dataset.

| Pooling   | Ablation                        | DNA_Model   | Prot_Model   | Network             | Geom_Features                                      |   Train1Val_ClinVarHQ |   Train1Val_ProteinGym |   Train1Val_Test |   Train1Val_UniProt |   Required_tolerance |
|:----------|:--------------------------------|:------------|:-------------|:--------------------|:---------------------------------------------------|----------------------:|-----------------------:|-----------------:|--------------------:|---------------------:|
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LVD_L2+dna_LVD_Cosine+prot_LLR+prot_LVD_Cosine |                0.0315 |                   0.05 |           0.0137 |              0.0438 |                 0.05 |

### Các config có mức bao phủ tốt nhất

| Pooling   | Ablation                   | DNA_Model   | Prot_Model   | Network                    | Geom_Features   |   Datasets_present |   Datasets_within_0.02 |
|:----------|:---------------------------|:------------|:-------------|:---------------------------|:----------------|-------------------:|-----------------------:|
| center    | bio_core_dna_prot          | nt_v3_650m  | esm1v_650m   | PyTorch_Gating             | None            |                  4 |                      3 |
| center    | bio_core_dna_prot_spliceai | nt_v3_650m  | esm1v_650m   | PyTorch_Gating             | None            |                  4 |                      3 |
| center    | bio_core_dna_prot_spliceai | nt_v3_650m  | esmc_600m    | Hybrid_Transformer_XGBoost | None            |                  4 |                      3 |

## Top 10 config gần top 1: `Train1Val_ClinVarHQ`

| Pooling   | Ablation                   | DNA_Model   | Prot_Model   | Network                    | Geom_Features   |    MCC |   Delta_to_top1 |
|:----------|:---------------------------|:------------|:-------------|:---------------------------|:----------------|-------:|----------------:|
| mean      | bio_core_dna_prot_spliceai | nt_v1_500m  | esm2_650m    | Hybrid_Gating_XGBoost      | None            | 0.8832 |          0      |
| cls       | bio_core_dna_prot_spliceai | nt_v1_500m  | esm2_650m    | Hybrid_Concat_XGBoost      | None            | 0.8748 |          0.0084 |
| mean      | bio_core_dna_prot          | nt_v1_500m  | esm1v_650m   | Hybrid_Transformer_XGBoost | None            | 0.8687 |          0.0145 |
| center    | bio_core_dna_prot_spliceai | nt_v3_650m  | esm2_650m    | Hybrid_CrossAttn_XGBoost   | None            | 0.868  |          0.0152 |
| center    | bio_core_dna_prot_spliceai | nt_v3_650m  | esmc_600m    | Pure_XGBoost_Concat        | None            | 0.8679 |          0.0153 |
| center    | bio_core_prot_spliceai     | None        | esmc_600m    | Pure_XGBoost_Concat        | None            | 0.8669 |          0.0163 |
| center    | bio_core_dna_prot_spliceai | nt_v3_650m  | esm2_650m    | Pure_XGBoost_Concat        | None            | 0.8658 |          0.0174 |
| center    | bio_core_dna_prot_spliceai | nt_v2_500m  | esmc_600m    | Hybrid_CrossAttn_XGBoost   | None            | 0.865  |          0.0182 |
| center    | bio_core_dna_prot_spliceai | nt_v3_650m  | esmc_600m    | Hybrid_Transformer_XGBoost | None            | 0.865  |          0.0182 |
| center    | bio_core_prot_spliceai     | None        | esm2_650m    | Pure_XGBoost_Concat        | None            | 0.8648 |          0.0184 |

## Top 10 config gần top 1: `Train1Val_ProteinGym`

| Pooling   | Ablation                        | DNA_Model   | Prot_Model   | Network             | Geom_Features                                                        |    MCC |   Delta_to_top1 |
|:----------|:--------------------------------|:------------|:-------------|:--------------------|:---------------------------------------------------------------------|-------:|----------------:|
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2+dna_LVD_Cosine+dna_LID+prot_LLR+prot_LID          | 0.7253 |          0      |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LID+prot_LLR+prot_LID                                    | 0.7215 |          0.0038 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2+prot_LLR+prot_LVD_Cosine+prot_LID                 | 0.7193 |          0.006  |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2+dna_LVD_Cosine+prot_LLR+prot_LVD_L2+prot_LID      | 0.719  |          0.0063 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+dna_LVD_L2+prot_LLR+prot_LVD_L2+prot_LID                     | 0.7189 |          0.0064 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LVD_Cosine+dna_LID+prot_LLR+prot_LVD_L2+prot_LVD_Cosine          | 0.7189 |          0.0064 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LVD_Cosine+dna_LID+prot_LLR+prot_LVD_L2+prot_LID                 | 0.7188 |          0.0065 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LVD_Cosine+dna_LID+prot_LLR+prot_LVD_L2+prot_LVD_Cosine+prot_LID | 0.7188 |          0.0065 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LLR+prot_LLR+prot_LVD_L2+prot_LID                                | 0.7187 |          0.0066 |
| mean      | bio_core_dna_prot_spliceai_geom | nt_v2_500m  | esm1b_650m   | Pure_XGBoost_Concat | dna_LVD_L2+dna_LVD_Cosine+prot_LLR+prot_LVD_Cosine+prot_LID          | 0.7187 |          0.0066 |

## Top 10 config gần top 1: `Train1Val_Test`

| Pooling   | Ablation                   | DNA_Model   | Prot_Model   | Network                  | Geom_Features   |    MCC |   Delta_to_top1 |
|:----------|:---------------------------|:------------|:-------------|:-------------------------|:----------------|-------:|----------------:|
| center    | bio_prot                   | None        | esmc_600m    | PyTorch_Concat           | None            | 0.8344 |          0      |
| center    | bio_dna_prot               | nt_v3_650m  | esmc_600m    | Hybrid_Concat_XGBoost    | None            | 0.8334 |          0.001  |
| center    | bio_dna_prot               | nt_v3_650m  | esmc_600m    | Hybrid_Gating_XGBoost    | None            | 0.8331 |          0.0013 |
| center    | bio_core_dna_prot_spliceai | nt_v3_650m  | esmc_600m    | Hybrid_Gating_XGBoost    | None            | 0.8328 |          0.0016 |
| center    | bio_core_dna_prot_spliceai | nt_v3_650m  | esm1v_650m   | Hybrid_CrossAttn_XGBoost | None            | 0.8307 |          0.0037 |
| mean      | bio_core_dna_prot_spliceai | nt_v2_500m  | esmc_600m    | PyTorch_Gating           | None            | 0.8301 |          0.0043 |
| center    | bio_core_dna_prot_spliceai | nt_v2_500m  | esmc_600m    | Hybrid_Gating_XGBoost    | None            | 0.8297 |          0.0047 |
| center    | bio_dna_prot               | nt_v3_650m  | esmc_600m    | Hybrid_CrossAttn_XGBoost | None            | 0.8296 |          0.0048 |
| center    | bio_core_dna_prot_spliceai | nt_v3_650m  | esmc_600m    | PyTorch_Concat           | None            | 0.8289 |          0.0055 |
| center    | bio_core_prot_spliceai     | None        | esmc_600m    | Pure_XGBoost_Concat      | None            | 0.8287 |          0.0057 |

## Top 10 config gần top 1: `Train1Val_UniProt`

| Pooling   | Ablation                   | DNA_Model   | Prot_Model   | Network                    | Geom_Features   |    MCC |   Delta_to_top1 |
|:----------|:---------------------------|:------------|:-------------|:---------------------------|:----------------|-------:|----------------:|
| center    | bio_dna_prot               | nt_v3_650m  | esmc_600m    | Hybrid_CrossAttn_XGBoost   | None            | 0.5247 |          0      |
| center    | bio_core_dna_prot          | nt_v3_650m  | esm1v_650m   | PyTorch_Transformer        | None            | 0.5246 |          0.0001 |
| center    | bio_core_dna_prot_spliceai | nt_v3_650m  | esm1v_650m   | PyTorch_Transformer        | None            | 0.5175 |          0.0072 |
| center    | bio_dna_prot               | nt_v3_650m  | esmc_600m    | PyTorch_CrossAttn          | None            | 0.5119 |          0.0128 |
| mean      | bio_dna_prot               | nt_v3_650m  | esm1b_650m   | PyTorch_Transformer        | None            | 0.5115 |          0.0132 |
| center    | bio_core_dna_prot          | nt_v2_500m  | esm1v_650m   | PyTorch_Concat             | None            | 0.5113 |          0.0134 |
| center    | bio_core_dna_prot          | nt_v3_650m  | esm1v_650m   | Hybrid_Transformer_XGBoost | None            | 0.5081 |          0.0166 |
| center    | bio_core_dna_prot_spliceai | nt_v3_650m  | esmc_600m    | Hybrid_CrossAttn_XGBoost   | None            | 0.508  |          0.0167 |
| center    | bio_core_dna_prot          | nt_v3_650m  | esm1v_650m   | Hybrid_Gating_XGBoost      | None            | 0.5079 |          0.0168 |
| center    | bio_dna_prot               | nt_v3_650m  | esmc_600m    | PyTorch_Gating             | None            | 0.5078 |          0.0169 |
