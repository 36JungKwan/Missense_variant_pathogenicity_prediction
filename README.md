# 🧬 Bioinformatics Sequence Research - AiTA Lab

Multi-task deep learning framework for biosequence analysis, pathogenicity prediction, and protein/nucleotide feature extraction.

**Research Focus:** Utilize pre-trained language models (Nucleotide Transformer, ESM-2) to build models for predicting biological properties of genetic variants.

---

## 📋 Project Overview

This project focuses on 3 main tasks:

| Task | Description | Data | Model |
|------|-------------|------|-------|
| **Task 1: Splicing Prediction** | Predict splicing site type (donor/acceptor) | Sequence ~200bp | NT embeddings |
| **Task 2: Protein Prediction** | Predict protein properties from sequence | Protein sequence | ESM-2 embeddings |
| **Task 3: Variant Pathogenicity** | Classify variants (pathogenic/benign) | ClinVar + DNA/Protein seq | Multi-modal Fusion Model |

---

## 📁 Directory Structure

```
Bio_sequence_Research_AITALAB/
│
├── data_processing/                          # 📊 Data preprocessing & preparation
│   ├── dataset1_ClinVar_preprocess_variant_summary.ipynb
│   ├── dataset2_map_csq_hgvsc_aDun.ipynb     # Map CSQ & HGVS-C
│   ├── dataset2_map_ref_alt_sequence_dna.ipynb
│   ├── dataset2_map_ref_alt_sequence_protein.ipynb
│   ├── dataset3_sequence_gencode.ipynb       # Extract sequences from GENCODE
│
├── tools/                                     # 🛠️ Supporting tools
│   ├── gnomAD_map_vep.ipynb                  # Map VEP annotations
│   ├── test_parse_hgvsc_offset.ipynb         # Parse HGVS-C format
│
├── train/                                     # 🎯 Training pipelines
│   │
│   ├── task1_splicing_prediction/            # Splicing site prediction
│   │   ├── data_preparation/
│   │   │   ├── data_prepare.ipynb            # Data preparation
│   │   │   ├── train_test_split.py
│   │   │   ├── ratio_split.py
│   │   │   └── extract_embed.py
│   │   └── training/
│   │       ├── main.ipynb                    # Training notebook
│   │       ├── model.py                      # LSTM model
│   │       ├── dataset.py                    # PyTorch Dataset
│   │       ├── train_set.py
│   │       ├── train_full.py
│   │       ├── metrics.py
│   │       ├── cm_visualize.py
│   │       └── fileio.py
│   │
│   ├── task2_protein_prediction/             # Protein property prediction
│   │
│   └── task3_variant_prediction/             # ⭐ Variant pathogenicity prediction (MAIN)
│       ├── config.py                         # Configuration
│       ├── split_data.py                     # Split by chromosome
│       ├── precompute_embeddings.py          # Extract NT + ESM-2 embeddings
│       ├── dataset.py                        # PyTorch Dataset
│       ├── model.py                          # Multi-modal Fusion model
│       ├── train.py                          # Training with tracking
│       ├── main.ipynb                        # Full pipeline
│       ├── README.md                         # Task-specific guide
│       ├── data/                             # Train/Val/Test splits
│       ├── embeddings/                       # Precomputed embeddings
│       ├── experiments/                      # Experiment configs & results
│       └── runs/                             # TensorBoard logs
│
└── README.md                                  # This file
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.9+
# CUDA 11.8+ (recommended for GPU support)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets numpy pandas scikit-learn matplotlib seaborn tensorboard jupyter
pip install pyarrow biopython pysam  # Bioinformatics tools
```

### Environment Configuration

Create a `.env` file or set environment variables:

```bash
# Task 3 data path
TASK3_PARQUET=<path_to>/variant_protein_sequence_101aa.parquet

# Hugging Face token (if needed)
HUGGING_FACE_HUB_TOKEN=<your_token>
```

---

## 📊 Project Pipeline

### 1️⃣ Data Processing (`data_processing/`)

**Goal:** Prepare data from different sources (ClinVar, GENCODE, VEP) into standard format

| Notebook | Purpose |
|----------|---------|
| `dataset1_ClinVar_preprocess_variant_summary.ipynb` | Filter & preprocess ClinVar variants |
| `dataset2_map_csq_hgvsc_aDun.ipynb` | Map CSQ → HGVS-C nomenclature |
| `dataset2_map_ref_alt_sequence_dna.ipynb` | Extract DNA sequences (ref & alt) |
| `dataset2_map_ref_alt_sequence_protein.ipynb` | Extract protein sequences |
| `dataset3_sequence_gencode.ipynb` | Get sequences from GENCODE reference |

**Output:** Parquet files with variant + sequences (DNA 601bp, Protein 101aa)

---

### 2️⃣ Task 1: Splicing Prediction (`train/task1_splicing_prediction/`)

**Goal:** Predict splicing site type from DNA sequence

**Pipeline:**
```
Raw Data (.csv) → Train/Test Split → Val Split → Model Training → Metrics
```

**Run:**
```bash
cd train/task1_splicing_prediction/data_preparation/
jupyter notebook data_prepare.ipynb

cd ../training/
jupyter notebook main.ipynb
```

---

### 3️⃣ Task 2: Protein Prediction (`train/task2_protein_prediction/`)

**Goal:** Predict protein properties/functions (TODO/In Progress)

---

### 4️⃣ Task 3: Variant Pathogenicity Prediction ⭐ (`train/task3_variant_prediction/`)

**Goal:** Classify genetic variants as Pathogenic or Benign

**Model Architecture:**
```
Input: DNA & Protein sequences from variants
       ↓
[DNA Seq] → Nucleotide Transformer (NT) → E_dna_ref, E_dna_alt
[Prot Seq] → ESM-2 → E_prot_ref, E_prot_alt
       ↓
Fusion Layer: [E_ref, E_alt, E_alt - E_ref]
       ↓
Concat DNA + Protein embeddings
       ↓
MLP Classifier → Pathogenic (1) / Benign (0)
```

**Pipeline:**

```bash
cd train/task3_variant_prediction/

# 1. Split data by chromosome (chr20/21 → test, rest → train/val)
python split_data.py

# 2. Precompute embeddings (NT + ESM-2)
python precompute_embeddings.py

# 3. Run training with experiment tracking
python train.py

# Or run full pipeline from notebook
jupyter notebook main.ipynb
```

**Key Features:**
- ✅ Multi-modal fusion (DNA + Protein)
- ✅ Automatic experiment tracking (config, results, checkpoints)
- ✅ TensorBoard logging
- ✅ Best model selection
- ✅ Train/Val/Test splits

**View Results:**
```bash
# TensorBoard
tensorboard --logdir=runs/

# Results JSON
cat experiments/experiment_*/results.json
```

---

## 🧠 Models & Pre-trained Embeddings

| Model | Purpose | Source | Input Size |
|-------|---------|--------|-----------|
| **Nucleotide Transformer (NT)** | DNA embedding extraction | InstaDeepAI/nucleotide-transformer-500m-human-ref | 601bp |
| **ESM-2** | Protein embedding extraction | facebook/esm2_t33_650M_UR50D | 101aa |
| **Custom MLP Classifier** | Pathogenicity prediction | Fusion model | 1024 (512*2) |

---

## 📈 Data Statistics

### Task 3 (Variant Prediction)
- **Source:** ClinVar variants + mapped sequences
- **Splits:**
  - **Train:** All variants except chr20/21
  - **Val:** 15% of training (stratified)
  - **Test:** chr20, chr21
- **Labels:** Pathogenic (1), Benign (0)
- **Sequence Length:** DNA 601bp, Protein 101aa

---

## 🔧 Configuration

**Main Config File:** [`train/task3_variant_prediction/config.py`](train/task3_variant_prediction/config.py)

```python
# Hyperparameters
LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 128
DROPOUT = 0.2
PATIENCE = 5

# Embeddings
PROJ_DIM = 512
FUSION_HIDDEN = [512, 256]

# Paths
TEST_CHROMS = {"chr20", "chr21"}
VAL_RATIO = 0.15
SEED = 42
```

---

## 📊 Results & Monitoring

### Experiment Tracking

Each training run saves:
- **args.json:** Command-line arguments
- **config.json:** Configuration parameters
- **config.py:** Copy of config file
- **results.json:** Final metrics (accuracy, precision, recall, F1)
- **tensorboard/:** TensorBoard events

```
experiments/
├── experiment_1/
│   ├── args.json
│   ├── config.json
│   ├── results.json
│   └── tensorboard/
└── experiment_N/
    └── ...
```

### View Results

```bash
# List all experiments
ls train/task3_variant_prediction/experiments/

# View results
cat train/task3_variant_prediction/experiments/experiment_4/results.json
```

---

## 💡 Usage Examples

### Inference (New Variants)

```python
import torch
from train.task3_variant_prediction.model import FusionClassifier
from train.task3_variant_prediction.dataset import VariantDataset

# Load trained model
model = FusionClassifier(dna_emb_dim=1024, prot_emb_dim=1024)
model.load_state_dict(torch.load('best_fusion_model.pt'))

# Make predictions
logits = model(dna_embedding, prot_embedding)
predictions = torch.sigmoid(logits)
```

### Add New Dataset

1. Add preprocessing script to `data_processing/`
2. Output parquet format: `[variant_id, sequence_dna, sequence_protein, label, chrom]`
3. Update `config.py` path
4. Run training pipeline

---

## 📚 References

- **ClinVar:** https://www.ncbi.nlm.nih.gov/clinvar/
- **Nucleotide Transformer:** https://github.com/instadeepai/nucleotide-transformer
- **ESM-2:** https://github.com/facebookresearch/protein-folding
- **VEP:** https://www.ensembl.org/info/docs/tools/vep/

---

## 🤝 Contributing

To add features or fix bugs:

1. Create feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -m "Add your feature"`
3. Push: `git push origin feature/your-feature`
4. Create pull request

---

## 📝 Notes

- All embeddings are precomputed from pre-trained models (not fine-tuned)
- Test set is fixed as chr20/21 for benchmarking
- Experiment tracking is automatic - no manual logging needed
- Stratified train/val split is used to balance classes

---

## 📞 Contact & Support

- **Lab:** AiTA Lab, FPTU
- **Project:** Biosequence Research & Variant Prediction
- **Date:** January 2026

---

**Last Updated:** 2026-01-07
