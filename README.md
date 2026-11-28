# TriMVPP: Missense Variant Pathogenicity Prediction

TriMVPP is a research model for predicting whether a human missense variant is **pathogenic** or **benign**. It combines three sources of information: DNA sequence, protein sequence, and tabular biological features. Cross-attention allows these sources to exchange information before the model produces a prediction score.

A missense variant changes an amino acid in a protein. Its consequences depend on factors ranging from sequence context, evolutionary conservation, and population frequency to potential effects on splicing. Many variants remain classified as variants of uncertain significance (VUS) because the available evidence is insufficient to interpret them.

The project goes beyond building a multimodal predictor. It also examines the contribution of each data source, latent geometry, variant representation, and fusion mechanism; assesses the risk of data circularity; and investigates which evidence the trained model relies on.

> **Intended use:** TriMVPP scores are computational evidence for research. They are not an ACMG/AMP variant classification and do not replace clinical assessment.

## Problem Definition

- **Variant-level input:** chromosome, position, reference allele, and alternate allele on the GRCh38 reference assembly. The pipeline also requires the corresponding reference/alternate sequences, biological annotations, and geometry features.
- **Output:** predicted probability of pathogenicity.
- **Labels:** Pathogenic/Likely pathogenic is `1`; Benign/Likely benign is `0`. VUS and variants with conflicting interpretations are excluded from the labels used for this binary classification task.

## The TriMVPP Model

| Source | Representation in the main configuration |
|---|---|
| DNA | [Nucleotide Transformer NTv1-500M](https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-human-ref), with a 1001-nucleotide sequence |
| Protein | [ESM-1b 650M](https://huggingface.co/facebook/esm1b_t33_650M_UR50S), with a 501-amino-acid sequence |
| Tabular features | 28 features: 11 core biological features, 9 SpliceAI features, and 8 geometry features |

Core biological features include allele frequencies and phyloP, phastCons, and GERP conservation scores. Geometry features comprise LID and three LVD measures (`LVD_L2`, `LVD_Cosine`, and `LVD_Relative`) for each of the DNA and protein branches. Geometry is a derived feature group within the tabular branch, **not a fourth modality**. The biological columns are listed in [dataset.py](core/module05v2_fusion_classifier/dataset.py); the selected geometry configuration is specified in the experiment notebooks.

```text
DNA ref/alt     -> NTv1-500M -> variant-centered token window --\
Protein ref/alt -> ESM-1b    -> variant-centered token window ---+-> Cross-attention
Bio + SpliceAI + geometry   -> feature tokens ------------------/         |
                                                                    Pooling + MLP
                                                                         |
                                                               Pathogenicity probability
```

In the main configuration, reference and alternate embeddings are concatenated by relative position; `k_radius=16` corresponds to a window of 33 token positions around the variant. The foundation model backbones are frozen during embedding extraction. The projection, cross-attention, pooling, and classifier components are trained for the task. For implementation details, see [cross_attention_3modal_experiment.py](core/module06_architecture_experiments/cross_attention_3modal_experiment.py) and the [methodology document](docs/scientific_methodology/01_CrossAttention_3modalities.docx).

## Evaluation and Results

The training, validation, and held-out test sets are split by chromosome to reduce leakage between them. Benchmark sets are prepared and checked for overlap separately; chromosome splitting alone does not establish that every source of data circularity has been eliminated. Models and thresholds are selected using validation data, while test results are reported separately.

In the repository's comparison table, TriMVPP is evaluated against 20 existing tools. The MCC values below are the model's mean ± standard deviation across five seeds; rankings are based on MCC in the corresponding table:

| Evaluation set | TriMVPP MCC | Rank |
|---|---:|---:|
| Held-out Test | 0.83722 ± 0.00424 | 1 |
| ClinVar HQ | 0.88108 ± 0.01038 | 1 |
| ProteinGym | 0.68140 ± 0.01276 | 3 |
| UniProt | 0.53486 ± 0.01811 | 9 |

See the [comparison table](docs/ours_vs_sota_performance_by_dataset.csv) and ablation tables in [docs](docs/). **Note:** the result files in `docs/` currently have a `.csv` extension but contain Excel workbooks (XLSX). Open or read them as XLSX rather than assuming they are CSV text. Results depend on the dataset, preparation, and experiment configuration used.

## Repository Contents

```text
.
├── controllers/                          Experiment notebooks and scripts
├── core/
│   ├── module01_data_processing_vep/     Preprocessing, VEP, splits, and sequence mapping
│   ├── module02_fm_embedding_extraction/ Foundation model embedding extraction
│   ├── module03_geometry_extraction/     Latent geometry extraction
│   ├── module04_geo_bio_context_normalization/ Feature normalization and audits
│   ├── module05_fusion_classifier/      Fusion/classifier pipeline and benchmarks
│   ├── module05v2_fusion_classifier/    Version 2 fusion pipeline and experiments
│   └── module06_architecture_experiments/ Architectures, ablations, and interpretability
└── docs/                                 Result tables and methodology documents
```

The modules contain different configurations and experiments; not every notebook uses the final TriMVPP architecture. The team also developed a web lookup system, but this repository does not contain its source code or deployment instructions at the repository root.

### Data Preparation

The notebooks in [module01_data_processing_vep](core/module01_data_processing_vep/) process ClinVar and dbSNP, map VEP annotations, create train/validation/test splits, prepare benchmarks, and generate reference/alternate sequences. Key starting points include:

| File | Purpose |
|---|---|
| [00_clinvar_raw_preprocess.ipynb](core/module01_data_processing_vep/00_clinvar_raw_preprocess.ipynb) | Preprocess ClinVar data, including GRCh38 selection |
| [01_1_clinvar_vep_mapping.ipynb](core/module01_data_processing_vep/01_1_clinvar_vep_mapping.ipynb) | Map VEP annotations for ClinVar |
| [01_2_clinvar_train_val_test_split.ipynb](core/module01_data_processing_vep/01_2_clinvar_train_val_test_split.ipynb) | Create and inspect chromosome-based data splits |
| [02_3_dbsnp_clinvar_train_val_test_split.ipynb](core/module01_data_processing_vep/02_3_dbsnp_clinvar_train_val_test_split.ipynb) | Prepare combined dbSNP/ClinVar training sets and splits |
| [04_01_test_data_process_overlapping.ipynb](core/module01_data_processing_vep/04_01_test_data_process_overlapping.ipynb) | Process benchmark data and check overlaps |
| [05_map_ref_alt_sequence_dna.ipynb](core/module01_data_processing_vep/05_map_ref_alt_sequence_dna.ipynb), [05_map_ref_alt_sequence_protein.ipynb](core/module01_data_processing_vep/05_map_ref_alt_sequence_protein.ipynb) | Generate reference/alternate DNA and protein sequences |

The `03_*` notebooks in the same directory prepare sets including ClinVar HQ, ProteinGym, UniProt, MaveDB, and VUS. See also the [Module 01 documentation](docs/scientific_methodology/Module01_Data_Processing_and_VEP_Documentation.docx).

### Controller Notebooks

| Notebook | Purpose |
|---|---|
| [00_feature_extraction.ipynb](controllers/00_feature_extraction.ipynb) | Extract embeddings and geometry and normalize features |
| [01_train_val_test_batch_runner.ipynb](controllers/01_train_val_test_batch_runner.ipynb) | Run configured training, validation, test, and benchmark pipelines |
| [02_geometry_extraction_v2.ipynb](controllers/02_geometry_extraction_v2.ipynb) | Orchestrate version 2 geometry extraction |
| [11_optimzed_architecture_run.ipynb](controllers/11_optimzed_architecture_run.ipynb) | Run architecture experiments using the notebook configuration |
| [12_full_reproduction_and_ablations.ipynb](controllers/12_full_reproduction_and_ablations.ipynb) | Module 5 reproduction experiments, tool comparisons, and ablations |
| [13_new_architecture_experiments.ipynb](controllers/13_new_architecture_experiments.ipynb) | Explore fusion, Transformer, and cross-attention architectures |
| [14_fm_pairs_ablation.ipynb](controllers/14_fm_pairs_ablation.ipynb) | Step 1: compare foundation model pairs and tune the selected pair |
| [15_sequence_length_ablation.ipynb](controllers/15_sequence_length_ablation.ipynb) | Step 2: compare DNA/protein sequence lengths and generate matched geometry |
| [16_fm_pooling_ablation.ipynb](controllers/16_fm_pooling_ablation.ipynb) | Step 3: compare token windows and pooling strategies |
| [17_fusion_classifier_benchmark.ipynb](controllers/17_fusion_classifier_benchmark.ipynb) | Compare fusion/classifier combinations; reuse Step 2 Cross-Attention checkpoints |
| [18_feature_ablation_studies.ipynb](controllers/18_feature_ablation_studies.ipynb) | Pure ML and feature-group, individual-feature, and modality ablations |
| [19_cross_attention_explainability.ipynb](controllers/19_cross_attention_explainability.ipynb) | Explain checkpoints using AF intervention, Integrated Gradients, occlusion, attention, and SHAP |
| [20_training_dataset_ablation.ipynb](controllers/20_training_dataset_ablation.ipynb) | Compare the effect of training datasets |

The Python scripts in [controllers](controllers/) support geometry/LID experiments, sequence-length studies, seed sweeps, MCC reporting, and a UniProt-augmented pipeline. Review each script's configuration before running it.

## Environment and Execution

The repository currently has no dependency manifest or version-pinned environment. Components use Python/Jupyter and, depending on the notebook, libraries such as PyTorch, Transformers, pandas, NumPy, scikit-learn, PyArrow, FAISS, Optuna, and XGBoost. Some notebooks also require tools or libraries for VEP, biological sequences, and model interpretability. Choose versions compatible with your operating system, GPU, and models; the repository does not provide a verified installation command for the entire pipeline.

To prepare an experiment:

1. Prepare source data, GRCh38/VEP resources, required annotations, reference/alternate sequences, and model weights. Notebooks refer to external Parquet files, caches, checkpoints, and resources; cloning the repository alone does not guarantee that these are available.
2. Open the required notebook in Jupyter or VS Code. Adjust paths such as `BASE_DIR`, batch directories, caches, reference FASTA files, and checkpoints for your environment. Many current values are absolute paths on the team's machine.
3. Check inputs and cell order. The main comparison steps have dependencies: Notebook 14 prepares Step 1 results; Notebook 15 uses the Step 1 configuration for Step 2; Notebooks 16 and 17 use Step 2 results/checkpoints. Notebook 18 uses the Notebook 17 baseline, and Notebook 19 requires trained checkpoints.
4. Run only the required stages. Extraction and training cells can require substantial GPU time and storage. Some cells run Hugging Face in offline mode, so model weights must already be cached.
5. Inspect the output and manifest for your own run to verify its data, seeds, configuration, and completion status. Outputs saved in notebooks may belong to earlier runs and do not necessarily reflect the current code.

The pipeline uses caches, checkpoints, and result directories to resume some experiments. When changing data, seeds, or configuration, check the resume conditions and use a new batch or study when necessary to avoid mixing incompatible results.

## References

- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)
- [Ensembl Variant Effect Predictor](https://www.ensembl.org/info/docs/tools/vep/)
- [Nucleotide Transformer](https://github.com/instadeepai/nucleotide-transformer)
- [ESM](https://github.com/facebookresearch/esm)
- [ProteinGym](https://proteingym.org/)
- [Project methodology documents](docs/scientific_methodology/)
