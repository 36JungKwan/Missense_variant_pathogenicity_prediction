export type ModalityKey = 'dna' | 'prot' | 'bio' | 'geom';

// 20 đặc trưng Bio hợp nhất (11 BIO_CORE_COLUMNS + 9 SPLICEAI_COLUMNS)
export interface BioFeatures {
  // Conservation & Allele Frequency (11)
  AF: number;
  gnomADe_AF: number;
  phyloP100way_vertebrate: number;
  phyloP470way_mammalian: number;
  phyloP17way_primate: number;
  phastCons100way_vertebrate: number;
  phastCons470way_mammalian: number;
  phastCons17way_primate: number;
  'GERP++_RS': number;
  'GERP++_NR': number;
  GERP_92_mammals: number;

  // SpliceAI Predictions (9)
  SpliceAI_pred_DS_AG: number;
  SpliceAI_pred_DS_AL: number;
  SpliceAI_pred_DS_DG: number;
  SpliceAI_pred_DS_DL: number;
  SpliceAI_pred_DP_AG: number;
  SpliceAI_pred_DP_AL: number;
  SpliceAI_pred_DP_DG: number;
  SpliceAI_pred_DP_DL: number;
  SpliceAI_pred_DS_max: number;
}

// 8 đặc trưng hình học không gian tiềm ẩn (không bao gồm LLR)
export interface GeometryFeatures {
  dna_LVD_L2: number;
  dna_LVD_Cosine: number;
  dna_LID: number;
  dna_LVD_Relative: number;
  prot_LVD_L2: number;
  prot_LVD_Cosine: number;
  prot_LID: number;
  prot_LVD_Relative: number;
}

export interface ShapContribution {
  feature: string;
  feature_group:
    | 'dna_pca_component'
    | 'protein_pca_component'
    | 'learned_fusion_latent'
    | 'geometry'
    | 'allele_frequencies'
    | 'spliceai'
    | 'conservation_scores';
  shap_value_raw_margin: number;
  rank: number;
}

export interface RouterWeights {
  dna_branch: number;
  protein_branch: number;
  tabular_branch: number; // Đại diện cho cụm Bio + Geometry
}

export interface VariantRecord {
  Variant_ID: string;
  chrom: string;
  pos: number;
  ref: string;
  alt: string;
  gene_symbol: string;
  transcript_id: string;
  consequence: string; // e.g., "missense_variant"
  hgvs_c: string;
  hgvs_p: string;
  clinvar_label?: 'Benign' | 'Pathogenic' | 'VUS';
  bio_features: BioFeatures;
  geometry: GeometryFeatures;
  // Xác suất cơ sở khi bật đủ cả 4 nhánh (dùng cho mock simulation)
  base_probability: number;
  decision_threshold: number;
  base_router_weights: RouterWeights;
  shap_explanations: ShapContribution[];
}

export interface PredictionRequest {
  variant_id: string;
  active_modalities: ModalityKey[];
  // Cho phép What-If Perturbation: người dùng chỉnh sửa tay các giá trị Bio/Geom
  custom_bio_features?: Partial<BioFeatures>;
  custom_geometry?: Partial<GeometryFeatures>;
}

export interface PredictionResponse {
  Variant_ID: string;
  gene_symbol: string;
  transcript_id: string;
  consequence: string;
  hgvs_c: string;
  hgvs_p: string;
  clinvar_label?: 'Benign' | 'Pathogenic' | 'VUS';
  Predicted_Probability: number;
  Decision_Threshold: number;
  Prediction_Class: 'Benign' | 'Pathogenic';
  active_modalities: ModalityKey[];
  ablation_key: string; // Chuỗi định danh 1 trong 16 cấu hình (vd: "bio_dna_geom_prot")
  router_weights: RouterWeights;
  bio_features: BioFeatures;
  geometry: GeometryFeatures;
  shap_explanations: ShapContribution[];
  inference_source: 'cache' | 'on_the_fly';
  latency_ms: number;
}

export interface BatchVariantRow {
  Variant_ID: string;
  gene_symbol: string;
  hgvs_p: string;
  consequence: string;
  is_missense: boolean;
  clinvar_label?: 'Benign' | 'Pathogenic' | 'VUS';
  Predicted_Probability: number;
  Decision_Threshold: number;
  Prediction_Class: 'Benign' | 'Pathogenic';
  inference_source: 'cache' | 'on_the_fly';
}