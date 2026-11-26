// Kiểu dữ liệu dùng chung cho toàn bộ Frontend (khớp với schema Pydantic của backend/app/schemas).

// ---------------------------------------------------------------------------
// Nhãn & kết quả mô hình
// ---------------------------------------------------------------------------
export type ClassificationType = "Pathogenic" | "Benign"
/** Nhãn ClinVar: thêm "Uncertain" khi ClinVar là VUS / conflicting / không có bản ghi. */
export type ClinVarLabel = ClassificationType | "Uncertain"

export interface ModelPrediction {
  probability: number
  label: ClassificationType
}

export interface ModelResults {
  trimvpp: ModelPrediction
}

/** Điểm SOTA từ dbNSFP; `null` khi dbNSFP không có điểm cho biến thể (hiển thị "N/A"). */
export interface SOTAResults {
  alphamissense: number | null
  revel: number | null
}

// ---------------------------------------------------------------------------
// Giải thích: Integrated Gradients (full_prediction) + attention map của cross-attention
// ---------------------------------------------------------------------------
export type IgModalityName = "DNA" | "Protein" | "Bio_SpliceAI" | "Geometry"

export interface IgModality {
  modality: IgModalityName
  signed: number
  absolute: number
}

export interface IgToken {
  modality: "DNA" | "Protein"
  allele: "Ref" | "Alt"
  position: number // -16..+16 quanh biến thể
  signed: number
  absolute: number
}

export interface IgFeature {
  feature: string
  name: string
  group: "Bio_SpliceAI" | "Geometry"
  description?: string | null
  raw_value?: number | null
  raw_note?: string | null
  input_value: number // giá trị đã chuẩn hoá đưa vào model
  signed: number
}

/** IG giải thích logit Pathogenic so với baseline train3: tổng đóng góp + residual = logit - baseline_logit. */
export interface IgAnalysis {
  method: string
  mode: string
  baseline: string
  steps: number
  logit: number
  baseline_logit: number
  completeness_residual: number
  relative_residual: number
  residual_over_5pct: boolean
  modalities: IgModality[]
  tokens: IgToken[]
  features: IgFeature[]
}

export type AttentionQuery = "dna" | "protein" | "bio_geometry"

export interface AttentionMap {
  query: AttentionQuery
  query_labels: string[]
  context_segments: { name: AttentionQuery; labels: string[] }[]
  key_labels: string[]
  weights: number[][][] // (heads, query, key); mỗi hàng query cộng lại = 1
}

export interface AttentionMaps {
  num_layers: number
  num_heads: number
  layers: { layer: number; maps: AttentionMap[] }[]
}

export interface VariantData {
  variant_id: string
  gene?: string
  consequence?: string
  clinvar_ground_truth: ClinVarLabel
  results: ModelResults
  sota: SOTAResults
  ig_analysis: IgAnalysis
}

export interface VariantPreset extends VariantData {
  id: string
  name: string
}

export interface BatchVariant {
  id: number
  variant_id: string
  gene: string
  consequence: string
  trimvpp_probability: number
  classification: ClassificationType
  clinvar_ground_truth: ClinVarLabel
  sota_alphamissense: number | null
  danger_rank: number
  isNewStreamItem?: boolean
}

/**
 * 3 nhóm modality khớp đúng 3 bit của checkpoint (use_dna, use_prot, use_biogeom):
 * Bio (11 core + 9 SpliceAI) và Geometry (8 LVD/LID) luôn bật/tắt cùng nhau.
 */
export interface AblationState {
  dnaFeatures: boolean
  proteinFeatures: boolean
  bioGeomFeatures: boolean
}

// ---------------------------------------------------------------------------
// API contract (/api/v1)
// ---------------------------------------------------------------------------
export interface AblationFlags {
  dna_enabled: boolean
  protein_enabled: boolean
  bio_enabled: boolean
  geo_enabled: boolean
}

export interface SingleVariantRequest extends AblationFlags {
  variant_id: string
  use_cache: boolean
}

/** Thời gian từng chặng của pipeline (ms): vep, spliceai, fm_dna, fm_prot, geom, norm, ig, attn, db. */
export type StageTimings = Partial<Record<"vep" | "spliceai" | "fm_dna" | "fm_prot" | "geom" | "norm" | "ig" | "attn" | "db", number>>

export interface InferenceMeta {
  is_cache_hit: boolean
  computation_time_ms: number
  ablation_mask: number
  active_modalities?: string[]
  checkpoint?: string | null
  architecture?: string | null
  decision_threshold?: number | null
  test_mcc?: number | null
  warnings?: string[]
  timings_ms?: StageTimings
}

export interface SingleVariantResponse extends VariantData {
  meta: InferenceMeta
}

/** GET /xai: attention = null khi tổ hợp chỉ có một nhánh (không có cross-attention). */
export interface XaiResponse extends SingleVariantResponse {
  attention: AttentionMaps | null
}

/** Trang mở trang chi tiết / XAI (để nút Back quay về đúng ngữ cảnh). */
export type NavOrigin = "single" | "batch"

export interface BatchStreamEvent {
  type: "variant_stream" | "variant_skipped" | "stream_completed" | "stream_error"
  progress_percent: number
  variant?: BatchVariant
  variant_id?: string
  detected_consequence?: string
  reason?: string
  error_code?: string
  summary?: {
    total_processed: number
    missense_accepted: number
    non_missense_rejected: number
  }
}

/** Dạng lỗi thống nhất của backend: {"detail": {...}}. */
export interface ApiErrorDetail {
  error_code: string
  message: string
  request_id?: string
  [extra: string]: unknown
}

// ---------------------------------------------------------------------------
// Kiểu cho các component trực quan hoá (components/ablation, components/variant)
// ---------------------------------------------------------------------------
/** 20 đặc trưng Bio (11 BIO_CORE_COLUMNS + 9 SPLICEAI_COLUMNS). */
export interface BioFeatures {
  AF: number
  gnomADe_AF: number
  phyloP100way_vertebrate: number
  phyloP470way_mammalian: number
  phyloP17way_primate: number
  phastCons100way_vertebrate: number
  phastCons470way_mammalian: number
  phastCons17way_primate: number
  "GERP++_RS": number
  "GERP++_NR": number
  GERP_92_mammals: number
  SpliceAI_pred_DS_AG: number
  SpliceAI_pred_DS_AL: number
  SpliceAI_pred_DS_DG: number
  SpliceAI_pred_DS_DL: number
  SpliceAI_pred_DP_AG: number
  SpliceAI_pred_DP_AL: number
  SpliceAI_pred_DP_DG: number
  SpliceAI_pred_DP_DL: number
  SpliceAI_pred_DS_max: number
}

/** 8 đặc trưng hình học không gian tiềm ẩn (không gồm LLR). */
export interface GeometryFeatures {
  dna_LVD_L2: number
  dna_LVD_Cosine: number
  dna_LID: number
  dna_LVD_Relative: number
  prot_LVD_L2: number
  prot_LVD_Cosine: number
  prot_LID: number
  prot_LVD_Relative: number
}
