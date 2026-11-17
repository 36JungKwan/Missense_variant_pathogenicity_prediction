export type ClassificationType = "Pathogenic" | "Benign"

export interface ModelPrediction {
  probability: number
  label: ClassificationType
}

export interface ModelResults {
  rb_net: ModelPrediction
  xgboost?: ModelPrediction
  pytorch?: ModelPrediction
}

export interface SOTAResults {
  alphamissense: number
  revel: number
}

export interface ShapItem {
  feature: string
  name?: string
  contribution_value: number
  description?: string
}

export interface VariantData {
  variant_id: string
  gene?: string
  consequence?: string
  clinvar_ground_truth: ClassificationType
  results: ModelResults
  sota: SOTAResults
  shap_analysis: ShapItem[]
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
  rb_net_probability: number
  classification: ClassificationType
  clinvar_ground_truth: ClassificationType
  sota_alphamissense: number
  danger_rank: number
  isNewStreamItem?: boolean
}

export interface AblationState {
  dnaFeatures: boolean
  proteinFeatures: boolean
  biologicalFeatures: boolean
  spatialGeometry: boolean
}

export interface CachedInference {
  cacheKey: string
  timestamp: number
  computationTimeMs: number
  isCacheHit: boolean
  data: VariantData
}
