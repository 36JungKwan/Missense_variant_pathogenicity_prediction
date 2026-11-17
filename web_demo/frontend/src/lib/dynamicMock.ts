import { AblationState, VariantData, CachedInference, ShapItem, ClassificationType } from "@/types"
import mockData from "@/mock/mockData.json"

/**
 * Builds standard Cache Key according to specification:
 * ${variant_id}_DNA:${dna}_PRO:${protein}_BIO:${bio}_GEO:${spatial}
 */
export function buildCacheKey(variantId: string, toggles: AblationState): string {
  const cleanId = variantId.trim().toUpperCase()
  return `${cleanId}_DNA:${toggles.dnaFeatures}_PRO:${toggles.proteinFeatures}_BIO:${toggles.biologicalFeatures}_GEO:${toggles.spatialGeometry}`
}

/**
 * In-memory Cache Map to store inference results by Cache Key
 */
const inferenceCache = new Map<string, CachedInference>()

/**
 * Retrieves cached inference if available
 */
export function getCachedInference(key: string): CachedInference | undefined {
  return inferenceCache.get(key)
}

/**
 * Saves inference result to in-memory cache
 */
export function setCachedInference(
  key: string,
  data: VariantData,
  computationTimeMs: number
): CachedInference {
  const entry: CachedInference = {
    cacheKey: key,
    timestamp: Date.now(),
    computationTimeMs,
    isCacheHit: false,
    data,
  }
  inferenceCache.set(key, entry)
  return entry
}

/**
 * Clears the inference cache
 */
export function clearInferenceCache(): void {
  inferenceCache.clear()
}

/**
 * Gets count of cached items
 */
export function getCacheSize(): number {
  return inferenceCache.size
}

/**
 * Dynamic Mock Generator:
 * Computes RB-Net model probabilities and SHAP feature attributions dynamically
 * based on the 4 Ablation Toggles state.
 *
 * Requirements:
 * - If Bio is false: SpliceAI and Allele Frequency contributions are suppressed to 0.00
 * - If DNA is false: Confidence drops without k-mer sequence representations
 * - If Protein is false: Physicochemical features are removed
 * - If Spatial Geometry (GEO) is false: LVD Cosine & spatial structural features are suppressed to 0.00 (-20% confidence)
 */
export function generateDynamicMockResult(
  variantId: string,
  toggles: AblationState,
  basePresetId = "pathogenic"
): VariantData {
  // Find baseline preset or default to first preset
  const preset =
    mockData.presets.find((p) => p.id === basePresetId) ||
    mockData.presets.find((p) => p.variant_id.toLowerCase() === variantId.toLowerCase()) ||
    mockData.presets[0]

  const isPathogenicBase = preset.results.rb_net.label === "Pathogenic"

  // Base RB-Net probability
  let rbNetProb = preset.results.rb_net.probability

  // 1. DNA Ablation Effect:
  if (!toggles.dnaFeatures) {
    if (isPathogenicBase) {
      rbNetProb = Math.max(0.25, Number((rbNetProb - 0.08).toFixed(3)))
    } else {
      rbNetProb = Math.min(0.75, Number((rbNetProb + 0.08).toFixed(3)))
    }
  }

  // 2. Protein Ablation Effect:
  if (!toggles.proteinFeatures) {
    if (isPathogenicBase) {
      rbNetProb = Math.max(0.25, Number((rbNetProb - 0.10).toFixed(3)))
    } else {
      rbNetProb = Math.min(0.75, Number((rbNetProb + 0.09).toFixed(3)))
    }
  }

  // 3. Biological Ablation Effect:
  if (!toggles.biologicalFeatures) {
    if (isPathogenicBase) {
      rbNetProb = Math.max(0.25, Number((rbNetProb - 0.14).toFixed(3)))
    } else {
      rbNetProb = Math.min(0.75, Number((rbNetProb + 0.11).toFixed(3)))
    }
  }

  // 4. Spatial Geometry (GEO) Ablation Effect:
  // Core proposed contribution of RB-Net: without spatial geometry LVD representations, confidence drops (-20%)
  if (!toggles.spatialGeometry) {
    if (isPathogenicBase) {
      rbNetProb = Math.max(0.3, Number((rbNetProb - 0.20).toFixed(3)))
    } else {
      rbNetProb = Math.min(0.7, Number((rbNetProb + 0.18).toFixed(3)))
    }
  }

  // Build dynamic SHAP attributions based on toggle states
  const dynamicShap: ShapItem[] = []

  preset.shap_analysis.forEach((item) => {
    let contrib = item.contribution_value
    let desc = item.description || ""

    // Biological features toggle: SpliceAI, Allele Frequency, PhyloP
    if (!toggles.biologicalFeatures) {
      if (item.feature.includes("SpliceAI") || item.feature.includes("Allele") || item.feature.includes("PhyloP")) {
        contrib = 0.00
        desc = "[Ablation: Bio OFF] Biological annotations and splicing features suppressed"
      }
    }

    // Spatial Geometry toggle: LVD_Cosine, Spatial_Structural_Perturbation
    if (!toggles.spatialGeometry) {
      if (item.feature.includes("LVD") || item.feature.includes("Spatial")) {
        contrib = 0.00
        desc = "[Ablation: Spatial Geometry OFF] Spatial coordinate representations suppressed"
      }
    }

    // DNA features toggle
    if (!toggles.dnaFeatures) {
      if (item.feature.includes("DNA") || item.feature.includes("k-mer")) {
        contrib = 0.00
        desc = "[Ablation: DNA OFF] Nucleotide sequence context removed"
      }
    }

    // Protein features toggle
    if (!toggles.proteinFeatures) {
      if (item.feature.includes("Protein") && !item.feature.includes("Spatial")) {
        contrib = 0.00
        desc = "[Ablation: Protein OFF] Amino acid physicochemical properties removed"
      }
    }

    dynamicShap.push({
      feature: item.feature,
      name: item.name || item.feature,
      contribution_value: Number(contrib.toFixed(3)),
      description: desc,
    })
  })

  // Strict Binary label (Pathogenic | Benign)
  const rbNetLabel: ClassificationType = rbNetProb >= 0.5 ? "Pathogenic" : "Benign"

  return {
    variant_id: variantId || preset.variant_id,
    gene: preset.gene,
    consequence: preset.consequence,
    clinvar_ground_truth: preset.clinvar_ground_truth as ClassificationType,
    results: {
      rb_net: { probability: Number(rbNetProb.toFixed(3)), label: rbNetLabel },
    },
    sota: preset.sota,
    shap_analysis: dynamicShap,
  }
}

/**
 * Pre-populate initial default states into cache for instant startup
 */
export function initializeDefaultCache(): void {
  const allOn: AblationState = {
    dnaFeatures: true,
    proteinFeatures: true,
    biologicalFeatures: true,
    spatialGeometry: true,
  }

  mockData.presets.forEach((preset) => {
    const key = buildCacheKey(preset.variant_id, allOn)
    const data = generateDynamicMockResult(preset.variant_id, allOn, preset.id)
    setCachedInference(key, data, 0)
  })
}

// Auto initialize on module load
initializeDefaultCache()
