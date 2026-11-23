// Giả lập backend cho chế độ demo offline (VITE_USE_MOCK_API="true"). Không dùng khi chạy với FastAPI thật.
import mockData from "@/mock/mockData.json"
import { computeAblationMask, decodeMaskToFlags } from "@/lib/ablation"
import type {
  AblationFlags,
  AblationState,
  BatchStreamEvent,
  BatchVariant,
  ClassificationType,
  SingleVariantRequest,
  SingleVariantResponse,
  VariantData,
} from "@/types"

const memoryCache = new Map<string, VariantData>()
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

function resolveBaseVariant(variantId: string): VariantData {
  const normalized = variantId.trim().toLowerCase()

  const preset = mockData.presets.find((p) => p.variant_id.toLowerCase() === normalized)
  if (preset) {
    return {
      variant_id: preset.variant_id,
      gene: preset.gene,
      consequence: preset.consequence,
      clinvar_ground_truth: preset.clinvar_ground_truth as ClassificationType,
      results: {
        rb_net: {
          probability: preset.results.rb_net.probability,
          label: preset.results.rb_net.label as ClassificationType,
        },
      },
      sota: preset.sota,
      shap_analysis: preset.shap_analysis,
    }
  }

  const batch = mockData.batch_variants.find((b) => b.variant_id.toLowerCase() === normalized)
  if (batch) {
    const sign = batch.classification === "Pathogenic" ? 1 : -1
    return {
      variant_id: batch.variant_id,
      gene: batch.gene,
      consequence: batch.consequence,
      clinvar_ground_truth: batch.clinvar_ground_truth as ClassificationType,
      results: {
        rb_net: { probability: batch.rb_net_probability, label: batch.classification as ClassificationType },
      },
      sota: {
        alphamissense: batch.sota_alphamissense,
        revel: Number(Math.max(0.05, batch.sota_alphamissense - 0.06).toFixed(2)),
      },
      shap_analysis: [
        { feature: "LVD_Cosine", name: "Spatial Geometry (LVD)", contribution_value: sign > 0 ? 0.32 : -0.24,
          description: "Spatial geometry vector divergence across local functional neighborhood" },
        { feature: "SpliceAI_Delta", name: "SpliceAI Delta Score", contribution_value: sign > 0 ? 0.25 : -0.18,
          description: "Predicted splice donor/acceptor disruption score" },
        { feature: "Spatial_Structural_Perturbation", name: "Spatial Structural Perturbation",
          contribution_value: sign > 0 ? 0.21 : -0.16,
          description: "Local conformational perturbation in residue coordination space" },
        { feature: "PhyloP_Conservation", name: "PhyloP Conservation Score", contribution_value: sign > 0 ? 0.15 : -0.11,
          description: "Evolutionary nucleotide conservation across vertebrate species" },
      ],
    }
  }

  return { ...(mockData.single_variant as VariantData), variant_id: variantId.trim() }
}

/** Mô phỏng ảnh hưởng của việc tắt từng nhánh lên xác suất + danh sách SHAP. */
function simulateAblation(base: VariantData, flags: AblationFlags): VariantData {
  let prob = base.results.rb_net.probability
  const dir = prob >= 0.5 ? 1 : -1
  if (!flags.protein_enabled) prob -= dir * 0.14
  if (!flags.bio_enabled) prob -= dir * 0.12
  if (!flags.geo_enabled) prob -= dir * 0.09
  if (!flags.dna_enabled) prob -= dir * 0.06

  const probability = Number(Math.max(0.01, Math.min(0.99, prob)).toFixed(4))
  const label: ClassificationType = probability >= 0.5 ? "Pathogenic" : "Benign"
  const shap = base.shap_analysis.filter((item) => {
    const f = item.feature.toLowerCase()
    if (!flags.geo_enabled && (f.includes("lvd") || f.includes("spatial"))) return false
    if (!flags.bio_enabled && (f.includes("splice") || f.includes("phylop") || f.includes("allele"))) return false
    return true
  })
  return {
    ...base,
    results: { ...base.results, rb_net: { probability, label } },
    shap_analysis: shap.length > 0 ? shap : base.shap_analysis,
  }
}

export async function mockPredictSingleVariant(req: SingleVariantRequest): Promise<SingleVariantResponse> {
  const mask = computeAblationMask(req)
  const cacheKey = `${req.variant_id.trim().toLowerCase()}::mask_${mask}`

  if (req.use_cache && memoryCache.has(cacheKey)) {
    const start = performance.now()
    await sleep(15)
    return {
      ...memoryCache.get(cacheKey)!,
      meta: { is_cache_hit: true, computation_time_ms: Math.max(4, Math.round(performance.now() - start)), ablation_mask: mask, decision_threshold: 0.5 },
    }
  }

  const computeMs = Math.floor(Math.random() * 140) + 160
  await sleep(computeMs)
  const evaluated = simulateAblation(resolveBaseVariant(req.variant_id), req)
  if (req.use_cache) memoryCache.set(cacheKey, evaluated)
  return { ...evaluated, meta: { is_cache_hit: false, computation_time_ms: computeMs, ablation_mask: mask, decision_threshold: 0.5 } }
}

export async function mockFetchVariantXai(variantId: string, mask: number): Promise<SingleVariantResponse> {
  await sleep(220)
  return {
    ...simulateAblation(resolveBaseVariant(variantId), decodeMaskToFlags(mask)),
    meta: { is_cache_hit: true, computation_time_ms: 220, ablation_mask: mask, decision_threshold: 0.5 },
  }
}

export function mockVcfStream(
  ablation: AblationState,
  onEvent: (event: BatchStreamEvent) => void,
  intervalMs: number
): { cancel: () => void } {
  const pool = mockData.batch_variants as BatchVariant[]
  const events: Array<{ kind: "valid"; item: BatchVariant } | { kind: "skip"; reason: string }> = [
    ...pool.map((item) => ({ kind: "valid" as const, item })),
    { kind: "skip", reason: "synonymous_variant (filtered by Gatekeeper)" },
    {
      kind: "valid",
      item: { id: 9, variant_id: "chr13:32906729:A>C", gene: "BRCA2", consequence: "p.Asn372His",
        rb_net_probability: 0.08, classification: "Benign", clinvar_ground_truth: "Benign",
        sota_alphamissense: 0.12, danger_rank: 9 },
    },
    { kind: "skip", reason: "intron_variant (filtered by Gatekeeper)" },
    {
      kind: "valid",
      item: { id: 10, variant_id: "chr19:11216264:G>A", gene: "LDLR", consequence: "p.Cys167Tyr",
        rb_net_probability: 0.87, classification: "Pathogenic", clinvar_ground_truth: "Pathogenic",
        sota_alphamissense: 0.83, danger_rank: 10 },
    },
  ]

  let cursor = 0
  const timer = window.setInterval(() => {
    if (cursor >= events.length) {
      window.clearInterval(timer)
      onEvent({ type: "stream_completed", progress_percent: 100 })
      return
    }
    const current = events[cursor]
    cursor += 1
    const progress = Math.min(99, Math.round((cursor / events.length) * 100))

    if (current.kind === "skip") {
      onEvent({ type: "variant_skipped", progress_percent: progress, reason: current.reason })
      return
    }
    let prob = current.item.rb_net_probability
    const dir = prob >= 0.5 ? 1 : -1
    if (!ablation.proteinFeatures) prob -= dir * 0.12
    if (!ablation.bioGeomFeatures) prob -= dir * 0.18
    if (!ablation.dnaFeatures) prob -= dir * 0.05
    const probability = Number(Math.max(0.01, Math.min(0.99, prob)).toFixed(3))
    onEvent({
      type: "variant_stream",
      progress_percent: progress,
      variant: { ...current.item, rb_net_probability: probability, classification: probability >= 0.5 ? "Pathogenic" : "Benign" },
    })
  }, intervalMs)

  return { cancel: () => window.clearInterval(timer) }
}
