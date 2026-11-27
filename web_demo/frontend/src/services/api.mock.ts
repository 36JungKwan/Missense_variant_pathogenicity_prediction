// Giả lập backend cho chế độ demo offline (VITE_USE_MOCK_API="true"). Không dùng khi chạy với FastAPI thật.
import mockData from "@/mock/mockData.json"
import { computeAblationMask, decodeMaskToFlags } from "@/lib/ablation"
import type {
  AblationFlags,
  AblationState,
  AttentionMap,
  AttentionMaps,
  AttentionQuery,
  BatchStreamEvent,
  BatchVariant,
  ClassificationType,
  IgAnalysis,
  IgFeature,
  IgModality,
  IgToken,
  SingleVariantRequest,
  SingleVariantResponse,
  VariantData,
  XaiResponse,
} from "@/types"

type BaseVariant = Omit<VariantData, "ig_analysis">

const memoryCache = new Map<string, VariantData>()
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

function resolveBaseVariant(variantId: string): BaseVariant {
  const normalized = variantId.trim().toLowerCase()

  const preset = mockData.presets.find((p) => p.variant_id.toLowerCase() === normalized)
  if (preset) {
    return {
      variant_id: preset.variant_id,
      gene: preset.gene,
      consequence: preset.consequence,
      clinvar_ground_truth: preset.clinvar_ground_truth as ClassificationType,
      results: {
        trimvpp: {
          probability: preset.results.trimvpp.probability,
          label: preset.results.trimvpp.label as ClassificationType,
        },
      },
      sota: preset.sota,
    }
  }

  const batch = mockData.batch_variants.find((b) => b.variant_id.toLowerCase() === normalized)
  if (batch) {
    return {
      variant_id: batch.variant_id,
      gene: batch.gene,
      consequence: batch.consequence,
      clinvar_ground_truth: batch.clinvar_ground_truth as ClassificationType,
      results: {
        trimvpp: { probability: batch.trimvpp_probability, label: batch.classification as ClassificationType },
      },
      sota: {
        alphamissense: batch.sota_alphamissense,
        revel: Number(Math.max(0.05, batch.sota_alphamissense - 0.06).toFixed(2)),
      },
    }
  }

  return { ...(mockData.single_variant as BaseVariant), variant_id: variantId.trim() }
}

// --- IG / attention giả lập (chỉ để trình diễn giao diện offline; không phải kết quả của model) ---
const K_RADIUS = 16
const BIO_NAMES = [
  "AF", "gnomADe_AF", "phyloP100way_vertebrate", "phyloP470way_mammalian", "phyloP17way_primate",
  "phastCons100way_vertebrate", "phastCons470way_mammalian", "phastCons17way_primate", "GERP++_RS", "GERP++_NR",
  "GERP_92_mammals", "SpliceAI_pred_DS_AG", "SpliceAI_pred_DS_AL", "SpliceAI_pred_DS_DG", "SpliceAI_pred_DS_DL",
  "SpliceAI_pred_DP_AG", "SpliceAI_pred_DP_AL", "SpliceAI_pred_DP_DG", "SpliceAI_pred_DP_DL", "SpliceAI_pred_DS_max",
]
const GEOM_NAMES = ["dna_LVD_L2", "dna_LVD_Cosine", "dna_LID", "dna_LVD_Relative",
  "prot_LVD_L2", "prot_LVD_Cosine", "prot_LID", "prot_LVD_Relative"]
const POSITIONS = Array.from({ length: 2 * K_RADIUS + 1 }, (_, i) => i - K_RADIUS)
const tokenLabels = (prefix: string) => POSITIONS.map((p) => `${prefix}:${p >= 0 ? "+" : ""}${p}`)

function seededRandom(seedText: string): () => number {
  let h = 2166136261
  for (const ch of seedText) h = Math.imul(h ^ ch.charCodeAt(0), 16777619)
  return () => {
    h = Math.imul(h ^ (h >>> 15), 2246822507) ^ Math.imul(h ^ (h >>> 13), 3266489909)
    h >>>= 0
    return (h % 100000) / 100000
  }
}

/** Chia `total` thành n phần (tổng đúng bằng total), có vài phần ngược dấu để biểu đồ có hai phía. */
function split(total: number, weights: number[], rand: () => number): number[] {
  const noisy = weights.map((w) => w * (0.4 + rand()))
  const sum = noisy.reduce((a, b) => a + b, 0) || 1
  const parts = noisy.map((w, i) => ((i % 5 === 3 ? -0.4 : 1) * total * w) / sum)
  parts[0] += total - parts.reduce((a, b) => a + b, 0)
  return parts
}

function mockIg(variantId: string, probability: number, flags: AblationFlags): IgAnalysis {
  const rand = seededRandom(`${variantId}|${computeAblationMask(flags)}`)
  const p = Math.min(0.999, Math.max(0.001, probability))
  const logit = Math.log(p / (1 - p))
  const baselineLogit = -0.35
  const active: IgModality["modality"][] = []
  if (flags.dna_enabled) active.push("DNA")
  if (flags.protein_enabled) active.push("Protein")
  if (flags.bio_enabled || flags.geo_enabled) active.push("Bio_SpliceAI", "Geometry")
  const shares = split(logit - baselineLogit, active.map((m) => (m === "Geometry" ? 0.6 : 1)), rand)
  const modalities: IgModality[] = active.map((modality, i) => ({ modality, signed: shares[i], absolute: Math.abs(shares[i]) * 1.6 }))
  const tokens: IgToken[] = []
  const features: IgFeature[] = []
  for (const m of modalities) {
    if (m.modality === "DNA" || m.modality === "Protein") {
      const weights = [0.6, 1.4].flatMap((w) => POSITIONS.map((pos) => w / (1 + Math.abs(pos) / 2)))
      const values = split(m.signed, weights, rand)
      const alleles = ["Ref", "Alt"] as const
      alleles.forEach((allele, a) => POSITIONS.forEach((position, i) => {
        const v = values[a * POSITIONS.length + i]
        tokens.push({ modality: m.modality as IgToken["modality"], allele, position, signed: v, absolute: Math.abs(v) * 1.5 })
      }))
    } else {
      const names = m.modality === "Bio_SpliceAI" ? BIO_NAMES : GEOM_NAMES
      const values = split(m.signed, names.map(() => rand() ** 2 + 0.05), rand)
      names.forEach((feature, i) => features.push({
        feature, name: feature.replace(/_/g, " "), group: m.modality as IgFeature["group"],
        description: "Offline demo value", raw_value: null, raw_note: "mock", input_value: rand() * 2 - 1, signed: values[i],
      }))
    }
  }
  features.sort((a, b) => Math.abs(b.signed) - Math.abs(a.signed))
  return {
    method: "integrated_gradients", mode: "full_prediction", baseline: "mock", steps: 64,
    logit, baseline_logit: baselineLogit, completeness_residual: 0, relative_residual: 0, residual_over_5pct: false,
    modalities, tokens, features,
  }
}

function mockAttention(variantId: string, flags: AblationFlags): AttentionMaps | null {
  const branches: AttentionQuery[] = []
  if (flags.dna_enabled) branches.push("dna")
  if (flags.protein_enabled) branches.push("protein")
  if (flags.bio_enabled || flags.geo_enabled) branches.push("bio_geometry")
  if (branches.length < 2) return null
  const rand = seededRandom(`attn|${variantId}`)
  const labels: Record<AttentionQuery, string[]> = {
    dna: tokenLabels("DNA"), protein: tokenLabels("Protein"), bio_geometry: [...BIO_NAMES, ...GEOM_NAMES],
  }
  const heads = 8
  const layers = [1, 2, 3].map((layer) => ({
    layer,
    maps: branches.map((query): AttentionMap => {
      const contexts = branches.filter((b) => b !== query)
      const keyLabels = contexts.flatMap((c) => labels[c])
      const weights = Array.from({ length: heads }, () => labels[query].map(() => {
        const scores = keyLabels.map((label) => Math.exp((label.endsWith(":+0") ? 2.5 * layer / 3 : 0) + 2.2 * rand()))
        const sum = scores.reduce((a, b) => a + b, 0)
        return scores.map((s) => s / sum)
      }))
      return {
        query, query_labels: labels[query],
        context_segments: contexts.map((name) => ({ name, labels: labels[name] })),
        key_labels: keyLabels, weights,
      }
    }),
  }))
  return { num_layers: 3, num_heads: heads, layers }
}

/** Mô phỏng ảnh hưởng của việc tắt từng nhánh lên xác suất + IG giả lập. */
function simulateAblation(base: BaseVariant, flags: AblationFlags): VariantData & { ig_analysis: IgAnalysis } {
  let prob = base.results.trimvpp.probability
  const dir = prob >= 0.5 ? 1 : -1
  if (!flags.protein_enabled) prob -= dir * 0.14
  if (!flags.bio_enabled) prob -= dir * 0.12
  if (!flags.geo_enabled) prob -= dir * 0.09
  if (!flags.dna_enabled) prob -= dir * 0.06

  const probability = Number(Math.max(0.01, Math.min(0.99, prob)).toFixed(4))
  const label: ClassificationType = probability >= 0.5 ? "Pathogenic" : "Benign"
  return {
    ...base,
    results: { ...base.results, trimvpp: { probability, label } },
    ig_analysis: mockIg(base.variant_id, probability, flags),
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

export async function mockFetchVariantXai(variantId: string, mask: number): Promise<XaiResponse> {
  await sleep(220)
  const flags = decodeMaskToFlags(mask)
  return {
    ...simulateAblation(resolveBaseVariant(variantId), flags),
    attention: mockAttention(variantId, flags),
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
      item: { id: 9, variant_id: "chr13:32332592:A>C", gene: "BRCA2", consequence: "p.Asn372His",
        trimvpp_probability: 0.08, classification: "Benign", clinvar_ground_truth: "Benign",
        sota_alphamissense: 0.12, danger_rank: 9 },
    },
    { kind: "skip", reason: "intron_variant (filtered by Gatekeeper)" },
    {
      kind: "valid",
      item: { id: 10, variant_id: "chr19:11216264:G>A", gene: "LDLR", consequence: "p.Cys167Tyr",
        trimvpp_probability: 0.87, classification: "Pathogenic", clinvar_ground_truth: "Pathogenic",
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
    let prob = current.item.trimvpp_probability
    const dir = prob >= 0.5 ? 1 : -1
    if (!ablation.proteinFeatures) prob -= dir * 0.12
    if (!ablation.bioGeomFeatures) prob -= dir * 0.18
    if (!ablation.dnaFeatures) prob -= dir * 0.05
    const probability = Number(Math.max(0.01, Math.min(0.99, prob)).toFixed(3))
    onEvent({
      type: "variant_stream",
      progress_percent: progress,
      variant: { ...current.item, trimvpp_probability: probability, classification: probability >= 0.5 ? "Pathogenic" : "Benign" },
    })
  }, intervalMs)

  return { cancel: () => window.clearInterval(timer) }
}
