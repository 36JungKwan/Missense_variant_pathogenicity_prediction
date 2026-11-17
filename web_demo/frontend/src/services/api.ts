import mockData from "@/mock/mockData.json";
import {
  AblationState,
  BatchVariant,
  ClassificationType,
  ModelResults,
  ShapItem,
  SOTAResults,
  VariantData,
} from "@/types";

// Mặc định false để gọi trực tiếp Backend FastAPI thật tại cổng 8000
const USE_MOCK_API = import.meta.env.VITE_USE_MOCK_API === "true";
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";

// ==========================================
// 1. SINGLE VARIANT INFERENCE (Single.tsx)
// ==========================================

export interface SingleVariantRequest {
  variant_id: string;
  dna_enabled: boolean;
  protein_enabled: boolean;
  bio_enabled: boolean;
  geo_enabled: boolean;
  use_cache: boolean;
}

export interface SingleVariantResponse {
  variant_id: string;
  gene?: string;
  consequence?: string;
  clinvar_ground_truth: ClassificationType;
  results: ModelResults;
  sota: SOTAResults;
  shap_analysis: ShapItem[];
  meta: {
    is_cache_hit: boolean;
    computation_time_ms: number;
    ablation_mask: number;
  };
}

function extractApiErrorMessage(errBody: any, fallback: string): string {
  if (!errBody) return fallback;
  if (typeof errBody.detail === "string") return errBody.detail;
  if (typeof errBody.detail === "object" && errBody.detail.message) {
    return errBody.detail.message;
  }
  return fallback;
}

const memoryCache = new Map<string, VariantData>();

function computeAblationMask(req: {
  dna_enabled: boolean;
  protein_enabled: boolean;
  bio_enabled: boolean;
  geo_enabled: boolean;
}): number {
  return (
    (req.dna_enabled ? 8 : 0) |
    (req.protein_enabled ? 4 : 0) |
    (req.bio_enabled ? 2 : 0) |
    (req.geo_enabled ? 1 : 0)
  );
}

function decodeMaskToFlags(mask: number) {
  return {
    dna_enabled: (mask & 8) !== 0,
    protein_enabled: (mask & 4) !== 0,
    bio_enabled: (mask & 2) !== 0,
    geo_enabled: (mask & 1) !== 0,
  };
}

function resolveBaseVariant(variantId: string): VariantData {
  const normalized = variantId.trim().toLowerCase();

  const matchedPreset = mockData.presets.find(
    (p) => p.variant_id.toLowerCase() === normalized
  );
  if (matchedPreset) {
    return {
      variant_id: matchedPreset.variant_id,
      gene: matchedPreset.gene,
      consequence: matchedPreset.consequence,
      clinvar_ground_truth: matchedPreset.clinvar_ground_truth as ClassificationType,
      results: {
        rb_net: {
          probability: matchedPreset.results.rb_net.probability,
          label: matchedPreset.results.rb_net.label as ClassificationType,
        },
      },
      sota: matchedPreset.sota,
      shap_analysis: matchedPreset.shap_analysis,
    };
  }

  const matchedBatch = mockData.batch_variants.find(
    (b) => b.variant_id.toLowerCase() === normalized
  );
  if (matchedBatch) {
    const isPathogenic = matchedBatch.classification === "Pathogenic";
    return {
      variant_id: matchedBatch.variant_id,
      gene: matchedBatch.gene,
      consequence: matchedBatch.consequence,
      clinvar_ground_truth: matchedBatch.clinvar_ground_truth as ClassificationType,
      results: {
        rb_net: {
          probability: matchedBatch.rb_net_probability,
          label: matchedBatch.classification as ClassificationType,
        },
      },
      sota: {
        alphamissense: matchedBatch.sota_alphamissense,
        revel: Number(Math.max(0.05, matchedBatch.sota_alphamissense - 0.06).toFixed(2)),
      },
      shap_analysis: [
        {
          feature: "LVD_Cosine",
          name: "Spatial Geometry (LVD)",
          contribution_value: isPathogenic ? 0.32 : -0.24,
          description: "Spatial geometry vector divergence across local functional neighborhood",
        },
        {
          feature: "SpliceAI_Delta",
          name: "SpliceAI Delta Score",
          contribution_value: isPathogenic ? 0.25 : -0.18,
          description: "Predicted splice donor/acceptor disruption score",
        },
        {
          feature: "Spatial_Structural_Perturbation",
          name: "Spatial Structural Perturbation",
          contribution_value: isPathogenic ? 0.21 : -0.16,
          description: "Local conformational perturbation in residue coordination space",
        },
        {
          feature: "PhyloP_Conservation",
          name: "PhyloP Conservation Score",
          contribution_value: isPathogenic ? 0.15 : -0.11,
          description: "Evolutionary nucleotide conservation across vertebrate species",
        },
      ],
    };
  }

  return {
    ...(mockData.single_variant as VariantData),
    variant_id: variantId.trim(),
  };
}

function simulateAblationResponse(
  baseVariant: VariantData,
  flags: {
    dna_enabled: boolean;
    protein_enabled: boolean;
    bio_enabled: boolean;
    geo_enabled: boolean;
  }
): VariantData {
  let prob = baseVariant.results.rb_net.probability;
  const dir = prob >= 0.5 ? 1 : -1;

  if (!flags.protein_enabled) prob -= dir * 0.14;
  if (!flags.bio_enabled) prob -= dir * 0.12;
  if (!flags.geo_enabled) prob -= dir * 0.09;
  if (!flags.dna_enabled) prob -= dir * 0.06;

  const clampedProb = Number(Math.max(0.01, Math.min(0.99, prob)).toFixed(4));
  const label: ClassificationType = clampedProb >= 0.5 ? "Pathogenic" : "Benign";

  const filteredShap = baseVariant.shap_analysis.filter((item) => {
    const f = item.feature.toLowerCase();
    if (!flags.geo_enabled && (f.includes("lvd") || f.includes("spatial"))) return false;
    if (!flags.bio_enabled && (f.includes("splice") || f.includes("phylop") || f.includes("allele")))
      return false;
    return true;
  });

  return {
    ...baseVariant,
    results: {
      ...baseVariant.results,
      rb_net: {
        probability: clampedProb,
        label,
      },
    },
    shap_analysis: filteredShap.length > 0 ? filteredShap : baseVariant.shap_analysis,
  };
}

export async function predictSingleVariant(
  req: SingleVariantRequest
): Promise<SingleVariantResponse> {
  const mask = computeAblationMask(req);

  if (mask === 0) {
    throw new Error(
      "At least one feature module (DNA, Protein, Biological, or Spatial Geometry) must be enabled."
    );
  }

  if (!USE_MOCK_API) {
    const res = await fetch(`${API_BASE_URL}/predict/single`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => null);
      throw new Error(extractApiErrorMessage(errBody, "FastAPI inference error"));
    }
    return res.json();
  }

  const normalizedId = req.variant_id.trim();
  const cacheKey = `${normalizedId.toLowerCase()}::mask_${mask}`;

  if (req.use_cache && memoryCache.has(cacheKey)) {
    const start = performance.now();
    await new Promise((r) => setTimeout(r, 15));
    const cached = memoryCache.get(cacheKey)!;
    return {
      ...cached,
      meta: {
        is_cache_hit: true,
        computation_time_ms: Math.max(4, Math.round(performance.now() - start)),
        ablation_mask: mask,
      },
    };
  }

  const simulatedComputeMs = Math.floor(Math.random() * 140) + 160;
  await new Promise((r) => setTimeout(r, simulatedComputeMs));

  const baseData = resolveBaseVariant(normalizedId);
  const evaluated = simulateAblationResponse(baseData, req);

  if (req.use_cache) {
    memoryCache.set(cacheKey, evaluated);
  }

  return {
    ...evaluated,
    meta: {
      is_cache_hit: false,
      computation_time_ms: simulatedComputeMs,
      ablation_mask: mask,
    },
  };
}

// ==========================================
// 2. XAI SHAP (GET /api/v1/xai/{variant_id}?ablation_mask=...)
// ==========================================

export async function fetchVariantXai(
  variantId: string,
  mask: number
): Promise<VariantData> {
  if (mask <= 0 || mask > 15) {
    throw new Error(
      "Invalid ablation mask (0/15). At least one modality must remain active to compute SHAP values."
    );
  }

  if (!USE_MOCK_API) {
    const res = await fetch(
      `${API_BASE_URL}/xai/${encodeURIComponent(variantId)}?ablation_mask=${mask}`
    );
    if (!res.ok) {
      const errBody = await res.json().catch(() => null);
      throw new Error(extractApiErrorMessage(errBody, "Failed to fetch SHAP values"));
    }
    return res.json();
  }

  await new Promise((r) => setTimeout(r, 220));
  const flags = decodeMaskToFlags(mask);
  const baseData = resolveBaseVariant(variantId);
  return simulateAblationResponse(baseData, flags);
}

// ==========================================
// 3. BATCH STREAM & UPLOAD (Batch.tsx)
// ==========================================

export interface BatchStreamEvent {
  type: "variant_stream" | "variant_skipped" | "stream_completed" | "stream_error";
  progress_percent: number;
  variant?: BatchVariant;
  variant_id?: string;
  detected_consequence?: string;
  reason?: string;
  summary?: {
    total_processed: number;
    missense_accepted: number;
    non_missense_rejected: number;
  };
}

export function createVcfStreamService(
  fileOrName: File | string | null,
  ablation: AblationState,
  onEvent: (event: BatchStreamEvent) => void,
  intervalMs = 150
): { cancel: () => void } {
  const mask = computeAblationMask({
    dna_enabled: ablation.dnaFeatures,
    protein_enabled: ablation.proteinFeatures,
    bio_enabled: ablation.biologicalFeatures,
    geo_enabled: ablation.spatialGeometry,
  });

  if (mask === 0) {
    setTimeout(() => {
      onEvent({
        type: "stream_error",
        progress_percent: 0,
        reason: "Cannot run screening with 0 active feature modules.",
      });
    }, 50);
    return { cancel: () => {} };
  }

  if (!USE_MOCK_API) {
    // Trường hợp 1: Người dùng có upload file .vcf / .csv / .txt thật -> gọi POST /predict/batch-upload
    if (fileOrName instanceof File) {
      const abortController = new AbortController();
      const formData = new FormData();
      formData.append("file", fileOrName);
      formData.append("stream_delay_ms", String(intervalMs));
      formData.append("dna_enabled", String(ablation.dnaFeatures));
      formData.append("protein_enabled", String(ablation.proteinFeatures));
      formData.append("bio_enabled", String(ablation.biologicalFeatures));
      formData.append("geo_enabled", String(ablation.spatialGeometry));

      (async () => {
        try {
          const response = await fetch(`${API_BASE_URL}/predict/batch-upload`, {
            method: "POST",
            body: formData,
            signal: abortController.signal,
          });

          if (!response.ok || !response.body) {
            const errBody = await response.json().catch(() => null);
            onEvent({
              type: "stream_error",
              progress_percent: 100,
              reason: extractApiErrorMessage(errBody, "Failed to process uploaded VCF file."),
            });
            return;
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder("utf-8");
          let buffer = "";

          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split(/\r?\n/);
            buffer = lines.pop() || "";

            for (const line of lines) {
              const trimmed = line.trim();
              if (trimmed.startsWith("data:")) {
                const jsonStr = trimmed.slice(5).trim();
                if (!jsonStr) continue;
                try {
                  const parsed: BatchStreamEvent = JSON.parse(jsonStr);
                  onEvent(parsed);
                } catch (err) {
                  console.error("SSE chunk parse error:", err);
                }
              }
            }
          }
        } catch (err: any) {
          if (err?.name !== "AbortError") {
            onEvent({
              type: "stream_error",
              progress_percent: 100,
              reason: err?.message || "Batch upload stream interrupted.",
            });
          }
        }
      })();

      return {
        cancel: () => abortController.abort(),
      };
    }

    // Trường hợp 2: Chạy cohort mặc định demo_cohort.vcf -> gọi GET /predict/batch-stream
    const params = new URLSearchParams({
      stream_delay_ms: String(intervalMs),
      dna_enabled: String(ablation.dnaFeatures),
      protein_enabled: String(ablation.proteinFeatures),
      bio_enabled: String(ablation.biologicalFeatures),
      geo_enabled: String(ablation.spatialGeometry),
    });

    const eventSource = new EventSource(
      `${API_BASE_URL}/predict/batch-stream?${params.toString()}`
    );

    eventSource.onmessage = (e) => {
      try {
        const parsed: BatchStreamEvent = JSON.parse(e.data);
        onEvent(parsed);
        if (parsed.type === "stream_completed" || parsed.type === "stream_error") {
          eventSource.close();
        }
      } catch (err) {
        console.error("SSE parse error", err);
      }
    };

    eventSource.onerror = () => {
      onEvent({
        type: "stream_error",
        progress_percent: 100,
        reason: "SSE connection closed or backend unavailable.",
      });
      eventSource.close();
    };

    return {
      cancel: () => eventSource.close(),
    };
  }

  // Chế độ Mock (khi VITE_USE_MOCK_API = "true")
  const basePool = mockData.batch_variants as BatchVariant[];
  const extraMockEvents: Array<
    | { kind: "valid"; item: BatchVariant }
    | { kind: "skip"; reason: string }
  > = [
    ...basePool.map((item) => ({ kind: "valid" as const, item })),
    { kind: "skip", reason: "synonymous_variant (filtered by Gatekeeper)" },
    {
      kind: "valid",
      item: {
        id: 9,
        variant_id: "chr13:32906729:A>C",
        gene: "BRCA2",
        consequence: "p.Asn372His",
        rb_net_probability: 0.08,
        classification: "Benign",
        clinvar_ground_truth: "Benign",
        sota_alphamissense: 0.12,
        danger_rank: 9,
      },
    },
    { kind: "skip", reason: "intron_variant (filtered by Gatekeeper)" },
    {
      kind: "valid",
      item: {
        id: 10,
        variant_id: "chr19:11216264:G>A",
        gene: "LDLR",
        consequence: "p.Cys167Tyr",
        rb_net_probability: 0.87,
        classification: "Pathogenic",
        clinvar_ground_truth: "Pathogenic",
        sota_alphamissense: 0.83,
        danger_rank: 10,
      },
    },
  ];

  let cursor = 0;
  const total = extraMockEvents.length;

  const timer = window.setInterval(() => {
    if (cursor >= total) {
      window.clearInterval(timer);
      onEvent({
        type: "stream_completed",
        progress_percent: 100,
      });
      return;
    }

    const current = extraMockEvents[cursor];
    cursor += 1;
    const progress = Math.min(99, Math.round((cursor / total) * 100));

    if (current.kind === "skip") {
      onEvent({
        type: "variant_skipped",
        progress_percent: progress,
        reason: current.reason,
      });
    } else {
      let prob = current.item.rb_net_probability;
      const dir = prob >= 0.5 ? 1 : -1;
      if (!ablation.proteinFeatures) prob -= dir * 0.12;
      if (!ablation.biologicalFeatures) prob -= dir * 0.1;
      if (!ablation.spatialGeometry) prob -= dir * 0.08;
      if (!ablation.dnaFeatures) prob -= dir * 0.05;

      const clampedProb = Number(Math.max(0.01, Math.min(0.99, prob)).toFixed(3));
      const classification: ClassificationType =
        clampedProb >= 0.5 ? "Pathogenic" : "Benign";

      onEvent({
        type: "variant_stream",
        progress_percent: progress,
        variant: {
          ...current.item,
          rb_net_probability: clampedProb,
          classification,
        },
      });
    }
  }, intervalMs);

  return {
    cancel: () => window.clearInterval(timer),
  };
}