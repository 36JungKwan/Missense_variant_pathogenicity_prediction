// Client cho FastAPI backend (/api/v1). Chế độ demo offline nằm riêng trong api.mock.ts.
import { ablationStateToFlags, computeAblationMask } from "@/lib/ablation"
import { mockFetchVariantXai, mockPredictSingleVariant, mockVcfStream } from "@/services/api.mock"
import type {
  AblationState,
  ApiErrorDetail,
  BatchStreamEvent,
  SingleVariantRequest,
  SingleVariantResponse,
} from "@/types"

export type { BatchStreamEvent, SingleVariantRequest, SingleVariantResponse } from "@/types"

const USE_MOCK_API = import.meta.env.VITE_USE_MOCK_API === "true"
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1"

/** Đọc thông điệp lỗi từ {"detail": {"error_code", "message", "request_id"}} (hoặc detail dạng chuỗi). */
function extractApiErrorMessage(errBody: unknown, fallback: string): string {
  const detail = (errBody as { detail?: string | ApiErrorDetail } | null)?.detail
  if (!detail) return fallback
  if (typeof detail === "string") return detail
  return detail.request_id ? `${detail.message} (request ${detail.request_id})` : detail.message || fallback
}

async function requestJson<T>(url: string, init: RequestInit | undefined, fallbackError: string): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const errBody = await res.json().catch(() => null)
    throw new Error(extractApiErrorMessage(errBody, fallbackError))
  }
  return res.json() as Promise<T>
}

// ==========================================
// 1. SINGLE VARIANT INFERENCE (Single.tsx)
// ==========================================
export async function predictSingleVariant(req: SingleVariantRequest): Promise<SingleVariantResponse> {
  if (computeAblationMask(req) === 0) {
    throw new Error("At least one feature module (DNA, Protein, Biological, or Spatial Geometry) must be enabled.")
  }
  if (USE_MOCK_API) return mockPredictSingleVariant(req)

  return requestJson<SingleVariantResponse>(
    `${API_BASE_URL}/predict/single`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req) },
    "FastAPI inference error"
  )
}

// ==========================================
// 2. XAI SHAP (GET /xai/{variant_id}?ablation_mask=...)
// ==========================================
export async function fetchVariantXai(variantId: string, mask: number): Promise<SingleVariantResponse> {
  if (mask <= 0 || mask > 15) {
    throw new Error("Invalid ablation mask (0/15). At least one modality must remain active to compute SHAP values.")
  }
  if (USE_MOCK_API) return mockFetchVariantXai(variantId, mask)

  return requestJson<SingleVariantResponse>(
    `${API_BASE_URL}/xai/${encodeURIComponent(variantId)}?ablation_mask=${mask}`,
    undefined,
    "Failed to fetch SHAP values"
  )
}

// ==========================================
// 3. BATCH STREAM & UPLOAD (Batch.tsx)
// ==========================================
function parseSseLines(chunk: string, onEvent: (event: BatchStreamEvent) => void): void {
  for (const line of chunk.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed.startsWith("data:")) continue
    const json = trimmed.slice(5).trim()
    if (!json) continue
    try {
      onEvent(JSON.parse(json) as BatchStreamEvent)
    } catch (err) {
      console.error("SSE chunk parse error:", err)
    }
  }
}

/**
 * Chạy sàng lọc theo lô: có file -> POST /predict/batch-upload, không có -> GET /predict/batch-stream (cohort demo).
 * `mockIntervalMs` chỉ dùng cho chế độ mock; backend thật stream ngay khi mỗi biến thể xong (stream_delay_ms = 0).
 */
export function createVcfStreamService(
  fileOrName: File | string | null,
  ablation: AblationState,
  onEvent: (event: BatchStreamEvent) => void,
  mockIntervalMs = 150
): { cancel: () => void } {
  const flags = ablationStateToFlags(ablation)
  if (computeAblationMask(flags) === 0) {
    setTimeout(() => onEvent({
      type: "stream_error",
      progress_percent: 0,
      reason: "Cannot run screening with 0 active feature modules.",
    }), 50)
    return { cancel: () => {} }
  }
  if (USE_MOCK_API) return mockVcfStream(ablation, onEvent, mockIntervalMs)

  if (fileOrName instanceof File) {
    const abortController = new AbortController()
    const formData = new FormData()
    formData.append("file", fileOrName)
    for (const [key, value] of Object.entries(flags)) formData.append(key, String(value))

    ;(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/predict/batch-upload`, {
          method: "POST",
          body: formData,
          signal: abortController.signal,
        })
        if (!response.ok || !response.body) {
          const errBody = await response.json().catch(() => null)
          onEvent({
            type: "stream_error",
            progress_percent: 100,
            reason: extractApiErrorMessage(errBody, "Failed to process uploaded VCF file."),
          })
          return
        }

        const reader = response.body.getReader()
        const decoder = new TextDecoder("utf-8")
        let buffer = ""
        while (true) {
          const { value, done } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const cut = buffer.lastIndexOf("\n")
          if (cut < 0) continue
          parseSseLines(buffer.slice(0, cut), onEvent)
          buffer = buffer.slice(cut + 1)
        }
        if (buffer) parseSseLines(buffer, onEvent)
      } catch (err: unknown) {
        if ((err as Error)?.name !== "AbortError") {
          onEvent({
            type: "stream_error",
            progress_percent: 100,
            reason: (err as Error)?.message || "Batch upload stream interrupted.",
          })
        }
      }
    })()

    return { cancel: () => abortController.abort() }
  }

  const params = new URLSearchParams(Object.entries(flags).map(([k, v]) => [k, String(v)]))
  const eventSource = new EventSource(`${API_BASE_URL}/predict/batch-stream?${params.toString()}`)
  eventSource.onmessage = (e) => {
    try {
      const parsed = JSON.parse(e.data) as BatchStreamEvent
      onEvent(parsed)
      if (parsed.type === "stream_completed" || parsed.type === "stream_error") eventSource.close()
    } catch (err) {
      console.error("SSE parse error", err)
    }
  }
  eventSource.onerror = () => {
    onEvent({ type: "stream_error", progress_percent: 100, reason: "SSE connection closed or backend unavailable." })
    eventSource.close()
  }
  return { cancel: () => eventSource.close() }
}
