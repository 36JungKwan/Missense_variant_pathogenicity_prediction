// Hàm dùng chung cho trang XAI: thang màu, nhãn rút gọn, gộp head attention.
import type { AttentionMap, AttentionQuery, IgModalityName } from "@/types"

/** Thang phân kỳ cho IG có dấu: hồng = đẩy về Pathogenic, xanh lục = đẩy về Benign, giữa = xám trung tính. */
export const IG_POSITIVE = "#e11d48" // rose-600
export const IG_NEGATIVE = "#059669" // emerald-600
const NEUTRAL: RGB = [241, 245, 249] // slate-100

/** Thang tuần tự một màu (tím) cho trọng số attention: nhạt = thấp, đậm = cao. */
const SEQUENTIAL: RGB[] = [
  [245, 243, 255], // violet-50
  [196, 181, 253], // violet-300
  [124, 58, 237], // violet-600
  [59, 7, 100], // purple-950
]

type RGB = [number, number, number]

function hexToRgb(hex: string): RGB {
  const v = parseInt(hex.slice(1), 16)
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255]
}

function mix(a: RGB, b: RGB, t: number): string {
  const c = a.map((x, i) => Math.round(x + (b[i] - x) * t))
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`
}

/** Màu của một giá trị IG có dấu, chuẩn hoá theo |max| của cùng biểu đồ. */
export function divergingColor(value: number, maxAbs: number): string {
  if (!maxAbs) return mix(NEUTRAL, NEUTRAL, 0)
  const t = Math.min(1, Math.abs(value) / maxAbs)
  return mix(NEUTRAL, hexToRgb(value >= 0 ? IG_POSITIVE : IG_NEGATIVE), t)
}

/** Màu của trọng số attention trong [0, max]. */
export function sequentialColor(value: number, max: number): string {
  const t = max > 0 ? Math.min(1, Math.max(0, value / max)) : 0
  const scaled = t * (SEQUENTIAL.length - 1)
  const i = Math.min(SEQUENTIAL.length - 2, Math.floor(scaled))
  return mix(SEQUENTIAL[i], SEQUENTIAL[i + 1], scaled - i)
}

export const SEQUENTIAL_GRADIENT = `linear-gradient(to right, ${SEQUENTIAL.map((c) => `rgb(${c.join(",")})`).join(", ")})`
export const DIVERGING_GRADIENT = `linear-gradient(to right, ${IG_NEGATIVE}, rgb(${NEUTRAL.join(",")}), ${IG_POSITIVE})`

export const MODALITY_LABEL: Record<IgModalityName, string> = {
  DNA: "DNA tokens (NT-v1)",
  Protein: "Protein tokens (ESM-1b)",
  Bio_SpliceAI: "Bio + SpliceAI (20)",
  Geometry: "Geometry (8)",
}

export const QUERY_LABEL: Record<AttentionQuery, string> = {
  dna: "DNA",
  protein: "Protein",
  bio_geometry: "Bio + Geom + SpliceAI",
}

/** Nhãn ngắn cho trục heatmap (tên đầy đủ hiện trong tooltip). */
export function shortLabel(label: string): string {
  return label
    .replace("SpliceAI_pred_", "SAI ")
    .replace("way_vertebrate", "v")
    .replace("way_mammalian", "m")
    .replace("way_primate", "p")
    .replace("phastCons", "phCons")
}

export function formatSigned(value: number, digits = 3): string {
  const text = value.toFixed(digits)
  return value > 0 ? `+${text}` : text
}

/** Ma trận (query, key) của một head, hoặc trung bình 8 head khi head = "mean". */
export function headMatrix(map: AttentionMap, head: number | "mean"): number[][] {
  if (head !== "mean") return map.weights[head]
  const heads = map.weights.length
  return map.weights[0].map((row, q) => row.map((_, k) => {
    let sum = 0
    for (let h = 0; h < heads; h += 1) sum += map.weights[h][q][k]
    return sum / heads
  }))
}

/** Query mặc định khi mở heatmap: token biến thể (+0) của nhánh chuỗi, hoặc gnomADe_AF của nhánh bảng. */
export function defaultFocus(map: AttentionMap): number {
  const preferred = map.query === "bio_geometry" ? "gnomADe_AF" : `${QUERY_LABEL[map.query]}:+0`
  const index = map.query_labels.indexOf(preferred)
  return index >= 0 ? index : 0
}
