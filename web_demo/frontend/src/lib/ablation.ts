import type { AblationFlags, AblationState } from "@/types"

export const FULL_ABLATION: AblationState = { dnaFeatures: true, proteinFeatures: true, bioGeomFeatures: true }

/** 4 cờ -> bitmask 0..15 (bit 3 DNA, bit 2 Protein, bit 1 Bio, bit 0 Geometry) — khớp backend. */
export function computeAblationMask(flags: AblationFlags): number {
  return (
    (flags.dna_enabled ? 8 : 0) |
    (flags.protein_enabled ? 4 : 0) |
    (flags.bio_enabled ? 2 : 0) |
    (flags.geo_enabled ? 1 : 0)
  )
}

export function decodeMaskToFlags(mask: number): AblationFlags {
  return {
    dna_enabled: (mask & 8) !== 0,
    protein_enabled: (mask & 4) !== 0,
    bio_enabled: (mask & 2) !== 0,
    geo_enabled: (mask & 1) !== 0,
  }
}

/** Nhóm Bio + Geom + SpliceAI điều khiển đồng thời 2 bit bio_enabled và geo_enabled. */
export function ablationStateToFlags(state: AblationState): AblationFlags {
  return {
    dna_enabled: state.dnaFeatures,
    protein_enabled: state.proteinFeatures,
    bio_enabled: state.bioGeomFeatures,
    geo_enabled: state.bioGeomFeatures,
  }
}

export function ablationStateToMask(state: AblationState): number {
  return computeAblationMask(ablationStateToFlags(state))
}

export function maskToAblationState(mask: number): AblationState {
  return { dnaFeatures: (mask & 8) !== 0, proteinFeatures: (mask & 4) !== 0, bioGeomFeatures: (mask & 3) !== 0 }
}

export function countActiveModules(state: AblationState): number {
  return Number(state.dnaFeatures) + Number(state.proteinFeatures) + Number(state.bioGeomFeatures)
}

/** Chuẩn hoá trạng thái đọc từ sessionStorage (kể cả định dạng 4 công tắc cũ). */
export function normalizeAblationState(raw: unknown): AblationState {
  if (!raw || typeof raw !== "object") return FULL_ABLATION
  const r = raw as Record<string, unknown>
  const bool = (v: unknown, fallback: boolean) => (typeof v === "boolean" ? v : fallback)
  const legacyBioGeom =
    typeof r.biologicalFeatures === "boolean" || typeof r.spatialGeometry === "boolean"
      ? Boolean(r.biologicalFeatures ?? true) && Boolean(r.spatialGeometry ?? true)
      : true
  return {
    dnaFeatures: bool(r.dnaFeatures, true),
    proteinFeatures: bool(r.proteinFeatures, true),
    bioGeomFeatures: bool(r.bioGeomFeatures, legacyBioGeom),
  }
}

/** Mô tả ngắn các module đang bật, ví dụ "DNA · Protein · Bio+Geom". */
export function describeModules(state: AblationState): string {
  const parts = [
    state.dnaFeatures && "DNA",
    state.proteinFeatures && "Protein",
    state.bioGeomFeatures && "Bio+Geom+SpliceAI",
  ].filter(Boolean)
  return parts.length ? parts.join(" · ") : "None"
}
