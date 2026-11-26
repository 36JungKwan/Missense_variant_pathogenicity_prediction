import React, { useMemo, useState } from "react"
import type { IgToken } from "@/types"
import { DIVERGING_GRADIENT, divergingColor, formatSigned } from "@/lib/xai"

const CELL = 22
const LEFT = 40
const TOP = 6
const BOTTOM = 22

/** IG theo vị trí token (-16..+16) của hai allele Ref/Alt cho một nhánh chuỗi. */
export const IgTokenStrip: React.FC<{ modality: "DNA" | "Protein"; tokens: IgToken[] }> = ({ modality, tokens }) => {
  const [hover, setHover] = useState<IgToken | null>(null)
  const rows = useMemo(() => {
    const own = tokens.filter((t) => t.modality === modality)
    return (["Ref", "Alt"] as const).map((allele) =>
      own.filter((t) => t.allele === allele).sort((a, b) => a.position - b.position))
  }, [tokens, modality])
  const maxAbs = Math.max(1e-9, ...rows.flat().map((t) => Math.abs(t.signed)))
  const columns = rows[0].length
  const width = LEFT + columns * CELL
  const height = TOP + 2 * CELL + BOTTOM
  const total = rows.flat().reduce((s, t) => s + t.signed, 0)

  const cells = useMemo(() => rows.flatMap((row, r) => row.map((t, c) => (
    <rect
      key={`${r}-${c}`}
      x={LEFT + c * CELL + 1}
      y={TOP + r * CELL + 1}
      width={CELL - 2}
      height={CELL - 2}
      rx={3}
      fill={divergingColor(t.signed, maxAbs)}
      onMouseEnter={() => setHover(t)}
    />
  ))), [rows, maxAbs])

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between text-xs">
        <span className="font-semibold text-slate-700">{modality} window (33 tokens &times; Ref/Alt)</span>
        <span className="font-mono text-slate-500">&Sigma; = {formatSigned(total)}</span>
      </div>
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full min-w-[560px]"
          role="img"
          aria-label={`Integrated Gradients per ${modality} token position`}
          onMouseLeave={() => setHover(null)}
        >
          {(["Ref", "Alt"] as const).map((allele, r) => (
            <text key={allele} x={LEFT - 6} y={TOP + r * CELL + CELL / 2 + 4} textAnchor="end" fontSize={10} fill="#475569">
              {allele}
            </text>
          ))}
          {cells}
          {/* cột biến thể (+0) */}
          <rect x={LEFT + 16 * CELL} y={TOP} width={CELL} height={2 * CELL} fill="none" stroke="#0f172a" strokeWidth={1.5} rx={3} />
          {rows[0].map((t, c) => (t.position % 4 === 0 ? (
            <text key={t.position} x={LEFT + c * CELL + CELL / 2} y={TOP + 2 * CELL + 14} textAnchor="middle" fontSize={9}
              fill={t.position === 0 ? "#0f172a" : "#64748b"} fontWeight={t.position === 0 ? 700 : 400}>
              {t.position > 0 ? `+${t.position}` : t.position}
            </text>
          ) : null))}
        </svg>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-500">
        <span className="min-h-4 font-mono">
          {hover
            ? `${modality} ${hover.allele} ${hover.position > 0 ? "+" : ""}${hover.position}: IG ${formatSigned(hover.signed, 4)} (|IG| ${hover.absolute.toFixed(4)})`
            : "Hover a cell for its value; the outlined column is the variant position."}
        </span>
        <span className="flex items-center gap-1.5">
          <span>{formatSigned(-maxAbs, 3)}</span>
          <span className="h-2 w-24 rounded" style={{ background: DIVERGING_GRADIENT }} />
          <span>{formatSigned(maxAbs, 3)}</span>
        </span>
      </div>
    </div>
  )
}
