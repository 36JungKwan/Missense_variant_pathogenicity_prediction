import React, { useMemo, useState } from "react"
import type { AttentionMaps, AttentionQuery } from "@/types"
import { QUERY_LABEL, SEQUENTIAL_GRADIENT, defaultFocus, headMatrix, sequentialColor, shortLabel } from "@/lib/xai"

const CELL = 11
const TOP = 16

function isFeatureSegment(name: AttentionQuery) {
  return name === "bio_geometry"
}

/** Vạch chia trên trục: token -> mỗi 8 vị trí; đặc trưng -> mọi tên (xoay dọc). */
function showTick(label: string, feature: boolean, hideEdges = false): boolean {
  if (feature) return true
  const position = Number(label.split(":")[1])
  return position % 8 === 0 && !(hideEdges && Math.abs(position) === 16)
}

const Segmented: React.FC<{
  label: string
  options: { value: string; text: string }[]
  value: string
  onChange: (value: string) => void
}> = ({ label, options, value, onChange }) => (
  <div className="flex flex-wrap items-center gap-1.5">
    <span className="mr-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">{label}</span>
    {options.map((o) => (
      <button
        key={o.value}
        type="button"
        onClick={() => onChange(o.value)}
        className={`rounded-md border px-2 py-1 text-[11px] font-medium transition-colors ${
          value === o.value
            ? "border-purple-600 bg-purple-600 text-white"
            : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
        }`}
      >
        {o.text}
      </button>
    ))}
  </div>
)

export const AttentionPanel: React.FC<{ attention: AttentionMaps }> = ({ attention }) => {
  const [layer, setLayer] = useState(attention.num_layers)
  const [head, setHead] = useState<number | "mean">("mean")
  const queries = attention.layers[0].maps.map((m) => m.query)
  const [query, setQuery] = useState<AttentionQuery>(queries.includes("protein") ? "protein" : queries[0])
  const map = attention.layers.find((l) => l.layer === layer)!.maps.find((m) => m.query === query)!
  // Query đang chọn để liệt kê context; nhớ riêng cho từng nhánh (số dòng khác nhau: 33 token / 28 đặc trưng).
  const [focusByQuery, setFocusByQuery] = useState<Partial<Record<AttentionQuery, number>>>({})
  const focus = focusByQuery[query] ?? defaultFocus(map)
  const [hover, setHover] = useState<{ q: number; k: number } | null>(null)

  const matrix = useMemo(() => headMatrix(map, head), [map, head])
  const max = useMemo(() => Math.max(1e-9, ...matrix.flat()), [matrix])
  const uniform = 1 / map.key_labels.length

  const queryIsFeature = isFeatureSegment(map.query)
  const left = queryIsFeature ? 118 : 72
  const hasFeatureKeys = map.context_segments.some((s) => isFeatureSegment(s.name))
  const bottom = hasFeatureKeys ? 92 : 22
  const width = left + map.key_labels.length * CELL + 4
  const height = TOP + map.query_labels.length * CELL + bottom

  const segmentStarts = useMemo(() => map.context_segments.map((s, i) => ({
    ...s,
    start: map.context_segments.slice(0, i).reduce((sum, prev) => sum + prev.labels.length, 0),
  })), [map])

  const cells = useMemo(() => matrix.flatMap((row, q) => row.map((w, k) => (
    <rect key={`${q}-${k}`} x={left + k * CELL} y={TOP + q * CELL} width={CELL - 0.6} height={CELL - 0.6}
      fill={sequentialColor(w, max)} onMouseEnter={() => setHover({ q, k })}
      onClick={() => setFocusByQuery((prev) => ({ ...prev, [map.query]: q }))} />
  ))), [matrix, max, left, map.query, setHover, setFocusByQuery])

  const top = useMemo(() => matrix[focus]
    .map((weight, k) => ({ weight, label: map.key_labels[k] }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 8), [matrix, focus, map])

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-2">
        <Segmented
          label="Query"
          options={queries.map((q) => ({ value: q, text: QUERY_LABEL[q] }))}
          value={query}
          onChange={(v) => setQuery(v as AttentionQuery)}
        />
        <Segmented
          label="Layer"
          options={attention.layers.map((l) => ({ value: String(l.layer), text: `Layer ${l.layer}` }))}
          value={String(layer)}
          onChange={(v) => setLayer(Number(v))}
        />
        <Segmented
          label="Head"
          options={[{ value: "mean", text: "Mean of heads" },
            ...Array.from({ length: attention.num_heads }, (_, h) => ({ value: String(h), text: `H${h + 1}` }))]}
          value={String(head)}
          onChange={(v) => setHead(v === "mean" ? "mean" : Number(v))}
        />
      </div>

      <div className="overflow-x-auto rounded-lg border border-slate-100 bg-white p-2">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[620px]" role="img"
          aria-label={`Cross-attention of the ${QUERY_LABEL[map.query]} branch, layer ${layer}`}
          onMouseLeave={() => setHover(null)}>
          {segmentStarts.map((s, i) => (
            <g key={s.name}>
              <text x={left + (s.start + s.labels.length / 2) * CELL} y={10} textAnchor="middle" fontSize={9} fontWeight={700} fill="#334155">
                {QUERY_LABEL[s.name]} (keys)
              </text>
              {i > 0 && <line x1={left + s.start * CELL - 0.3} x2={left + s.start * CELL - 0.3} y1={TOP - 2}
                y2={TOP + map.query_labels.length * CELL} stroke="#0f172a" strokeWidth={1} />}
              {s.labels.map((label, j) => showTick(label, isFeatureSegment(s.name), map.context_segments.length > 1) ? (
                isFeatureSegment(s.name) ? (
                  <text key={label} fontSize={7} fill="#64748b" textAnchor="end"
                    transform={`translate(${left + (s.start + j) * CELL + CELL / 2 + 2.5}, ${TOP + map.query_labels.length * CELL + 4}) rotate(-90)`}>
                    {shortLabel(label)}
                  </text>
                ) : (
                  <text key={label} x={left + (s.start + j) * CELL + CELL / 2} y={TOP + map.query_labels.length * CELL + 11}
                    textAnchor="middle" fontSize={8} fill={label.endsWith(":+0") ? "#0f172a" : "#64748b"}
                    fontWeight={label.endsWith(":+0") ? 700 : 400}>
                    {label.split(":")[1]}
                  </text>
                )
              ) : null)}
            </g>
          ))}
          {map.query_labels.map((label, q) => (queryIsFeature || showTick(label, false) ? (
            <text key={label} x={left - 4} y={TOP + q * CELL + CELL / 2 + 2.5} textAnchor="end"
              fontSize={queryIsFeature ? 7.5 : 8} fill={q === focus ? "#6d28d9" : "#475569"} fontWeight={q === focus ? 700 : 400}>
              {queryIsFeature ? shortLabel(label) : label}
            </text>
          ) : null))}
          {cells}
          <rect x={left - 1} y={TOP + focus * CELL - 0.5} width={map.key_labels.length * CELL + 1.4} height={CELL + 0.4}
            fill="none" stroke="#6d28d9" strokeWidth={1.2} pointerEvents="none" />
        </svg>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-500">
        <span className="min-h-4 font-mono">
          {hover && matrix[hover.q]?.[hover.k] !== undefined
            ? `${map.query_labels[hover.q]} → ${map.key_labels[hover.k]}: ${matrix[hover.q][hover.k].toFixed(4)}`
            : "Hover a cell for its weight; click a row to list its top context keys."}
        </span>
        <span className="flex items-center gap-1.5">
          <span>0</span>
          <span className="h-2 w-24 rounded" style={{ background: SEQUENTIAL_GRADIENT }} />
          <span>{max.toFixed(3)}</span>
        </span>
      </div>

      <div className="rounded-lg border border-slate-100 bg-slate-50/60 p-3">
        <p className="mb-2 text-xs font-semibold text-slate-700">
          Top context keys for query <span className="font-mono text-purple-700">{map.query_labels[focus]}</span>
          <span className="ml-1 font-normal text-slate-400">
            (layer {layer}, {head === "mean" ? "mean of heads" : `head ${head + 1}`}; uniform = {uniform.toFixed(3)})
          </span>
        </p>
        <div className="space-y-1">
          {top.map(({ label, weight }) => (
            <div key={label} className="grid grid-cols-[minmax(0,11rem)_1fr_3.5rem] items-center gap-2 text-[11px]">
              <span className="truncate font-mono text-slate-700" title={label}>{label}</span>
              <div className="h-2 rounded bg-white">
                <div className="h-2 rounded bg-violet-600" style={{ width: `${(weight / top[0].weight) * 100}%` }} />
              </div>
              <span className="text-right font-mono text-slate-800">{weight.toFixed(3)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
