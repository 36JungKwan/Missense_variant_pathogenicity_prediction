import React, { useMemo } from "react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts"
import type { IgFeature } from "@/types"
import { IG_NEGATIVE, IG_POSITIVE, formatSigned } from "@/lib/xai"

const GROUP_LABEL = { Bio_SpliceAI: "Bio / SpliceAI", Geometry: "Geometry" } as const

/** IG của 28 đặc trưng bảng, sắp theo |IG| giảm dần. */
export const IgFeatureChart: React.FC<{ features: IgFeature[] }> = ({ features }) => {
  const data = useMemo(
    () => [...features]
      .sort((a, b) => Math.abs(b.signed) - Math.abs(a.signed))
      .map((f) => ({ ...f, value: Number(f.signed.toFixed(4)) })),
    [features]
  )
  return (
    <div style={{ height: data.length * 24 + 50 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 4 }} barCategoryGap={2}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
          <ReferenceLine x={0} stroke="#94a3b8" strokeWidth={1.5} />
          <XAxis type="number" tick={{ fontSize: 10, fill: "#64748b" }} domain={["auto", "auto"]} />
          <YAxis dataKey="name" type="category" width={190} interval={0} tick={{ fontSize: 10.5, fill: "#475569" }} />
          <RechartsTooltip
            cursor={{ fill: "#f8fafc" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null
              const f = payload[0].payload as IgFeature & { value: number }
              const raw = f.raw_value !== null && f.raw_value !== undefined ? f.raw_value.toPrecision(4) : f.raw_note ?? "n/a"
              return (
                <div className="max-w-[270px] rounded-xl border border-slate-200 bg-white p-3 text-[11px] shadow-lg">
                  <p className="mb-0.5 text-xs font-bold text-slate-800">{f.name}</p>
                  <p className="mb-2 font-mono text-[10px] text-slate-400">{f.feature} &middot; {GROUP_LABEL[f.group]}</p>
                  {f.description && <p className="mb-2 leading-tight text-slate-500">{f.description}</p>}
                  <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-slate-700">
                    <span className="text-slate-500">IG</span><span>{formatSigned(f.signed, 4)}</span>
                    <span className="text-slate-500">Raw value</span><span>{raw}</span>
                    <span className="text-slate-500">Model input</span><span>{f.input_value.toFixed(3)}</span>
                  </div>
                </div>
              )
            }}
          />
          <Bar dataKey="value" barSize={14} radius={[3, 3, 3, 3]}>
            {data.map((f) => <Cell key={f.feature} fill={f.signed >= 0 ? IG_POSITIVE : IG_NEGATIVE} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
