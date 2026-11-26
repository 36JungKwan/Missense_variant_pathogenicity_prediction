import React from "react"
import type { IgModality } from "@/types"
import { IG_NEGATIVE, IG_POSITIVE, MODALITY_LABEL, formatSigned } from "@/lib/xai"

/** IG theo nhánh: thanh phân kỳ quanh trục 0 (phải = Pathogenic, trái = Benign). */
export const IgModalityBars: React.FC<{ modalities: IgModality[] }> = ({ modalities }) => {
  const maxAbs = Math.max(1e-9, ...modalities.map((m) => Math.abs(m.signed)))
  const totalAbs = modalities.reduce((sum, m) => sum + Math.abs(m.signed), 0) || 1
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-[minmax(0,10rem)_1fr_5.5rem] items-center gap-3 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
        <span>Branch</span>
        <div className="flex justify-between"><span>&larr; Benign</span><span>Pathogenic &rarr;</span></div>
        <span className="text-right">Signed IG</span>
      </div>
      {modalities.map((m) => {
        const width = (Math.abs(m.signed) / maxAbs) * 50
        const positive = m.signed >= 0
        return (
          <div
            key={m.modality}
            className="grid grid-cols-[minmax(0,10rem)_1fr_5.5rem] items-center gap-3 text-xs"
            title={`Sum of |IG| over all inputs of this branch: ${m.absolute.toFixed(3)}`}
          >
            <span className="truncate font-medium text-slate-700">{MODALITY_LABEL[m.modality]}</span>
            <div className="relative h-6 rounded bg-slate-50">
              <div className="absolute inset-y-0 left-1/2 w-px bg-slate-300" />
              <div
                className="absolute inset-y-1 rounded"
                style={{
                  width: `${width}%`,
                  left: positive ? "50%" : `${50 - width}%`,
                  background: positive ? IG_POSITIVE : IG_NEGATIVE,
                }}
              />
            </div>
            <span className="text-right font-mono text-slate-800">
              {formatSigned(m.signed)}
              <span className="block text-[10px] text-slate-400">{((Math.abs(m.signed) / totalAbs) * 100).toFixed(0)}% of |IG|</span>
            </span>
          </div>
        )
      })}
    </div>
  )
}
