import React, { useEffect, useState } from "react"
import { Binary, Dna, Flame, Layers, RotateCcw } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { FULL_ABLATION, ablationStateToKey, countActiveModules } from "@/lib/ablation"
import { fetchTestMcc } from "@/services/api"
import type { AblationState } from "@/types"

interface ModuleSpec {
  key: keyof AblationState
  title: string
  model: string
  detail: string
  icon: React.ElementType
  iconClass: string
}

/** 3 nhóm khớp đúng 3 bit (use_dna, use_prot, use_biogeom) của checkpoint TriMVPP. */
const MODULES: ModuleSpec[] = [
  { key: "dnaFeatures", title: "DNA Sequence", model: "NT-v1 500M", detail: "1001 bp window, 33 ref/alt tokens",
    icon: Binary, iconClass: "text-blue-500" },
  { key: "proteinFeatures", title: "Protein Sequence", model: "ESM-1b 650M", detail: "501 aa window, 33 ref/alt tokens",
    icon: Dna, iconClass: "text-purple-500" },
  { key: "bioGeomFeatures", title: "Bio + Geom + SpliceAI", model: "28 tabular features",
    detail: "11 conservation/AF + 9 SpliceAI + 8 manifold geometry (LVD/LID)", icon: Flame, iconClass: "text-amber-500" },
]

interface FeatureAblationPanelProps {
  value: AblationState
  onChange: (next: AblationState) => void
  disabled?: boolean
  title?: string
  /** "vertical" xếp 3 module thành 1 cột (dùng khi panel nằm trong cột hẹp). */
  orientation?: "horizontal" | "vertical"
}

export const FeatureAblationPanel: React.FC<FeatureAblationPanelProps> = ({
  value,
  onChange,
  disabled = false,
  title = "Feature Ablation Control Panel",
  orientation = "horizontal",
}) => {
  const active = countActiveModules(value)
  const [testMcc, setTestMcc] = useState<Record<string, number>>({})
  useEffect(() => {
    let mounted = true
    fetchTestMcc().then((scores) => mounted && setTestMcc(scores))
    return () => { mounted = false }
  }, [])
  const mcc = testMcc[ablationStateToKey(value)]

  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Layers className="h-4 w-4 text-indigo-600" />
            <span>{title}</span>
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className={`text-xs ${active === 0 ? "border-rose-300 text-rose-700" : "border-slate-200"}`}>
              {active}/3 Modules Active
            </Badge>
            {active < 3 && (
              <Button
                variant="ghost"
                size="sm"
                disabled={disabled}
                onClick={() => onChange(FULL_ABLATION)}
                className="h-7 px-2 text-xs text-slate-600 hover:text-blue-600"
                title="Enable all modules"
              >
                <RotateCcw className="h-3 w-3 mr-1" />
                Reset
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className={`grid grid-cols-1 ${orientation === "vertical" ? "md:grid-cols-3 lg:grid-cols-1" : "md:grid-cols-3"} gap-3 p-3 rounded-xl bg-slate-50 border border-slate-100`}>
          {MODULES.map(({ key, title: label, model, detail, icon: Icon, iconClass }, index) => (
            <label
              key={key}
              className={`p-3 bg-white rounded-lg border shadow-2xs flex items-start justify-between gap-3 ${
                value[key] ? "border-slate-200/80" : "border-dashed border-slate-300 opacity-75"
              } ${disabled ? "cursor-not-allowed" : "cursor-pointer"}`}
            >
              <div className="space-y-0.5 min-w-0">
                <div className="flex items-center gap-2">
                  <Icon className={`h-4 w-4 shrink-0 ${iconClass}`} />
                  <span className="text-sm font-semibold text-slate-800">
                    {index + 1}. {label}
                  </span>
                </div>
                <p className="text-xs font-medium text-slate-600">{model}</p>
                <p className="text-[11px] text-slate-500 leading-snug">{detail}</p>
              </div>
              <Switch
                checked={value[key]}
                disabled={disabled}
                onCheckedChange={(checked) => onChange({ ...value, [key]: checked })}
              />
            </label>
          ))}
        </div>
        {mcc !== undefined && (
          <p className="mt-2 text-[11px] text-slate-400">Test-set MCC of this configuration: {mcc.toFixed(2)}</p>
        )}
        {active === 0 && (
          <p className="mt-2 text-xs font-medium text-rose-600">At least one module must stay enabled.</p>
        )}
      </CardContent>
    </Card>
  )
}
