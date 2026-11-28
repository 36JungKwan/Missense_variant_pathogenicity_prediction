import React, { useState, useEffect, useMemo } from "react"
import { useSearchParams, useNavigate } from "react-router-dom"
import {
  ArrowLeft,
  AlertTriangle,
  BarChart3,
  Activity,
  Binary,
  CheckCircle2,
  Dna,
  Flame,
  Grid3x3,
  Info,
  Layers,
  Loader2,
  ShieldAlert,
  BrainCircuit,
  ChevronRight,
  Sigma,
} from "lucide-react"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { AttentionPanel } from "@/components/xai/AttentionPanel"
import { IgFeatureChart } from "@/components/xai/IgFeatureChart"
import { IgModalityBars } from "@/components/xai/IgModalityBars"
import { IgTokenStrip } from "@/components/xai/IgTokenStrip"
import { countActiveModules, describeModules, maskToAblationState } from "@/lib/ablation"
import { formatSigned } from "@/lib/xai"
import type { XaiResponse } from "@/types"
import { fetchVariantXai } from "@/services/api"

interface Driver {
  key: string
  label: string
  detail: string
  value: number
}

export const Xai: React.FC = () => {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()

  // Parse query parameters
  const variantIdParam = searchParams.get("variant") || "chr7:140753336:A>T"
  const maskParam = parseInt(searchParams.get("mask") || "15", 10)
  const validMask = (isNaN(maskParam) || maskParam < 0 || maskParam > 15) ? 15 : maskParam

  // Ngữ cảnh điều hướng: mở từ Batch Screening hay Single Variant (mặc định)
  const fromBatch = searchParams.get("from") === "batch"
  const back = useMemo(() => {
    if (fromBatch) return { label: "Back to Batch Screening", crumb: "Batch Screening", to: "/batch" }
    const params = new URLSearchParams({ variant: variantIdParam, mask: String(validMask) })
    // Single Variant được mở từ Batch -> giữ ngữ cảnh đó khi quay lại
    if (searchParams.get("origin") === "batch") params.set("from", "batch")
    return { label: "Back to Single Variant", crumb: "Single Variant", to: `/single?${params.toString()}` }
  }, [fromBatch, searchParams, variantIdParam, validMask])

  const [data, setData] = useState<XaiResponse | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true

    const loadXai = async () => {
      try {
        setIsLoading(true)
        setErrorMessage(null)
        const responseData = await fetchVariantXai(variantIdParam, validMask)
        if (isMounted) {
          setData(responseData)
        }
      } catch (error: any) {
        if (isMounted) {
          setErrorMessage(error.message || "Failed to load Explainable AI data.")
          setData(null)
        }
      } finally {
        if (isMounted) {
          setIsLoading(false)
        }
      }
    }

    loadXai()
    return () => { isMounted = false }
  }, [variantIdParam, validMask])

  // Bitmask -> 3 nhóm modality (Bio + Geom + SpliceAI luôn đi cùng nhau)
  const modules = useMemo(() => maskToAblationState(validMask), [validMask])
  const moduleRows = [
    { label: "DNA Sequence (NTv1-500M)", active: modules.dnaFeatures, icon: Binary, tone: "text-blue-500", on: "bg-blue-50 text-blue-700" },
    { label: "Protein Sequence (ESM-1b 650M)", active: modules.proteinFeatures, icon: Dna, tone: "text-purple-500", on: "bg-purple-50 text-purple-700" },
    { label: "Bio + Geom + SpliceAI", active: modules.bioGeomFeatures, icon: Flame, tone: "text-amber-500", on: "bg-amber-50 text-amber-700" },
  ]
  const threshold = data?.meta?.decision_threshold ?? null
  const ig = data?.ig_analysis ?? null
  const tokenModalities = useMemo(
    () => (["DNA", "Protein"] as const).filter((m) => ig?.tokens.some((t) => t.modality === m)),
    [ig]
  )

  // 3 đầu vào đẩy mạnh nhất về mỗi phía: đặc trưng bảng + từng token (vị trí, allele).
  const topDrivers = useMemo(() => {
    if (!ig) return { positive: [] as Driver[], negative: [] as Driver[] }
    const drivers: Driver[] = [
      ...ig.features.map((f) => ({ key: f.feature, label: f.name, detail: f.description ?? f.feature, value: f.signed })),
      ...ig.tokens.map((t) => ({
        key: `${t.modality}-${t.allele}-${t.position}`,
        label: `${t.modality} ${t.allele} token ${t.position > 0 ? "+" : ""}${t.position}`,
        detail: `Token at offset ${t.position} of the ${t.allele.toLowerCase()} ${t.modality} window`,
        value: t.signed,
      })),
    ].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    return {
      positive: drivers.filter((d) => d.value > 0).slice(0, 3),
      negative: drivers.filter((d) => d.value < 0).slice(0, 3),
    }
  }, [ig])

  return (
    <div className="space-y-6 text-left">
      {/* Header & Navigation */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="space-y-1">
          <nav className="flex items-center gap-1 text-xs font-semibold uppercase tracking-wider" aria-label="Breadcrumb">
            <button
              type="button"
              onClick={() => navigate(back.to)}
              className="text-slate-500 hover:text-purple-700 hover:underline cursor-pointer"
            >
              {back.crumb}
            </button>
            <ChevronRight className="h-3.5 w-3.5 text-slate-400" />
            <span className="flex items-center gap-1.5 text-purple-600">
              <BrainCircuit className="h-4 w-4" />
              XAI Attribution
            </span>
          </nav>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">
            Integrated Gradients &amp; Attention
          </h1>
          <p className="text-sm text-slate-500">
            <span className="font-mono font-semibold text-slate-700">{variantIdParam}</span>
            <span className="mx-2 text-slate-300">|</span>
            Active modules: {describeModules(modules)}
          </p>
        </div>

        <Button
          variant="outline"
          onClick={() => navigate(back.to)}
          className="text-xs border-slate-200 hover:bg-slate-50 self-start sm:self-auto"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          {back.label}
        </Button>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="flex flex-col items-center justify-center py-20 bg-slate-50/50 rounded-2xl border border-slate-100">
          <Loader2 className="h-10 w-10 text-purple-600 animate-spin mb-4" />
          <h3 className="text-lg font-medium text-slate-700">Computing Integrated Gradients and attention maps...</h3>
          <p className="text-sm text-slate-500">Querying PostgreSQL via PyTorch Engine</p>
        </div>
      )}

      {/* Error State (Gatekeeper HTTP 422) */}
      {!isLoading && errorMessage && (
        <div className="p-6 rounded-2xl bg-rose-50 border border-rose-200 flex flex-col items-center justify-center text-center">
          <ShieldAlert className="h-12 w-12 text-rose-500 mb-3" />
          <h3 className="text-lg font-bold text-rose-800 mb-1">XAI Generation Failed</h3>
          <p className="text-sm text-rose-600 max-w-md">{errorMessage}</p>
          <Button
            variant="outline"
            onClick={() => navigate(back.to)}
            className="mt-6 bg-white border-rose-200 text-rose-700 hover:bg-rose-100"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            {back.label}
          </Button>
        </div>
      )}

      {/* Main Content */}
      {!isLoading && data && !errorMessage && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Context & Metadata (dính khi cuộn) */}
          <div className="lg:col-span-4 space-y-6 lg:sticky lg:top-24">
            {/* Variant Meta Card */}
            <Card className="border-slate-200 shadow-sm bg-white">
              <CardHeader className="bg-slate-50/50 border-b border-slate-100 pb-4">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Activity className="h-4 w-4 text-slate-500" />
                  Target Variant Context
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4 space-y-4">
                <div>
                  <span className="text-xs text-slate-400 font-medium block">ID:</span>
                  <span className="text-sm font-mono font-bold text-slate-800">{data.variant_id}</span>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-xs text-slate-400 font-medium block">Gene:</span>
                    <span className="text-sm font-semibold text-slate-700">{data.gene || "N/A"}</span>
                  </div>
                  <div>
                    <span className="text-xs text-slate-400 font-medium block">Consequence:</span>
                    <span className="text-sm text-slate-700 truncate block">{data.consequence || "N/A"}</span>
                  </div>
                </div>
                
                <div className="pt-4 border-t border-slate-100">
                  <span className="text-xs text-slate-400 font-medium block mb-2">TriMVPP Classification:</span>
                  <div className="flex items-center justify-between">
                    <Badge variant={data.results.trimvpp.label === "Pathogenic" ? "destructive" : "success"}>
                      {data.results.trimvpp.label}
                    </Badge>
                    <span className="text-sm font-mono font-bold">
                      {(data.results.trimvpp.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  {threshold !== null && (
                    <p className="mt-2 text-xs font-semibold text-slate-700">
                      Threshold: &ge; {(threshold * 100).toFixed(1)}% &rarr; Pathogenic
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Ablation State Card */}
            <Card className="border-slate-200 shadow-sm bg-white">
              <CardHeader className="bg-slate-50/50 border-b border-slate-100 pb-4">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Layers className="h-4 w-4 text-slate-500" />
                  Active Modules
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4 space-y-2.5">
                {moduleRows.map(({ label, active, icon: Icon, tone, on }) => (
                  <div key={label} className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-2">
                      <Icon className={`h-3.5 w-3.5 ${tone}`} /> {label}
                    </span>
                    <Badge variant={active ? "secondary" : "outline"} className={active ? on : "text-slate-400"}>
                      {active ? "Active" : "Ablated"}
                    </Badge>
                  </div>
                ))}
              </CardContent>
            </Card>


            {/* Chất lượng IG: completeness */}
            {ig && (
              <Card className="border-slate-200 shadow-sm bg-white">
                <CardHeader className="bg-slate-50/50 border-b border-slate-100 pb-4">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Sigma className="h-4 w-4 text-slate-500" />
                    Attribution Quality
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4 space-y-3 text-xs">
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 font-mono">
                    <span className="font-sans text-slate-500">Logit (variant)</span><span className="text-right">{formatSigned(ig.logit)}</span>
                    <span className="font-sans text-slate-500">Logit (baseline)</span><span className="text-right">{formatSigned(ig.baseline_logit)}</span>
                    <span className="font-sans text-slate-500">&Sigma; IG</span>
                    <span className="text-right">{formatSigned(ig.logit - ig.baseline_logit - ig.completeness_residual)}</span>
                    <span className="font-sans text-slate-500">Residual</span>
                    <span className="text-right">{formatSigned(ig.completeness_residual, 4)} ({(ig.relative_residual * 100).toFixed(2)}%)</span>
                    <span className="font-sans text-slate-500">Integration steps</span><span className="text-right">{ig.steps}</span>
                  </div>
                  {ig.residual_over_5pct ? (
                    <p className="flex items-start gap-1.5 rounded-lg bg-amber-50 p-2 text-amber-800">
                      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                      Warning: residual above 5% of |logit &minus; baseline| even with {ig.steps} steps; read the attributions with care.
                    </p>
                  ) : (
                    <p className="flex items-start gap-1.5 rounded-lg bg-emerald-50 p-2 text-emerald-800">
                      <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                      Completeness OK: the attributions sum to the logit change within 5%.
                    </p>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Top drivers: 3 đầu vào đẩy mạnh nhất về mỗi phía */}
            {ig && (
              <Card className="border-slate-200 shadow-sm bg-white">
                <CardHeader className="bg-slate-50/50 border-b border-slate-100 pb-4">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-slate-500" />
                    Top Drivers
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4 space-y-4">
                  {[
                    { title: "Toward Pathogenic", items: topDrivers.positive, tone: "text-rose-700", empty: "No positive contributions." },
                    { title: "Toward Benign", items: topDrivers.negative, tone: "text-emerald-700", empty: "No negative contributions." },
                  ].map(({ title, items, tone, empty }) => (
                    <div key={title} className="space-y-1.5">
                      <span className={`text-xs font-bold uppercase tracking-wider ${tone}`}>{title}</span>
                      {items.length === 0 ? (
                        <p className="text-xs text-slate-400">{empty}</p>
                      ) : (
                        items.map((item) => (
                          <div key={item.key} className="flex items-center justify-between gap-3 text-xs">
                            <span className="truncate text-slate-700" title={item.detail}>{item.label}</span>
                            <span className="font-mono font-semibold text-slate-800">{formatSigned(item.value)}</span>
                          </div>
                        ))
                      )}
                    </div>
                  ))}
                  <p className="text-[11px] text-slate-400">Ranked over single tokens and tabular features.</p>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Right Column: IG + attention */}
          {ig && (
            <div className="lg:col-span-8 space-y-6">
              <Card className="border-slate-200 shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-purple-600" />
                    Integrated Gradients by Branch
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Contribution of each input group to the change of the Pathogenic logit from the training baseline
                    ({formatSigned(ig.baseline_logit)}) to this variant ({formatSigned(ig.logit)}). Positive values push toward
                    Pathogenic, negative values toward Benign.
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-6 pt-2">
                  <IgModalityBars modalities={ig.modalities} />
                </CardContent>
              </Card>

              {tokenModalities.length > 0 && (
                <Card className="border-slate-200 shadow-sm">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Binary className="h-5 w-5 text-purple-600" />
                      Integrated Gradients by Token Position
                    </CardTitle>
                    <CardDescription className="text-xs">
                      Attribution of each token in the 33-token windows (16 on each side of the variant), summed over the
                      1,280 embedding dimensions, for the reference and alternative sequences.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-6 pt-2 space-y-6">
                    {tokenModalities.map((m) => <IgTokenStrip key={m} modality={m} tokens={ig.tokens} />)}
                  </CardContent>
                </Card>
              )}

              {ig.features.length > 0 && (
                <Card className="border-slate-200 shadow-sm">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Flame className="h-5 w-5 text-purple-600" />
                      Integrated Gradients by Feature
                    </CardTitle>
                    <CardDescription className="text-xs">
                      The 20 biological and SpliceAI features and the 8 geometry features, ranked by absolute attribution.
                      Hover a bar for the raw and standardized values.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-6 pt-2">
                    <IgFeatureChart features={ig.features} />
                  </CardContent>
                </Card>
              )}

              <Card className="border-slate-200 shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Grid3x3 className="h-5 w-5 text-purple-600" />
                    Cross-Attention Maps
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Each row shows how one query token of the selected branch distributes its attention over the tokens of the
                    other branches (each row sums to 1).
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-6 pt-2">
                  {data.attention ? (
                    <AttentionPanel attention={data.attention} />
                  ) : (
                    <p className="flex items-start gap-2 rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
                      <Info className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
                      {countActiveModules(modules) === 1
                        ? "This configuration uses a single branch, so the model has no cross-attention layers. Only Integrated Gradients are shown."
                        : "Attention maps are not available for this response."}
                    </p>
                  )}
                </CardContent>
              </Card>

              <Card className="border-slate-200 shadow-sm bg-slate-50/50">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Info className="h-4 w-4 text-slate-500" />
                    How to Read This Page
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-6 pt-1">
                  <ul className="list-disc space-y-1.5 pl-4 text-xs leading-relaxed text-slate-600">
                    <li>
                      Integrated Gradients compare the variant with a baseline input: the mean token windows and the median
                      tabular features of 256 training variants. The gradient of the Pathogenic logit is integrated along the
                      straight path from the baseline to the variant ({ig.steps} steps).
                    </li>
                    <li>
                      The attributions of all inputs add up to the logit difference between the variant and the baseline, up to
                      the residual shown on the left. They explain the logit, not the probability.
                    </li>
                    <li>
                      Attention weights show where a branch reads information from the other branches. They describe the
                      information flow inside the model and are not importance scores: a heavily attended token does not
                      necessarily change the prediction.
                    </li>
                  </ul>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
