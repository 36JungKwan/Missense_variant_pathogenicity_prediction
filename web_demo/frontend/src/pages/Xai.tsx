import React, { useState, useEffect, useMemo } from "react"
import { useSearchParams, useNavigate } from "react-router-dom"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine
} from "recharts"
import { 
  ArrowLeft,
  BarChart3,
  Activity,
  Binary,
  Dna,
  Flame,
  Layers,
  Loader2,
  ShieldAlert,
  BrainCircuit,
  ChevronRight,
} from "lucide-react"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { describeModules, maskToAblationState } from "@/lib/ablation"
import type { SingleVariantResponse } from "@/types"
import { fetchVariantXai } from "@/services/api"

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

  const [data, setData] = useState<SingleVariantResponse | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  // Fetch XAI data from real backend
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
    { label: "DNA Sequence (NT-v1 500M)", active: modules.dnaFeatures, icon: Binary, tone: "text-blue-500", on: "bg-blue-50 text-blue-700" },
    { label: "Protein Sequence (ESM2 650M)", active: modules.proteinFeatures, icon: Dna, tone: "text-purple-500", on: "bg-purple-50 text-purple-700" },
    { label: "Bio + Geom + SpliceAI", active: modules.bioGeomFeatures, icon: Flame, tone: "text-amber-500", on: "bg-amber-50 text-amber-700" },
  ]
  const threshold = data?.meta?.decision_threshold ?? null

  // Prepare SHAP chart data
  const chartData = useMemo(() => {
    if (!data || !data.shap_analysis) return []
    // Sort by absolute contribution magnitude for waterfall effect
    return [...data.shap_analysis]
      .sort((a, b) => Math.abs(b.contribution_value) - Math.abs(a.contribution_value))
      .map(item => ({
        ...item,
        displayValue: Number(item.contribution_value.toFixed(3)),
        isPositive: item.contribution_value >= 0
      }))
  }, [data])

  // chartData đã sắp theo |SHAP| giảm dần -> lấy 3 đầu tiên mỗi phía
  const topDrivers = useMemo(
    () => ({
      positive: chartData.filter((d) => d.isPositive).slice(0, 3),
      negative: chartData.filter((d) => !d.isPositive).slice(0, 3),
    }),
    [chartData]
  )

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
            SHAP Feature Attribution
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
          <h3 className="text-lg font-medium text-slate-700">Extracting SHAP values...</h3>
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
                  <span className="text-xs text-slate-400 font-medium block mb-2">RB-Net Classification:</span>
                  <div className="flex items-center justify-between">
                    <Badge variant={data.results.rb_net.label === "Pathogenic" ? "destructive" : "success"}>
                      {data.results.rb_net.label}
                    </Badge>
                    <span className="text-sm font-mono font-bold">
                      {(data.results.rb_net.probability * 100).toFixed(1)}%
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

            {/* Top drivers: 3 đặc trưng đẩy mạnh nhất về mỗi phía */}
            {chartData.length > 0 && (
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
                          <div key={item.name} className="flex items-center justify-between gap-3 text-xs">
                            <span className="truncate text-slate-700" title={item.description}>{item.name}</span>
                            <span className="font-mono font-semibold text-slate-800">
                              {item.isPositive ? "+" : ""}{item.displayValue}
                            </span>
                          </div>
                        ))
                      )}
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}
          </div>

          {/* Right Column: SHAP Chart */}
          <Card className="lg:col-span-8 border-slate-200 shadow-sm flex flex-col">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-purple-600" />
                SHAP Impact on Pathogenicity
              </CardTitle>
              <CardDescription className="text-xs">
                Visualizing local feature contributions driving the current prediction.
                (Positive values push toward Pathogenic, negative toward Benign).
              </CardDescription>
            </CardHeader>
            <CardContent className="flex-1 p-6" style={{ height: Math.max(480, chartData.length * 44 + 60) }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  layout="vertical"
                  margin={{ top: 20, right: 30, left: 10, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e2e8f0" />
                  <ReferenceLine x={0} stroke="#94a3b8" strokeWidth={2} />
                  <XAxis 
                    type="number" 
                    tick={{ fontSize: 11, fill: '#64748b' }} 
                    domain={['auto', 'auto']} 
                  />
                  <YAxis 
                    dataKey="name" 
                    type="category" 
                    width={180} 
                    tick={{ fontSize: 11, fill: '#475569', fontWeight: 500 }} 
                  />
                  <RechartsTooltip
                    cursor={{ fill: '#f8fafc' }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const pData = payload[0].payload
                        return (
                          <div className="bg-white p-3 border border-slate-200 shadow-lg rounded-xl max-w-[250px]">
                            <p className="text-xs font-bold text-slate-800 mb-1">{pData.name}</p>
                            <p className="text-[11px] text-slate-500 mb-2 leading-tight">{pData.description}</p>
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-medium text-slate-600">SHAP Value:</span>
                              <Badge variant={pData.isPositive ? "destructive" : "success"} className="text-[10px] px-1.5 font-mono">
                                {pData.isPositive ? "+" : ""}{pData.displayValue}
                              </Badge>
                            </div>
                          </div>
                        )
                      }
                      return null
                    }}
                  />
                  <Bar dataKey="displayValue" barSize={32} radius={[4, 4, 4, 4]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.isPositive ? '#f43f5e' : '#10b981'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}