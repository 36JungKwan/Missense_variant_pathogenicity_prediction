import React, { useCallback, useEffect, useRef, useState } from "react"
import { useNavigate, useSearchParams } from "react-router-dom"
import {
  ArrowLeft,
  BarChart3,
  CheckCircle2,
  Cpu,
  Database,
  Dna,
  Info,
  Loader2,
  Scale,
  Search,
  ShieldAlert,
  Sparkles,
  Zap,
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { FeatureAblationPanel } from "@/components/ablation/FeatureAblationPanel"
import mockData from "@/mock/mockData.json"
import {
  FULL_ABLATION,
  ablationStateToFlags,
  ablationStateToMask,
  countActiveModules,
  maskToAblationState,
} from "@/lib/ablation"
import type { AblationState, ClinVarLabel, InferenceMeta, VariantData } from "@/types"
import { predictSingleVariant } from "@/services/api"

const DEFAULT_VARIANT = "chr7:140753336:A>T"
const ALPHAMISSENSE_THRESHOLD = 0.56
const REVEL_THRESHOLD = 0.5

/** Một dòng so sánh với mô hình SOTA; dbNSFP không có điểm (null) -> hiển thị "N/A". */
const SotaRow: React.FC<{ label: string; score: number | null; threshold: number; rbNetProbability: number }> = ({
  label,
  score,
  threshold,
  rbNetProbability,
}) => {
  const available = score !== null && score !== undefined
  return (
    <TableRow>
      <TableCell className="font-medium text-slate-800">{label}</TableCell>
      <TableCell className="font-mono text-slate-700">{available ? score.toFixed(3) : "N/A"}</TableCell>
      <TableCell>
        {available ? (
          <Badge variant={score >= threshold ? "destructive" : "success"}>
            {score >= threshold ? "Pathogenic" : "Benign"}
          </Badge>
        ) : (
          <Badge variant="outline" title="dbNSFP has no score for this variant">N/A</Badge>
        )}
      </TableCell>
      <TableCell className="text-right text-slate-600 font-mono">
        {available ? `${(100 - Math.abs(rbNetProbability - score) * 100).toFixed(1)}%` : "N/A"}
      </TableCell>
    </TableRow>
  )
}

interface SotaChartRow {
  label: string
  score: number | null
  threshold: number | null
  anchor?: boolean
}

const AXIS_TICKS = [0, 0.25, 0.5, 0.75, 1]

/** Biểu đồ cột ngang: điểm của từng mô hình trên thang 0-1, kèm vạch ngưỡng riêng của mỗi mô hình. */
const SotaScoreChart: React.FC<{ rows: SotaChartRow[] }> = ({ rows }) => (
  <div className="space-y-3" role="img" aria-label="Score comparison chart across models">
    {rows.map(({ label, score, threshold, anchor }) => {
      const available = score !== null && score !== undefined
      return (
        <div key={label} className="group relative grid grid-cols-[minmax(0,9rem)_1fr_3rem] items-center gap-3">
          <span className={`text-xs truncate ${anchor ? "font-semibold text-slate-900" : "text-slate-600"}`} title={label}>
            {label}
          </span>
          <div className="relative h-6">
            {/* Lưới dọc mờ */}
            {AXIS_TICKS.map((t) => (
              <div key={t} className="absolute top-0 bottom-0 w-px bg-slate-100" style={{ left: `${t * 100}%` }} />
            ))}
            {available ? (
              <div
                className={`absolute left-0 top-1 bottom-1 rounded-r ${anchor ? "bg-blue-600" : "bg-blue-300"} transition-all duration-500`}
                style={{ width: `${score * 100}%` }}
              />
            ) : (
              <div className="absolute inset-y-1 left-0 right-0 rounded border border-dashed border-slate-300" />
            )}
            {threshold !== null && (
              <div
                className="absolute -top-0.5 -bottom-0.5 w-0.5 bg-slate-900"
                style={{ left: `${threshold * 100}%` }}
              />
            )}
            {/* Tooltip khi hover */}
            <div className="pointer-events-none absolute z-10 bottom-full mb-1.5 left-1/2 -translate-x-1/2 hidden group-hover:block whitespace-nowrap rounded-md bg-slate-900 px-2.5 py-1.5 text-[11px] text-white shadow-lg">
              <div className="font-semibold">{label}</div>
              <div>Score: {available ? score.toFixed(3) : "N/A (no dbNSFP score)"}</div>
              {threshold !== null && <div>Threshold: {threshold.toFixed(3)}</div>}
            </div>
          </div>
          <span className="text-xs font-mono text-slate-700 text-right">{available ? score.toFixed(3) : "N/A"}</span>
        </div>
      )
    })}
    {/* Trục x */}
    <div className="grid grid-cols-[minmax(0,9rem)_1fr_3rem] gap-3">
      <span />
      <div className="relative h-4">
        {AXIS_TICKS.map((t) => (
          <span
            key={t}
            className="absolute -translate-x-1/2 text-[10px] font-mono text-slate-400"
            style={{ left: `${t * 100}%` }}
          >
            {t}
          </span>
        ))}
      </div>
      <span />
    </div>
    <p className="flex items-center gap-1.5 text-[11px] text-slate-500">
      <span className="inline-block h-3 w-0.5 bg-slate-900" />
      Model-specific classification threshold (≥ threshold → Pathogenic)
    </p>
  </div>
)

const clinvarBadgeVariant =(label: ClinVarLabel) =>
  label === "Pathogenic" ? "destructive" : label === "Benign" ? "success" : "warning"

export const Single: React.FC = () => {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const openedFromBatch = searchParams.get("from") === "batch"
  const initialMask = Number.parseInt(searchParams.get("mask") || "", 10)

  // Input (chỉ chạy suy luận khi bấm Run / Enter — gõ phím không gọi API)
  const [variantInput, setVariantInput] = useState<string>(searchParams.get("variant") || DEFAULT_VARIANT)
  const [activePresetId, setActivePresetId] = useState<string | null>(searchParams.get("variant") ? null : "pathogenic")
  const [useCache, setUseCache] = useState<boolean>(true)
  const [ablation, setAblation] = useState<AblationState>(
    Number.isNaN(initialMask) ? FULL_ABLATION : maskToAblationState(initialMask)
  )

  // Kết quả
  const [submittedVariant, setSubmittedVariant] = useState<string | null>(null)
  const [currentResult, setCurrentResult] = useState<VariantData | null>(null)
  const [meta, setMeta] = useState<InferenceMeta | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const requestSeq = useRef(0)

  const runInference = useCallback(
    async (variantId: string, state: AblationState, cache: boolean) => {
      if (!variantId || countActiveModules(state) === 0) return
      const seq = ++requestSeq.current // bỏ qua phản hồi cũ nếu người dùng đã chạy request mới hơn
      setIsLoading(true)
      setErrorMessage(null)
      try {
        const response = await predictSingleVariant({
          variant_id: variantId,
          ...ablationStateToFlags(state),
          use_cache: cache,
        })
        if (seq !== requestSeq.current) return
        const { meta: responseMeta, ...data } = response
        setCurrentResult(data)
        setMeta(responseMeta)
      } catch (error: unknown) {
        if (seq !== requestSeq.current) return
        setErrorMessage((error as Error)?.message || "An unexpected error occurred during prediction.")
        setCurrentResult(null)
        setMeta(null)
      } finally {
        if (seq === requestSeq.current) setIsLoading(false)
      }
    },
    []
  )

  const submit = (variantId: string, state: AblationState = ablation) => {
    const trimmed = variantId.trim()
    if (!trimmed) return
    setSubmittedVariant(trimmed)
    // Ghi biến thể + mask lên URL để nút Back từ trang XAI quay lại đúng biến thể này.
    const next = new URLSearchParams(searchParams)
    next.set("variant", trimmed)
    next.set("mask", String(ablationStateToMask(state)))
    setSearchParams(next, { replace: true })
    runInference(trimmed, state, useCache)
  }

  // Mở bằng liên kết (từ Batch Screening / nút Back của XAI): nạp kết quả ngay, ưu tiên cache PostgreSQL.
  useEffect(() => {
    const linked = searchParams.get("variant")
    if (linked) {
      setSubmittedVariant(linked)
      runInference(linked, ablation, true)
    }
    // chỉ chạy một lần khi trang được mở
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Đổi tổ hợp modality là thao tác chủ động -> chạy lại cho biến thể ĐÃ submit (không phải chuỗi đang gõ).
  const handleAblationChange = (next: AblationState) => {
    setAblation(next)
    if (submittedVariant) submit(submittedVariant, next)
  }

  const handleSelectPreset = (presetId: string) => {
    const found = mockData.presets.find((p) => p.id === presetId)
    if (found) {
      setActivePresetId(presetId)
      setVariantInput(found.variant_id) // chỉ điền ô nhập; bấm Run để chạy
    }
  }

  const threshold = meta?.decision_threshold ?? null
  const thresholdPct = threshold !== null ? (threshold * 100).toFixed(1) : null
  const resultMask = meta?.ablation_mask ?? ablationStateToMask(ablation)

  const openXai = () => {
    if (!currentResult) return
    const params = new URLSearchParams({ variant: currentResult.variant_id, mask: String(resultMask), from: "single" })
    if (openedFromBatch) params.set("origin", "batch")
    navigate(`/xai?${params.toString()}`)
  }

  const presetButton = (id: string, label: string, tone: "rose" | "emerald") => (
    <button
      type="button"
      onClick={() => handleSelectPreset(id)}
      className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-all cursor-pointer flex items-center gap-1.5 ${
        activePresetId === id
          ? tone === "rose"
            ? "bg-rose-50 text-rose-700 border-rose-300 font-semibold shadow-2xs"
            : "bg-emerald-50 text-emerald-700 border-emerald-300 font-semibold shadow-2xs"
          : `bg-white text-slate-600 border-slate-200 ${
              tone === "rose" ? "hover:bg-rose-50/50 hover:text-rose-600" : "hover:bg-emerald-50/50 hover:text-emerald-600"
            }`
      }`}
    >
      <span className={`h-2 w-2 rounded-full ${tone === "rose" ? "bg-rose-500" : "bg-emerald-500"}`}></span>
      <span>{label}</span>
    </button>
  )

  return (
    <div className="space-y-6 text-left">
      {/* Page Header */}
      <div className="space-y-1">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-blue-600 uppercase tracking-wider">
            <Dna className="h-4 w-4" />
            <span>{openedFromBatch ? "Batch Screening › Variant Detail" : "Single Variant Inspection Mode"}</span>
          </div>
          <div className="flex items-center gap-2">
            {openedFromBatch && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => navigate("/batch")}
                className="text-xs border-indigo-200 text-indigo-700 hover:bg-indigo-50"
              >
                <ArrowLeft className="h-3.5 w-3.5 mr-1.5" />
                Back to Batch Screening
              </Button>
            )}
            <Badge variant="outline" className="text-xs bg-emerald-50 text-emerald-700 border-emerald-200 flex items-center gap-1.5 shadow-2xs">
              <Database className="h-3.5 w-3.5 text-emerald-600" />
              <span>PostgreSQL Connected</span>
            </Badge>
          </div>
        </div>

        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">
          Single Variant Inspection &amp; Analysis
        </h1>
        <p className="text-sm text-slate-500">
          Evaluate variant pathogenicity via on-the-fly inference with feature ablation testing and ClinVar ground truth validation.
        </p>
      </div>

      {/* Dashboard 2 cột: điều khiển (trái) | kết quả (phải); màn hình nhỏ tự xếp 1 cột */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      {/* ===== Cột trái: nhập liệu + ablation ===== */}
      <div className="lg:col-span-5 space-y-6">
      {/* Input & Preset Quick Select Card */}
      <Card className="border-slate-200 shadow-sm overflow-hidden">
        <CardHeader className="bg-slate-50/70 border-b border-slate-100 pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">Genomic Coordinate Input (GRCh38)</CardTitle>
            <div className="flex items-center gap-2" title="Ignore the PostgreSQL prediction cache and recompute">
              <span className={`text-xs font-semibold ${!useCache ? "text-amber-600" : "text-slate-500"}`}>
                Live Bypass Mode
              </span>
              <Switch
                checked={!useCache}
                onCheckedChange={(checked) => setUseCache(!checked)}
                className={!useCache ? "bg-amber-500" : ""}
              />
            </div>
          </div>
        </CardHeader>

        <CardContent className="pt-6 space-y-5">
          <form
            className="flex flex-col sm:flex-row gap-3"
            onSubmit={(e) => {
              e.preventDefault()
              submit(variantInput)
            }}
          >
            <div className="relative flex-1">
              <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
              <input
                type="text"
                value={variantInput}
                onChange={(e) => {
                  setVariantInput(e.target.value)
                  setActivePresetId(null)
                }}
                placeholder="Enter variant ID (e.g., chr7:140753336:A>T) and press Enter"
                className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-slate-200 text-sm font-mono text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all bg-white"
              />
            </div>

            <Button
              type="submit"
              disabled={isLoading || !variantInput.trim() || countActiveModules(ablation) === 0}
              className="sm:w-44 bg-blue-600 hover:bg-blue-700 text-white font-medium flex items-center justify-center gap-2 shadow-md shadow-blue-500/15"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Computing...</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  <span>Run Inference</span>
                </>
              )}
            </Button>
          </form>

          {/* Quick Preset Selector Buttons (điền ô nhập, không tự chạy) */}
          <div className="pt-2 border-t border-slate-100 flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium text-slate-500 flex items-center gap-1">
              <Info className="h-3.5 w-3.5" />
              Quick Validation Presets:
            </span>
            {presetButton("pathogenic", "BRAF V600E (Pathogenic)", "rose")}
            {presetButton("brca1", "BRCA1 p.Arg1699Gln (Pathogenic)", "rose")}
            {presetButton("benign", "BRCA2 p.Asn372His (Benign)", "emerald")}
          </div>
        </CardContent>
      </Card>

      <FeatureAblationPanel value={ablation} onChange={handleAblationChange} orientation="vertical" />
      </div>

      {/* ===== Cột phải: kết quả (dính khi cuộn) ===== */}
      <div className="lg:col-span-7 space-y-6 lg:sticky lg:top-24">
      {errorMessage && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 flex items-start gap-3 shadow-sm">
          <ShieldAlert className="h-5 w-5 shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-bold">Inference Failed</h4>
            <p className="text-xs mt-1 leading-relaxed font-medium">{errorMessage}</p>
          </div>
        </div>
      )}

      {!currentResult && !errorMessage && (
        <div className="lg:min-h-[520px] p-8 rounded-2xl border-2 border-dashed border-slate-300 bg-white/70 flex flex-col items-center justify-center text-center gap-4">
          {isLoading ? (
            <>
              <Loader2 className="h-10 w-10 text-blue-500 animate-spin" />
              <p className="text-sm font-medium text-slate-600">Running RB-Net inference...</p>
            </>
          ) : (
            <>
              <div className="p-4 rounded-2xl bg-blue-50 text-blue-500">
                <Dna className="h-10 w-10" />
              </div>
              <div className="space-y-1.5 max-w-sm">
                <h3 className="text-base font-semibold text-slate-800">No prediction yet</h3>
                <p className="text-sm text-slate-500 leading-relaxed">
                  Enter a GRCh38 coordinate or pick a Quick Validation Preset, then press <strong>Run Inference</strong>{" "}
                  (or <kbd className="px-1.5 py-0.5 rounded border border-slate-300 bg-slate-50 text-xs font-mono">Enter</kbd>).
                </p>
              </div>
              <ul className="text-xs text-slate-400 space-y-1">
                <li>• Pathogenicity probability &amp; MCC-optimal threshold</li>
                <li>• ClinVar ground truth benchmark</li>
                <li>• SOTA comparison (AlphaMissense, REVEL)</li>
              </ul>
            </>
          )}
        </div>
      )}

      {/* Main Result Panel */}
      {currentResult && !errorMessage && (
        <div className={`space-y-6 transition-opacity duration-300 ${isLoading ? "opacity-40 pointer-events-none" : "opacity-100"}`}>
          <Card className="border-blue-300 bg-gradient-to-b from-blue-50/40 via-white to-white shadow-lg ring-1 ring-blue-500/20 overflow-hidden">
            <CardHeader className="pb-4 border-b border-blue-100/60 bg-blue-50/20">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-xl bg-blue-600 text-white shadow-md shadow-blue-500/20">
                    <Dna className="h-6 w-6" />
                  </div>
                  <div>
                    <CardTitle className="text-xl font-extrabold text-blue-950">RB-Net Hybrid Architecture</CardTitle>
                    <CardDescription className="text-xs text-blue-800/70 mt-0.5">
                      Router-expert fusion of DNA (NT-v1), protein (ESM2) and Bio + Geometry + SpliceAI features
                    </CardDescription>
                  </div>
                </div>

                <div className="flex items-center gap-2 self-start sm:self-auto">
                  {isLoading ? (
                    <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-300 flex items-center gap-1.5 py-1 text-xs">
                      <Loader2 className="h-3 w-3 animate-spin text-amber-600" />
                      <span>Status: Computing...</span>
                    </Badge>
                  ) : meta?.is_cache_hit ? (
                    <Badge variant="outline" className="bg-emerald-50 text-emerald-700 border-emerald-300 flex items-center gap-1.5 py-1 text-xs font-semibold">
                      <Zap className="h-3 w-3 text-emerald-600" />
                      <span>Cached Result ({meta.computation_time_ms}ms)</span>
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-300 flex items-center gap-1.5 py-1 text-xs font-semibold">
                      <Cpu className="h-3 w-3 text-blue-600" />
                      <span>Live Inference ({meta?.computation_time_ms ?? 0}ms)</span>
                    </Badge>
                  )}
                </div>
              </div>
            </CardHeader>

            <CardContent className="pt-6 space-y-6">
              {/* Variant Meta Information */}
              <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 text-xs">
                <div className="flex items-center gap-4 flex-wrap">
                  <div>
                    <span className="text-slate-400 block font-medium">Genomic Coordinate:</span>
                    <strong className="font-mono text-sm text-slate-800">{currentResult.variant_id}</strong>
                  </div>
                  {currentResult.gene && (
                    <div className="border-l border-slate-200 pl-4">
                      <span className="text-slate-400 block font-medium">Gene Symbol:</span>
                      <strong className="text-sm text-blue-700 font-semibold">{currentResult.gene}</strong>
                    </div>
                  )}
                  {currentResult.consequence && (
                    <div className="border-l border-slate-200 pl-4">
                      <span className="text-slate-400 block font-medium">Protein Change:</span>
                      <strong className="text-sm text-slate-800 font-mono">{currentResult.consequence}</strong>
                    </div>
                  )}
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={openXai}
                  className="text-xs text-blue-600 border-blue-200 hover:bg-blue-50 flex items-center gap-1.5"
                >
                  <BarChart3 className="h-3.5 w-3.5" />
                  <span>Inspect SHAP Attribution</span>
                </Button>
              </div>

              {/* Prediction vs Ground Truth */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center p-6 rounded-2xl bg-white border border-slate-200 shadow-xs">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">RB-Net Prediction</span>
                    <Badge
                      variant={currentResult.results.rb_net.label === "Pathogenic" ? "destructive" : "success"}
                      className="text-xs px-3 py-1 font-bold"
                    >
                      {currentResult.results.rb_net.label}
                    </Badge>
                  </div>

                  <div className="flex items-baseline gap-2">
                    <span className={`text-4xl sm:text-5xl font-extrabold font-mono ${
                      currentResult.results.rb_net.label === "Pathogenic" ? "text-rose-600" : "text-emerald-600"
                    }`}>
                      {(currentResult.results.rb_net.probability * 100).toFixed(1)}%
                    </span>
                    <span className="text-xs text-slate-500 font-medium">Pathogenicity Probability</span>
                  </div>

                  {/* Thanh xác suất + vạch ngưỡng */}
                  <div className="relative w-full h-3 rounded-full bg-slate-100">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        currentResult.results.rb_net.label === "Pathogenic" ? "bg-rose-500" : "bg-emerald-500"
                      }`}
                      style={{ width: `${currentResult.results.rb_net.probability * 100}%` }}
                    />
                    {threshold !== null && (
                      <div
                        className="absolute -top-1 -bottom-1 w-0.5 bg-slate-900"
                        style={{ left: `${threshold * 100}%` }}
                        title={`Decision threshold ${thresholdPct}%`}
                      />
                    )}
                  </div>

                  {/* Ngưỡng phân loại (lấy động từ checkpoint) */}
                  <div className="flex items-start gap-2.5 p-3 rounded-lg bg-slate-900 text-white">
                    <Scale className="h-5 w-5 shrink-0 mt-0.5 text-amber-300" />
                    {thresholdPct !== null ? (
                      <div className="space-y-0.5">
                        <p className="text-sm font-bold">
                          Classification Threshold: &ge; {thresholdPct}% &rarr; <span className="text-rose-300">Pathogenic</span>
                        </p>
                        <p className="text-xs text-slate-300">
                          &lt; {thresholdPct}% &rarr; Benign
                          {meta?.checkpoint ? ` · MCC-optimal threshold of ${meta.checkpoint}` : ""}
                        </p>
                      </div>
                    ) : (
                      <p className="text-sm font-semibold">Classification threshold not reported by the backend.</p>
                    )}
                  </div>
                </div>

                {/* ClinVar Ground Truth */}
                <div className="p-5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                    <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">ClinVar Ground Truth Benchmark</span>
                  </div>

                  <div className="flex items-center justify-between pt-1">
                    <span className="text-xs text-slate-600 font-medium">Official ClinVar Validation:</span>
                    <Badge variant={clinvarBadgeVariant(currentResult.clinvar_ground_truth)} className="text-xs px-3 py-0.5 font-bold">
                      {currentResult.clinvar_ground_truth}
                    </Badge>
                  </div>

                  <div className="pt-2 border-t border-slate-200/60 text-xs">
                    {currentResult.clinvar_ground_truth === "Uncertain" ? (
                      <span className="text-slate-600 font-semibold flex items-center gap-1.5">
                        <Info className="h-4 w-4 text-slate-500 shrink-0" />
                        <span>ClinVar has no definitive label (VUS / conflicting); no benchmark comparison.</span>
                      </span>
                    ) : currentResult.results.rb_net.label === currentResult.clinvar_ground_truth ? (
                      <span className="text-emerald-700 font-semibold flex items-center gap-1.5">
                        <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0" />
                        <span>Consistent: RB-Net prediction matches ClinVar Ground Truth.</span>
                      </span>
                    ) : (
                      <span className="text-amber-700 font-semibold flex items-center gap-1.5">
                        <ShieldAlert className="h-4 w-4 text-amber-600 shrink-0" />
                        <span>Discrepancy observed under active feature ablation state.</span>
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
      </div>
      </div>

      {/* ===== Hàng dưới (rộng hết): so sánh SOTA — bảng | biểu đồ ===== */}
      {currentResult && !errorMessage && (
        <div className={`transition-opacity duration-300 ${isLoading ? "opacity-40 pointer-events-none" : "opacity-100"}`}>
          <Card className="border-slate-200 shadow-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center justify-between">
                <span>State-of-the-Art (SOTA) Model Comparison</span>
                <span className="text-xs font-normal text-slate-500">Probability Scale [0.0 - 1.0]</span>
              </CardTitle>
              <CardDescription className="text-xs">
                Comparative validation of RB-Net against Google DeepMind AlphaMissense and REVEL.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
              <div className="lg:col-span-7">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[32%]">Model / Algorithm</TableHead>
                    <TableHead className="w-[22%]">Score (Probability)</TableHead>
                    <TableHead className="w-[26%]">Classification</TableHead>
                    <TableHead className="w-[20%] text-right">Alignment</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow className="bg-blue-50/50 font-semibold">
                    <TableCell className="flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-blue-600"></span>
                      <span>RB-Net Hybrid (Proposed)</span>
                    </TableCell>
                    <TableCell className="font-mono text-blue-700 font-bold">
                      {currentResult.results.rb_net.probability.toFixed(3)}
                    </TableCell>
                    <TableCell>
                      <Badge variant={currentResult.results.rb_net.label === "Pathogenic" ? "destructive" : "success"}>
                        {currentResult.results.rb_net.label}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right text-blue-700 font-medium">Anchor Model</TableCell>
                  </TableRow>
                  <SotaRow
                    label="AlphaMissense (Google DeepMind)"
                    score={currentResult.sota.alphamissense}
                    threshold={ALPHAMISSENSE_THRESHOLD}
                    rbNetProbability={currentResult.results.rb_net.probability}
                  />
                  <SotaRow
                    label="REVEL (Ensemble Predictor)"
                    score={currentResult.sota.revel}
                    threshold={REVEL_THRESHOLD}
                    rbNetProbability={currentResult.results.rb_net.probability}
                  />
                </TableBody>
              </Table>
              </div>
              <div className="lg:col-span-5 lg:border-l lg:border-slate-100 lg:pl-8">
                <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4">Score Comparison</p>
                <SotaScoreChart
                  rows={[
                    { label: "RB-Net Hybrid", score: currentResult.results.rb_net.probability, threshold, anchor: true },
                    { label: "AlphaMissense", score: currentResult.sota.alphamissense, threshold: ALPHAMISSENSE_THRESHOLD },
                    { label: "REVEL", score: currentResult.sota.revel, threshold: REVEL_THRESHOLD },
                  ]}
                />
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
