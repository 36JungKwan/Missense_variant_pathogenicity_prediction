import React, { useState, useEffect, useRef, useMemo } from "react"
import { useNavigate } from "react-router-dom"
import {
  Play,
  Square,
  Activity,
  AlertTriangle,
  FileDigit,
  Database,
  ShieldAlert,
  CheckCircle2,
  ListFilter,
  BarChart3,
  Download,
  UploadCloud,
  ChevronRight,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { FeatureAblationPanel } from "@/components/ablation/FeatureAblationPanel"
import { ablationStateToMask, countActiveModules, normalizeAblationState } from "@/lib/ablation"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

import { BatchVariant, AblationState } from "@/types"
import { createVcfStreamService, BatchStreamEvent } from "@/services/api"

const STORAGE_KEY = "trimvpp_batch_state"
const ACCEPTED_EXTENSIONS = [".vcf", ".csv", ".txt", ".parquet", ".xlsx"]
const ACCEPT_ATTR = ACCEPTED_EXTENSIONS.join(",")

export const Batch: React.FC = () => {
  const navigate = useNavigate()

  // --- 1. Load Initial State from SessionStorage (Giữ dữ liệu khi Back lại) ---
  const loadSavedState = () => {
    try {
      const saved = sessionStorage.getItem(STORAGE_KEY)
      if (saved) return JSON.parse(saved)
    } catch (e) {
      console.error("Failed to parse session storage", e)
    }
    return null
  }
  const savedState = loadSavedState()

  // --- 2. States ---
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isStreaming, setIsStreaming] = useState<boolean>(false)
  const [progress, setProgress] = useState<number>(savedState?.progress || 0)
  const [streamError, setStreamError] = useState<string | null>(null)
  
  const [stats, setStats] = useState<{
    totalProcessed: number;
    missenseAccepted: number;
    nonMissenseRejected: number;
  }>(savedState?.stats || {
    totalProcessed: 0,
    missenseAccepted: 0,
    nonMissenseRejected: 0
  })

  const [streamedVariants, setStreamedVariants] = useState<BatchVariant[]>(savedState?.streamedVariants || [])
  const [dangerQueue, setDangerQueue] = useState<BatchVariant[]>(savedState?.dangerQueue || [])

  // Ablation: 3 nhóm (DNA, Protein, Bio+Geom+SpliceAI); chuyển đổi tự động từ trạng thái 4 công tắc cũ.
  const [ablation, setAblation] = useState<AblationState>(normalizeAblationState(savedState?.ablation))
  // Mask của lần chạy gần nhất: dùng khi mở chi tiết/XAI để đọc đúng bản ghi cache trong PostgreSQL.
  const [runMask, setRunMask] = useState<number>(
    typeof savedState?.runMask === "number" ? savedState.runMask : ablationStateToMask(ablation)
  )
  const [isDragging, setIsDragging] = useState<boolean>(false)

  const streamServiceRef = useRef<{ cancel: () => void } | null>(null)

  // --- 3. Lưu State vào SessionStorage mỗi khi có thay đổi ---
  useEffect(() => {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
      progress,
      stats,
      streamedVariants,
      dangerQueue,
      ablation,
      runMask,
    }))
  }, [progress, stats, streamedVariants, dangerQueue, ablation, runMask])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamServiceRef.current) {
        streamServiceRef.current.cancel()
      }
    }
  }, [])

  const activeToggleCount = useMemo(() => countActiveModules(ablation), [ablation])

  // Chọn / kéo thả file mới -> kiểm tra định dạng rồi reset toàn bộ bảng dữ liệu
  const selectFile = (file: File | null) => {
    if (file && !ACCEPTED_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext))) {
      setStreamError(`Unsupported file "${file.name}". Accepted formats: ${ACCEPTED_EXTENSIONS.join(", ")}.`)
      return
    }
    setSelectedFile(file)
    setStreamError(null)
    setStreamedVariants([])
    setDangerQueue([])
    setProgress(0)
    setStats({ totalProcessed: 0, missenseAccepted: 0, nonMissenseRejected: 0 })
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => selectFile(e.target.files?.[0] || null)

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
    if (!isStreaming) selectFile(e.dataTransfer.files?.[0] || null)
  }

  // Mở chi tiết một biến thể (giao diện Single Variant, đọc cache PostgreSQL của lần chạy batch)
  const openVariantDetail = (variantId: string) => {
    navigate(`/single?${new URLSearchParams({ variant: variantId, mask: String(runMask), from: "batch" })}`)
  }

  const openVariantXai = (variantId: string) => {
    navigate(`/xai?${new URLSearchParams({ variant: variantId, mask: String(runMask), from: "batch" })}`)
  }

  // --- 4. Stream Handlers ---
  const startStream = () => {
    if (activeToggleCount === 0) {
      setStreamError("At least one feature module must remain enabled before starting screening.")
      return
    }

    // Luôn dọn sạch bảng kết quả cũ trước khi bắt đầu một phiên stream mới
    setStreamedVariants([])
    setDangerQueue([])
    setProgress(0)
    setStats({ totalProcessed: 0, missenseAccepted: 0, nonMissenseRejected: 0 })
    setStreamError(null)
    setIsStreaming(true)
    setRunMask(ablationStateToMask(ablation))

    const service = createVcfStreamService(
      selectedFile, 
      ablation, 
      handleStreamEvent, 
      150
    )
    streamServiceRef.current = service
  }

  const stopStream = () => {
    if (streamServiceRef.current) {
      streamServiceRef.current.cancel()
      streamServiceRef.current = null
    }
    setIsStreaming(false)
  }

  const handleStreamEvent = (event: BatchStreamEvent) => {
    setProgress(event.progress_percent)

    if (event.type === "variant_stream" && event.variant) {
      const newVariant = { ...event.variant, isNewStreamItem: true }
      
      setStreamedVariants((prev) => {
        const updatedPrev = prev.map(v => ({ ...v, isNewStreamItem: false }))
        return [newVariant, ...updatedPrev].slice(0, 500)
      })

      if (newVariant.classification === "Pathogenic") {
        setDangerQueue((prev) => {
          const newQueue = [...prev, newVariant].sort((a, b) => b.trimvpp_probability - a.trimvpp_probability)
          return newQueue.slice(0, 5)
        })
      }

      setStats(prev => ({
        ...prev,
        totalProcessed: prev.totalProcessed + 1,
        missenseAccepted: prev.missenseAccepted + 1
      }))

    } else if (event.type === "variant_skipped") {
      setStats(prev => ({
        ...prev,
        totalProcessed: prev.totalProcessed + 1,
        nonMissenseRejected: prev.nonMissenseRejected + 1
      }))

    } else if (event.type === "stream_error") {
      setIsStreaming(false)
      setStreamError(event.reason || "Batch stream encountered an error.")

    } else if (event.type === "stream_completed") {
      setIsStreaming(false)
      setProgress(100)
    }
  }

  // --- 5. Xuất file CSV ---
  const exportToCSV = () => {
    if (streamedVariants.length === 0) return

    const headers = ["ID", "Variant", "Gene", "Consequence", "Probability", "Classification", "ClinVar_GT", "AlphaMissense"]
    const rows = streamedVariants.map(v => [
      v.id,
      v.variant_id,
      v.gene || "N/A",
      `"${v.consequence || "N/A"}"`,
      v.trimvpp_probability,
      v.classification,
      v.clinvar_ground_truth,
      v.sota_alphamissense ?? "N/A"
    ])

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n")
    const encodedUri = encodeURI(csvContent)
    
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", `TriMVPP_Batch_Results_${new Date().getTime()}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="space-y-6 text-left pb-10">
      {/* Header */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-semibold text-indigo-600 uppercase tracking-wider">
            <ListFilter className="h-4 w-4" />
            <span>High-Throughput VCF Screening</span>
          </div>
          <Badge variant="outline" className="text-xs bg-emerald-50 text-emerald-700 border-emerald-200 flex items-center gap-1.5 shadow-2xs">
            <Database className="h-3.5 w-3.5 text-emerald-600" />
            <span>PostgreSQL SSE Stream Active</span>
          </Badge>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">
          Batch Variant Processing
        </h1>
        <p className="text-sm text-slate-500">
          Live streaming inference with automatic Gatekeeper filtering for non-missense variants.
        </p>
      </div>

      {/* Dashboard 2 cột: điều khiển (trái) | tiến trình + kết quả (phải); màn hình nhỏ tự xếp 1 cột */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      {/* ===== Cột trái: điều khiển pipeline + ablation ===== */}
      <div className="lg:col-span-4 space-y-6 lg:sticky lg:top-24">
      <Card className="border-slate-200 shadow-sm overflow-hidden">
        <CardContent className="p-0">
            <div className="p-6 bg-slate-50/50 flex flex-col justify-center gap-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-800">Pipeline Control</h3>
                <div
                  className={`mt-2 p-3 rounded-lg border-2 border-dashed transition-colors ${
                    isDragging ? "border-indigo-400 bg-indigo-50" : "border-slate-200 bg-white"
                  }`}
                  onDragOver={(e) => {
                    e.preventDefault()
                    if (!isStreaming) setIsDragging(true)
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                >
                  <div className="flex items-center gap-2 text-xs text-slate-600 mb-2">
                    <UploadCloud className="h-4 w-4 text-indigo-500" />
                    <span>Drop a file here or browse</span>
                  </div>
                  <input
                    type="file"
                    accept={ACCEPT_ATTR}
                    onChange={handleFileChange}
                    disabled={isStreaming}
                    className="text-xs w-full file:mr-2 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:bg-indigo-50 file:text-indigo-700 file:font-semibold hover:file:bg-indigo-100 text-slate-500 border border-slate-200 rounded-md cursor-pointer disabled:opacity-50 bg-white"
                  />
                  <p className="text-[11px] text-slate-500 mt-1.5">
                    VCF / CSV / TXT / Parquet / Excel (.xlsx) — columns CHROM, POS, REF, ALT (any case) or chrN:POS:REF&gt;ALT ids.
                  </p>
                </div>
                {selectedFile && (
                  <p className="text-[10px] text-slate-500 mt-1.5 font-mono">
                    Ready: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
                  </p>
                )}
                {!selectedFile && (
                  <p className="text-[10px] text-slate-500 mt-1.5 font-mono">
                    Defaulting to: demo_cohort.vcf
                  </p>
                )}
              </div>

              <div className="flex items-center gap-2">
                {!isStreaming ? (
                  <Button
                    onClick={startStream}
                    disabled={activeToggleCount === 0}
                    className="w-full bg-indigo-600 hover:bg-indigo-700 text-white shadow-md shadow-indigo-500/20 disabled:opacity-50"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    Start Screening
                  </Button>
                ) : (
                  <Button
                    onClick={stopStream}
                    variant="destructive"
                    className="w-full shadow-md"
                  >
                    <Square className="h-4 w-4 mr-2 fill-current" />
                    Stop Process
                  </Button>
                )}
              </div>
            </div>
        </CardContent>
      </Card>

      <FeatureAblationPanel
        value={ablation}
        onChange={setAblation}
        disabled={isStreaming}
        title="Feature Ablation Selection"
        orientation="vertical"
      />
      </div>

      {/* ===== Cột phải: tiến trình, thống kê, kết quả stream ===== */}
      <div className="lg:col-span-8 space-y-6">
      <Card className="border-slate-200 shadow-sm overflow-hidden">
        <CardContent className="p-0">
            <div className="p-6 space-y-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className={`h-5 w-5 ${isStreaming ? "text-indigo-600 animate-pulse" : "text-slate-400"}`} />
                  <span className="font-semibold text-slate-700">Stream Progress</span>
                </div>
                <span className="text-xl font-bold font-mono text-indigo-700">{progress}%</span>
              </div>
              <Progress value={progress} className="h-3 bg-slate-100" />
              
              <div className="grid grid-cols-3 gap-4 pt-2">
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/60">
                  <div className="text-xs text-slate-500 font-medium mb-1 flex items-center gap-1.5">
                    <FileDigit className="h-3.5 w-3.5" /> Processed
                  </div>
                  <div className="text-xl font-bold font-mono text-slate-800">{stats.totalProcessed}</div>
                </div>
                <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200/60">
                  <div className="text-xs text-emerald-700 font-medium mb-1 flex items-center gap-1.5">
                    <CheckCircle2 className="h-3.5 w-3.5" /> Missense Valid
                  </div>
                  <div className="text-xl font-bold font-mono text-emerald-700">{stats.missenseAccepted}</div>
                </div>
                <div className="p-3 rounded-lg bg-rose-50 border border-rose-200/60">
                  <div className="text-xs text-rose-700 font-medium mb-1 flex items-center gap-1.5">
                    <ShieldAlert className="h-3.5 w-3.5" /> Gatekeeper Rejected
                  </div>
                  <div className="text-xl font-bold font-mono text-rose-700">{stats.nonMissenseRejected}</div>
                </div>
              </div>
            </div>
        </CardContent>
      </Card>

      {/* Error Alert Banner */}
      {streamError && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 flex items-start gap-3 shadow-sm">
          <ShieldAlert className="h-5 w-5 shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-bold">Batch Screening Error</h4>
            <p className="text-xs mt-1 leading-relaxed font-medium">{streamError}</p>
          </div>
        </div>
      )}

      {/* Bảng stream | hàng đợi ưu tiên */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">

        {/* Live Stream Table */}
        <Card className="xl:col-span-3 border-slate-200 shadow-sm flex flex-col h-[500px]">
          <CardHeader className="py-4 border-b border-slate-100 bg-slate-50/50">
            <CardTitle className="text-sm flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Database className="h-4 w-4 text-slate-500" />
                Accepted Missense Variants
                <span className="text-xs font-normal text-slate-500">(click a row for full detail)</span>
              </span>
              <div className="flex gap-2 items-center">
                <Badge variant="secondary" className="font-mono text-xs">
                  {streamedVariants.length} Items
                </Badge>
                <Button 
                  onClick={exportToCSV}
                  disabled={streamedVariants.length === 0}
                  size="sm" 
                  variant="outline" 
                  className="h-7 text-xs border-indigo-200 text-indigo-700 hover:bg-indigo-50"
                >
                  <Download className="h-3 w-3 mr-1" /> Export CSV
                </Button>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0 overflow-auto flex-1 relative">
            {!isStreaming && streamedVariants.length === 0 ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400">
                <Database className="h-10 w-10 mb-3 opacity-20" />
                <p className="text-sm">Click "Start Screening" to begin stream</p>
              </div>
            ) : (
              <Table>
                <TableHeader className="sticky top-0 bg-white/95 backdrop-blur shadow-sm z-10">
                  <TableRow>
                    <TableHead className="w-[80px]">ID</TableHead>
                    <TableHead>Variant</TableHead>
                    <TableHead>Gene</TableHead>
                    <TableHead className="text-right">Probability</TableHead>
                    <TableHead className="w-[32px]"></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {streamedVariants.map((v, i) => (
                    <TableRow
                      key={`${v.variant_id}-${i}`}
                      role="button"
                      tabIndex={0}
                      title="Open full variant detail"
                      onClick={() => openVariantDetail(v.variant_id)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault()
                          openVariantDetail(v.variant_id)
                        }
                      }}
                      className={`group cursor-pointer transition-colors duration-500 hover:bg-indigo-50 focus:outline-none focus-visible:bg-indigo-50 ${
                        v.isNewStreamItem ? "bg-indigo-50/50" : ""
                      }`}
                    >
                      <TableCell className="font-mono text-xs text-slate-500">#{v.id}</TableCell>
                      <TableCell className="font-medium text-xs text-indigo-700 group-hover:underline">{v.variant_id}</TableCell>
                      <TableCell className="text-xs">{v.gene}</TableCell>
                      <TableCell className="text-right">
                        <Badge
                          variant={v.classification === "Pathogenic" ? "destructive" : "success"}
                          className="font-mono text-xs"
                        >
                          {(v.trimvpp_probability * 100).toFixed(1)}%
                        </Badge>
                      </TableCell>
                      <TableCell className="text-slate-300 group-hover:text-indigo-500">
                        <ChevronRight className="h-4 w-4" />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        {/* Right Col: High-Risk Queue */}
        <Card className="xl:col-span-2 border-rose-200 shadow-sm flex flex-col h-[500px] overflow-hidden">
          <CardHeader className="py-4 border-b border-rose-100 bg-rose-50">
            <CardTitle className="text-sm flex items-center text-rose-800">
              <AlertTriangle className="h-4 w-4 mr-2" />
              Top Pathogenic Triage
            </CardTitle>
            <CardDescription className="text-xs text-rose-600/80">
              Auto-sorted by TriMVPP probability score.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0 overflow-y-auto bg-white flex-1 relative">
            {dangerQueue.length === 0 ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 p-6 text-center">
                <AlertTriangle className="h-8 w-8 mb-2 opacity-20" />
                <p className="text-xs">No pathogenic variants detected yet.</p>
              </div>
            ) : (
              <div className="divide-y divide-slate-100">
                {dangerQueue.map((v, i) => {
                  return (
                    <div key={`danger-${v.variant_id}-${i}`} className="p-4 hover:bg-slate-50 transition-colors">
                      <div className="flex items-center justify-between mb-2">
                        <button
                          type="button"
                          onClick={() => openVariantDetail(v.variant_id)}
                          className="font-mono text-xs font-bold text-slate-800 hover:text-indigo-700 hover:underline cursor-pointer"
                          title="Open full variant detail"
                        >
                          {v.variant_id}
                        </button>
                        <Badge variant="destructive" className="font-mono text-[10px] px-1.5 py-0">
                          {(v.trimvpp_probability * 100).toFixed(1)}%
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-500 mb-3">
                        <span>Gene: <strong className="text-slate-700">{v.gene}</strong></span>
                        <span>•</span>
                        <span className="truncate">{v.consequence}</span>
                      </div>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="w-full text-xs h-7 text-indigo-600 border-indigo-200 hover:bg-indigo-50"
                        onClick={() => openVariantXai(v.variant_id)}
                      >
                        <BarChart3 className="h-3 w-3 mr-1.5" />
                        View Explainability
                      </Button>
                    </div>
                  )
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      </div>
      </div>
    </div>
  )
}