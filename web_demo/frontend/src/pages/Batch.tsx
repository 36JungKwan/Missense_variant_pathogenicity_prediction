import React, { useState, useEffect, useRef, useMemo } from "react"
import { useBlocker, useNavigate } from "react-router-dom"
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
  Loader2,
  Timer,
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

import { BatchVariant, AblationState, RejectedVariant } from "@/types"
import { createVcfStreamService, BatchStreamEvent } from "@/services/api"

const STORAGE_KEY = "trimvpp_batch_state"
const ACCEPTED_EXTENSIONS = [".vcf", ".csv", ".txt", ".parquet", ".xlsx"]
const ACCEPT_ATTR = ACCEPTED_EXTENSIONS.join(",")
/** Bảng chỉ vẽ N dòng mới nhất cho nhẹ; CSV xuất đủ toàn bộ kết quả. */
const TABLE_ROW_LIMIT = 500
const LEAVE_WARNING = "Batch screening is still running. Leaving this page will stop it. Leave anyway?"

const currentTimeMs = () => Date.now()

function formatDuration(ms: number): string {
  const total = Math.max(0, Math.round(ms / 1000))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const pad = (n: number) => String(n).padStart(2, "0")
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`
}

/** Lưu trạng thái trang vào sessionStorage; vượt dung lượng thì chỉ giữ các dòng mới nhất. */
function saveSnapshot(snapshot: { streamedVariants: BatchVariant[] } & Record<string, unknown>): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot))
  } catch {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
        ...snapshot, streamedVariants: snapshot.streamedVariants.slice(0, TABLE_ROW_LIMIT),
      }))
    } catch {
      // bộ nhớ bị chặn: bỏ qua, trang vẫn chạy bình thường
    }
  }
}

/** Mốc thời gian theo khối do backend báo (stream_status), dùng để ước tính thời gian còn lại. */
interface ChunkTimeline {
  chunkSize: number
  totalChunks: number
  totalVariants: number
  /** thời điểm bắt đầu phân loại của từng khối (1-based) */
  classifyStart: Record<number, number>
  /** khối mà GPU phải chờ VEP + SpliceAI xong mới chạy được */
  waited: Record<number, boolean>
}

/**
 * Ước tính thời gian còn lại theo đúng cách backend chạy: khối k+1 được chú giải (CPU) trong lúc khối k được
 * phân loại (GPU), nên mỗi khối bắt đầu phân loại khi cả hai việc đó xong. Tốc độ chú giải lấy từ khối đầu
 * (và các khối phải chờ), tốc độ phân loại lấy từ khối đang chạy.
 */
function estimateRemainingMs(tl: ChunkTimeline, startedAt: number, now: number, processed: number): number | null {
  const started = Object.keys(tl.classifyStart).map(Number).sort((a, b) => a - b)
  if (started.length === 0) return null
  const size = (k: number) => Math.max(0, Math.min(tl.chunkSize, tl.totalVariants - (k - 1) * tl.chunkSize))
  const current = started[started.length - 1]

  let annotatePerVariant = started[0] === 1 ? (tl.classifyStart[1] - startedAt) / Math.max(1, size(1)) : 0
  for (const k of started) {
    if (k > 1 && tl.waited[k] && tl.classifyStart[k - 1] !== undefined) {
      annotatePerVariant = Math.max(annotatePerVariant, (tl.classifyStart[k] - tl.classifyStart[k - 1]) / Math.max(1, size(k)))
    }
  }

  const doneInCurrent = processed - (current - 1) * tl.chunkSize
  let classifyPerVariant: number | null = null
  if (doneInCurrent > 0) {
    classifyPerVariant = (now - tl.classifyStart[current]) / doneInCurrent
  } else if (tl.classifyStart[current - 1] !== undefined) {
    classifyPerVariant = (tl.classifyStart[current] - tl.classifyStart[current - 1]) / Math.max(1, size(current - 1))
  }
  if (classifyPerVariant === null) return null

  let classifyBegin = tl.classifyStart[current]
  let classifyEnd = classifyBegin + classifyPerVariant * size(current)
  for (let k = current + 1; k <= tl.totalChunks; k++) {
    classifyBegin = Math.max(classifyBegin + annotatePerVariant * size(k), classifyEnd)
    classifyEnd = classifyBegin + classifyPerVariant * size(k)
  }
  return Math.max(0, classifyEnd - now)
}

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
  const [rejectedVariants, setRejectedVariants] = useState<RejectedVariant[]>(savedState?.rejectedVariants || [])
  const [timeline, setTimeline] = useState<ChunkTimeline | null>(null)

  // Ablation: 3 nhóm (DNA, Protein, Bio+Geom+SpliceAI); chuyển đổi tự động từ trạng thái 4 công tắc cũ.
  const [ablation, setAblation] = useState<AblationState>(normalizeAblationState(savedState?.ablation))
  // Mask của lần chạy gần nhất: dùng khi mở chi tiết/XAI để đọc đúng bản ghi cache trong PostgreSQL.
  const [runMask, setRunMask] = useState<number>(
    typeof savedState?.runMask === "number" ? savedState.runMask : ablationStateToMask(ablation)
  )
  const [isDragging, setIsDragging] = useState<boolean>(false)
  // Trạng thái theo khối do backend gửi (stream_status) + đồng hồ để hiện thời gian đã chạy / còn lại.
  const [statusText, setStatusText] = useState<string | null>(null)
  const [totalVariants, setTotalVariants] = useState<number | null>(null)
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [now, setNow] = useState<number>(currentTimeMs)
  const [newestId, setNewestId] = useState<number | null>(null)

  const streamServiceRef = useRef<{ cancel: () => void } | null>(null)
  // Trạng thái mới nhất, để ghi ngay khi rời trang (bản ghi gộp bên dưới có thể chưa kịp chạy).
  const latestSnapshotRef = useRef<Parameters<typeof saveSnapshot>[0] | null>(null)

  // --- 3. Lưu State vào SessionStorage (gộp ghi khi đang stream để không serialize hàng nghìn dòng mỗi sự kiện) ---
  useEffect(() => {
    const snapshot = { progress, stats, streamedVariants, dangerQueue, rejectedVariants, ablation, runMask }
    latestSnapshotRef.current = snapshot
    if (!isStreaming) {
      saveSnapshot(snapshot)
      return
    }
    const timer = window.setTimeout(() => saveSnapshot(snapshot), 1500)
    return () => window.clearTimeout(timer)
  }, [progress, stats, streamedVariants, dangerQueue, rejectedVariants, ablation, runMask, isStreaming])

  // Đồng hồ thời gian đã chạy
  useEffect(() => {
    if (!isStreaming) return
    const timer = window.setInterval(() => setNow(currentTimeMs()), 1000)
    return () => window.clearInterval(timer)
  }, [isStreaming])

  // Chặn rời trang khi đang chạy: điều hướng trong web (menu, nút Back, mở chi tiết) và đóng tab / tải lại.
  const blocker = useBlocker(
    ({ currentLocation, nextLocation }) => isStreaming && currentLocation.pathname !== nextLocation.pathname
  )
  useEffect(() => {
    if (blocker.state !== "blocked") return
    // Đồng ý rời trang -> trang Batch unmount, cleanup bên dưới huỷ stream.
    if (window.confirm(LEAVE_WARNING)) {
      blocker.proceed()
    } else {
      blocker.reset()
    }
  }, [blocker])

  useEffect(() => {
    if (!isStreaming) return
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault()
      e.returnValue = ""
    }
    window.addEventListener("beforeunload", handler)
    return () => window.removeEventListener("beforeunload", handler)
  }, [isStreaming])

  // Rời trang: huỷ stream và ghi ngay kết quả đã nhận để khi quay lại vẫn còn.
  useEffect(() => {
    const service = streamServiceRef
    const latest = latestSnapshotRef
    return () => {
      if (service.current) {
        service.current.cancel()
      }
      if (latest.current) saveSnapshot(latest.current)
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
    setStatusText(null)
    setStreamedVariants([])
    setDangerQueue([])
    setRejectedVariants([])
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
    setRejectedVariants([])
    setProgress(0)
    setStats({ totalProcessed: 0, missenseAccepted: 0, nonMissenseRejected: 0 })
    setStreamError(null)
    setStatusText("Uploading and parsing the variant list...")
    setTotalVariants(null)
    setTimeline(null)
    setNewestId(null)
    const startTime = currentTimeMs()
    setStartedAt(startTime)
    setNow(startTime)
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
    setStatusText("Stopped by user.")
  }

  const handleStreamEvent = (event: BatchStreamEvent) => {
    // Lỗi không phải là hoàn tất: giữ nguyên tiến độ đã đạt được.
    if (event.type !== "stream_error") setProgress(event.progress_percent)

    if (event.type === "stream_status") {
      if (event.status) setStatusText(event.status)
      if (event.total_variants) setTotalVariants(event.total_variants)
      const { chunk, total_chunks, total_variants, chunk_size, stage } = event
      if (chunk && total_chunks && total_variants && chunk_size) {
        const at = currentTimeMs()
        setTimeline((prev) => {
          const base: ChunkTimeline = prev ?? {
            chunkSize: chunk_size, totalChunks: total_chunks, totalVariants: total_variants, classifyStart: {}, waited: {},
          }
          return stage === "classifying"
            ? { ...base, classifyStart: { ...base.classifyStart, [chunk]: at } }
            : { ...base, waited: { ...base.waited, [chunk]: true } }
        })
      }

    } else if (event.type === "variant_stream" && event.variant) {
      const newVariant = event.variant
      // Giữ đủ mọi kết quả (cho CSV); bảng chỉ vẽ TABLE_ROW_LIMIT dòng mới nhất.
      setStreamedVariants((prev) => [newVariant, ...prev])
      setNewestId(newVariant.id)

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
      if (event.variant_id) {
        const rejected: RejectedVariant = {
          variant_id: event.variant_id,
          consequence: event.detected_consequence || "unknown",
          reason: event.reason || "Rejected",
        }
        setRejectedVariants((prev) => [...prev, rejected])
      }
      setStats(prev => ({
        ...prev,
        totalProcessed: prev.totalProcessed + 1,
        nonMissenseRejected: prev.nonMissenseRejected + 1
      }))

    } else if (event.type === "stream_error") {
      setIsStreaming(false)
      setStatusText(null)
      setStreamError(event.reason || "Batch stream encountered an error.")

    } else if (event.type === "stream_completed") {
      setIsStreaming(false)
      setProgress(100)
      setStatusText(null)
    }
  }

  // --- 5. Xuất file CSV ---
  const exportToCSV = () => {
    if (streamedVariants.length === 0 && rejectedVariants.length === 0) return

    const quote = (s: string) => `"${s.replace(/"/g, '""')}"`
    const headers = ["ID", "Variant", "Gene", "Consequence", "Probability", "Classification", "ClinVar_GT", "AlphaMissense", "Rejection_Reason"]
    const rows = [
      ...streamedVariants.map(v => [
        v.id,
        v.variant_id,
        v.gene || "N/A",
        quote(v.consequence || "N/A"),
        v.trimvpp_probability,
        v.classification,
        v.clinvar_ground_truth,
        v.sota_alphamissense ?? "N/A",
        "",
      ]),
      // Biến thể bị loại: không có xác suất, ghi lý do để người dùng biết cần sửa gì.
      ...rejectedVariants.map(r => ["", r.variant_id, "", quote(r.consequence), "", "Rejected", "", "", quote(r.reason)]),
    ]

    const csv = [headers.join(","), ...rows.map(e => e.join(","))].join("\n")
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }))
    const link = document.createElement("a")
    link.setAttribute("href", url)
    link.setAttribute("download", `TriMVPP_Batch_Results_${new Date().getTime()}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const etaMs = isStreaming && timeline && startedAt !== null
    ? estimateRemainingMs(timeline, startedAt, now, stats.totalProcessed)
    : null

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
              {(isStreaming || statusText) && (
                <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-indigo-50/60 border border-indigo-100 px-3 py-2 text-xs">
                  <span className="flex items-center gap-2 text-indigo-800 font-medium">
                    {isStreaming && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                    {statusText ?? "Working..."}
                  </span>
                  {isStreaming && startedAt !== null && (
                    <span className="flex items-center gap-1.5 font-mono text-slate-600">
                      <Timer className="h-3.5 w-3.5" />
                      {formatDuration(now - startedAt)} elapsed
                      {totalVariants && stats.totalProcessed < totalVariants && etaMs !== null && (
                        <> · ~{formatDuration(etaMs)} left</>
                      )}
                    </span>
                  )}
                </div>
              )}
              {isStreaming && (
                <p className="text-[11px] text-slate-500">
                  Keep this page open until screening finishes; leaving it stops the run.
                </p>
              )}
              
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
                <span className="text-xs font-normal text-slate-500">
                  (click a row for full detail{streamedVariants.length > TABLE_ROW_LIMIT ? `; showing latest ${TABLE_ROW_LIMIT}, CSV has all` : ""})
                </span>
              </span>
              <div className="flex gap-2 items-center">
                <Badge variant="secondary" className="font-mono text-xs">
                  {streamedVariants.length} Items
                </Badge>
                <Button 
                  onClick={exportToCSV}
                  disabled={streamedVariants.length === 0 && rejectedVariants.length === 0}
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
                  {streamedVariants.slice(0, TABLE_ROW_LIMIT).map((v, i) => (
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
                        v.id === newestId ? "bg-indigo-50/50" : ""
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

      {/* Biến thể bị loại kèm lý do (không phải missense, REF sai, toạ độ không hợp lệ...) */}
      {rejectedVariants.length > 0 && (
        <Card className="border-slate-200 shadow-sm overflow-hidden">
          <CardHeader className="py-4 border-b border-slate-100 bg-slate-50/50">
            <CardTitle className="text-sm flex items-center justify-between">
              <span className="flex items-center gap-2">
                <ShieldAlert className="h-4 w-4 text-rose-600" />
                Rejected Variants
              </span>
              <Badge variant="secondary" className="font-mono text-xs">{rejectedVariants.length} Items</Badge>
            </CardTitle>
            <CardDescription className="text-xs">
              Not classified: the variant is not missense or its coordinates are invalid. Included in the CSV export.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0 max-h-[260px] overflow-auto">
            <Table>
              <TableHeader className="sticky top-0 bg-white/95 backdrop-blur shadow-sm z-10">
                <TableRow>
                  <TableHead>Variant</TableHead>
                  <TableHead>Detected</TableHead>
                  <TableHead>Reason</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rejectedVariants.slice(0, TABLE_ROW_LIMIT).map((r, i) => (
                  <TableRow key={`${r.variant_id}-${i}`}>
                    <TableCell className="font-mono text-xs text-slate-700 whitespace-nowrap">{r.variant_id}</TableCell>
                    <TableCell className="text-xs">
                      <Badge variant="outline" className="font-mono text-[10px] border-rose-200 text-rose-700">
                        {r.consequence}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs text-slate-600">{r.reason}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
      </div>
      </div>
    </div>
  )
}