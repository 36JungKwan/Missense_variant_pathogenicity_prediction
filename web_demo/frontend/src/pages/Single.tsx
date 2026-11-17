import React, { useState, useEffect, useMemo } from "react"
import { useNavigate } from "react-router-dom"
import {
  Search,
  Loader2,
  Dna,
  Binary,
  Layers,
  Box,
  Flame,
  BarChart3,
  Sparkles,
  Info,
  Zap,
  RotateCcw,
  CheckCircle2,
  ShieldAlert,
  Database,
  Cpu
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import mockData from "@/mock/mockData.json"
import { AblationState, VariantData } from "@/types"
import { predictSingleVariant, SingleVariantRequest, SingleVariantResponse } from "@/services/api"

export const Single: React.FC = () => {
  const navigate = useNavigate()

  // Input variant ID
  const [variantInput, setVariantInput] = useState<string>("chr7:140753336:A>T")
  const [activePresetId, setActivePresetId] = useState<string>("pathogenic")

  // Core Execution States
  const [useCache, setUseCache] = useState<boolean>(true)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  // 4 Ablation Toggles state (DNA, Protein, Biological, Spatial Geometry)
  const [ablation, setAblation] = useState<AblationState>({
    dnaFeatures: true,
    proteinFeatures: true,
    biologicalFeatures: true,
    spatialGeometry: true,
  })

  // Results & Loading States
  const [currentResult, setCurrentResult] = useState<VariantData | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [isCacheHit, setIsCacheHit] = useState<boolean>(true)
  const [responseTimeMs, setResponseTimeMs] = useState<number>(0)

  // Core Request Execution via FastAPI Service
  useEffect(() => {
    let isMounted = true

    const fetchPrediction = async () => {
      // Don't fetch if input is empty
      if (!variantInput.trim()) return

      try {
        setIsLoading(true)
        setErrorMessage(null)

        const requestPayload: SingleVariantRequest = {
          variant_id: variantInput,
          dna_enabled: ablation.dnaFeatures,
          protein_enabled: ablation.proteinFeatures,
          bio_enabled: ablation.biologicalFeatures,
          geo_enabled: ablation.spatialGeometry,
          use_cache: useCache,
        }

        const response: SingleVariantResponse = await predictSingleVariant(requestPayload)

        if (!isMounted) return

        setCurrentResult({
          variant_id: response.variant_id,
          gene: response.gene,
          consequence: response.consequence,
          clinvar_ground_truth: response.clinvar_ground_truth,
          results: response.results,
          sota: response.sota,
          shap_analysis: response.shap_analysis,
        })

        setIsCacheHit(response.meta.is_cache_hit)
        setResponseTimeMs(response.meta.computation_time_ms)
      } catch (error: any) {
        if (isMounted) {
          setErrorMessage(error.message || "An unexpected error occurred during prediction.")
          setCurrentResult(null)
        }
      } finally {
        if (isMounted) {
          setIsLoading(false)
        }
      }
    }

    // Debounce or trigger fetch
    const timeoutId = setTimeout(() => {
      fetchPrediction()
    }, 300)

    return () => {
      isMounted = false
      clearTimeout(timeoutId)
    }
  }, [variantInput, ablation, activePresetId, useCache])

  // Count active ablation toggles
  const activeToggleCount = useMemo(() => {
    let count = 0
    if (ablation.dnaFeatures) count++
    if (ablation.proteinFeatures) count++
    if (ablation.biologicalFeatures) count++
    if (ablation.spatialGeometry) count++
    return count
  }, [ablation])

  // Tính toán chính xác Bitmask (0-15) dựa trên trạng thái bật/tắt
  const currentMask = useMemo(() => {
    return (
      (ablation.dnaFeatures ? 8 : 0) |
      (ablation.proteinFeatures ? 4 : 0) |
      (ablation.biologicalFeatures ? 2 : 0) |
      (ablation.spatialGeometry ? 1 : 0)
    )
  }, [ablation])

  // Preset Selection Handler
  const handleSelectPreset = (presetId: string) => {
    const found = mockData.presets.find(p => p.id === presetId)
    if (found) {
      setActivePresetId(presetId)
      setVariantInput(found.variant_id)
      setUseCache(true)
      setAblation({
        dnaFeatures: true,
        proteinFeatures: true,
        biologicalFeatures: true,
        spatialGeometry: true,
      })
    }
  }

  // Toggle reset
  const handleResetToggles = () => {
    setAblation({
      dnaFeatures: true,
      proteinFeatures: true,
      biologicalFeatures: true,
      spatialGeometry: true,
    })
  }

  return (
    <div className="space-y-8 text-left max-w-5xl mx-auto">
      {/* Page Header */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-semibold text-blue-600 uppercase tracking-wider">
            <Dna className="h-4 w-4" />
            <span>Single Variant Inspection Mode</span>
          </div>

          {/* Database Status Badge */}
          <div className="flex items-center gap-2">
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

      {/* Input & Preset Quick Select Card */}
      <Card className="border-slate-200 shadow-sm overflow-hidden">
        <CardHeader className="bg-slate-50/70 border-b border-slate-100 pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">Genomic Coordinate Input (GRCh38)</CardTitle>
            
            {/* Live Bypass Toggle */}
            <div className="flex items-center gap-2">
              <span className={`text-xs font-semibold ${!useCache ? 'text-amber-600' : 'text-slate-500'}`}>
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
          {/* Input field and action button */}
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
              <input
                type="text"
                value={variantInput}
                onChange={(e) => setVariantInput(e.target.value)}
                placeholder="Enter variant ID (e.g., chr7:140753336:A>T)"
                className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-slate-200 text-sm font-mono text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all bg-white"
              />
            </div>

            <Button
              disabled={isLoading || !variantInput.trim()}
              className="sm:w-52 bg-blue-600 hover:bg-blue-700 text-white font-medium flex items-center justify-center gap-2 shadow-md shadow-blue-500/15"
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
          </div>

          {/* Quick Preset Selector Buttons */}
          <div className="pt-2 border-t border-slate-100 flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium text-slate-500 flex items-center gap-1">
              <Info className="h-3.5 w-3.5" />
              Quick Validation Presets:
            </span>

            <button
              type="button"
              onClick={() => handleSelectPreset("pathogenic")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-all cursor-pointer flex items-center gap-1.5 ${
                activePresetId === "pathogenic"
                  ? "bg-rose-50 text-rose-700 border-rose-300 font-semibold shadow-2xs"
                  : "bg-white text-slate-600 border-slate-200 hover:bg-rose-50/50 hover:text-rose-600"
              }`}
            >
              <span className="h-2 w-2 rounded-full bg-rose-500"></span>
              <span>BRAF V600E (Pathogenic)</span>
            </button>

            <button
              type="button"
              onClick={() => handleSelectPreset("brca1")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-all cursor-pointer flex items-center gap-1.5 ${
                activePresetId === "brca1"
                  ? "bg-rose-50 text-rose-700 border-rose-300 font-semibold shadow-2xs"
                  : "bg-white text-slate-600 border-slate-200 hover:bg-rose-50/50 hover:text-rose-600"
              }`}
            >
              <span className="h-2 w-2 rounded-full bg-rose-500"></span>
              <span>BRCA1 p.Arg1699Gln (Pathogenic)</span>
            </button>

            <button
              type="button"
              onClick={() => handleSelectPreset("benign")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-all cursor-pointer flex items-center gap-1.5 ${
                activePresetId === "benign"
                  ? "bg-emerald-50 text-emerald-700 border-emerald-300 font-semibold shadow-2xs"
                  : "bg-white text-slate-600 border-slate-200 hover:bg-emerald-50/50 hover:text-emerald-600"
              }`}
            >
              <span className="h-2 w-2 rounded-full bg-emerald-500"></span>
              <span>BRCA2 p.Asn372His (Benign)</span>
            </button>
          </div>
        </CardContent>
      </Card>

      {/* Gatekeeper Error Alert */}
      {errorMessage && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 flex items-start gap-3 shadow-sm animate-in fade-in zoom-in duration-300">
          <ShieldAlert className="h-5 w-5 shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-bold">Gatekeeper Validation Failed</h4>
            <p className="text-xs mt-1 leading-relaxed font-medium">{errorMessage}</p>
          </div>
        </div>
      )}

      {/* Ablation Study Toggles Card */}
      <Card className="border-slate-200 shadow-sm">
        <CardHeader className="pb-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <CardTitle className="text-base flex items-center gap-2">
                <Layers className="h-4 w-4 text-indigo-600" />
                <span>Feature Ablation Control Panel</span>
              </CardTitle>
            </div>

            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-xs border-slate-200">
                {activeToggleCount}/4 Modules Active
              </Badge>

              {activeToggleCount < 4 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleResetToggles}
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
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 p-4 rounded-xl bg-slate-50 border border-slate-100">
            {/* Toggle 1: DNA Features */}
            <div className="p-3 bg-white rounded-lg border border-slate-200/80 shadow-2xs flex flex-col justify-between gap-2">
              <div className="flex items-center gap-2">
                <Binary className="h-4 w-4 text-blue-500" />
                <span className="text-xs font-semibold text-slate-800">1. DNA Sequence Context</span>
              </div>
              <Switch
                checked={ablation.dnaFeatures}
                onCheckedChange={(checked) =>
                  setAblation((prev) => ({ ...prev, dnaFeatures: checked }))
                }
              />
            </div>

            {/* Toggle 2: Protein Features */}
            <div className="p-3 bg-white rounded-lg border border-slate-200/80 shadow-2xs flex flex-col justify-between gap-2">
              <div className="flex items-center gap-2">
                <Dna className="h-4 w-4 text-purple-500" />
                <span className="text-xs font-semibold text-slate-800">2. Protein Physicochemical</span>
              </div>
              <Switch
                checked={ablation.proteinFeatures}
                onCheckedChange={(checked) =>
                  setAblation((prev) => ({ ...prev, proteinFeatures: checked }))
                }
              />
            </div>

            {/* Toggle 3: Biological Features */}
            <div className="p-3 bg-white rounded-lg border border-slate-200/80 shadow-2xs flex flex-col justify-between gap-2">
              <div className="flex items-center gap-2">
                <Flame className="h-4 w-4 text-amber-500" />
                <span className="text-xs font-semibold text-slate-800">3. Biological &amp; Splicing</span>
              </div>
              <Switch
                checked={ablation.biologicalFeatures}
                onCheckedChange={(checked) =>
                  setAblation((prev) => ({ ...prev, biologicalFeatures: checked }))
                }
              />
            </div>

            {/* Toggle 4: Spatial Geometry (LVD) */}
            <div className="p-3 bg-white rounded-lg border border-slate-200/80 shadow-2xs flex flex-col justify-between gap-2">
              <div className="flex items-center gap-2">
                <Box className="h-4 w-4 text-emerald-500" />
                <span className="text-xs font-semibold text-slate-800">4. Spatial Geometry (LVD)</span>
              </div>
              <Switch
                checked={ablation.spatialGeometry}
                onCheckedChange={(checked) =>
                  setAblation((prev) => ({ ...prev, spatialGeometry: checked }))
                }
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Single Result Panel: RB-Net Hybrid as the Sole Centerpiece */}
      {currentResult && !errorMessage && (
        <div className={`space-y-6 transition-opacity duration-300 ${isLoading ? "opacity-40 pointer-events-none" : "opacity-100"}`}>
          {/* Result Card: RB-Net Hybrid */}
          <Card className="border-blue-300 bg-gradient-to-b from-blue-50/40 via-white to-white shadow-lg ring-1 ring-blue-500/20 overflow-hidden">
            <CardHeader className="pb-4 border-b border-blue-100/60 bg-blue-50/20">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-xl bg-blue-600 text-white shadow-md shadow-blue-500/20">
                    <Dna className="h-6 w-6" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <CardTitle className="text-xl font-extrabold text-blue-950">
                        RB-Net Hybrid Architecture
                      </CardTitle>
                    </div>
                    <CardDescription className="text-xs text-blue-800/70 mt-0.5">
                      Multi-modal integration of sequence, physicochemical, biological, and spatial geometry representations
                    </CardDescription>
                  </div>
                </div>

                {/* Compact Status Badge */}
                <div className="flex items-center gap-2 self-start sm:self-auto">
                  {isLoading ? (
                    <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-300 flex items-center gap-1.5 py-1 text-xs">
                      <Loader2 className="h-3 w-3 animate-spin text-amber-600" />
                      <span>Status: Computing...</span>
                    </Badge>
                  ) : isCacheHit ? (
                    <Badge variant="outline" className="bg-emerald-50 text-emerald-700 border-emerald-300 flex items-center gap-1.5 py-1 text-xs font-semibold">
                      <Zap className="h-3 w-3 text-emerald-600" />
                      <span>Cached Result ({responseTimeMs}ms)</span>
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-300 flex items-center gap-1.5 py-1 text-xs font-semibold">
                      <Cpu className="h-3 w-3 text-blue-600" />
                      <span>Live Bypass ({responseTimeMs}ms)</span>
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
                  onClick={() => navigate(`/xai?variant=${encodeURIComponent(currentResult.variant_id)}&mask=${currentMask}`)}
                  className="text-xs text-blue-600 border-blue-200 hover:bg-blue-50 flex items-center gap-1.5"
                >
                  <BarChart3 className="h-3.5 w-3.5" />
                  <span>Inspect SHAP Attribution</span>
                </Button>
              </div>

              {/* Central Result Grid: Prediction vs Ground Truth */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center p-6 rounded-2xl bg-white border border-slate-200 shadow-xs">
                {/* Left: RB-Net Prediction Score */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                      RB-Net Prediction
                    </span>
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
                    <span className="text-xs text-slate-400 font-medium">
                      Pathogenicity Probability
                    </span>
                  </div>

                  {/* Visual Bar */}
                  <div className="w-full h-3 rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        currentResult.results.rb_net.label === "Pathogenic" ? "bg-rose-500" : "bg-emerald-500"
                      }`}
                      style={{ width: `${currentResult.results.rb_net.probability * 100}%` }}
                    />
                  </div>

                  <p className="text-xs text-slate-500">
                    Threshold: &ge; 50.0% classified as Pathogenic, &lt; 50.0% classified as Benign.
                  </p>
                </div>

                {/* Right: ClinVar Ground Truth Validation Benchmark */}
                <div className="p-5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                    <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                      ClinVar Ground Truth Benchmark
                    </span>
                  </div>

                  <div className="flex items-center justify-between pt-1">
                    <span className="text-xs text-slate-600 font-medium">Official ClinVar Validation:</span>
                    <Badge
                      variant={currentResult.clinvar_ground_truth === "Pathogenic" ? "destructive" : "success"}
                      className="text-xs px-3 py-0.5 font-bold"
                    >
                      {currentResult.clinvar_ground_truth}
                    </Badge>
                  </div>

                  <div className="pt-2 border-t border-slate-200/60 text-xs">
                    {currentResult.results.rb_net.label === currentResult.clinvar_ground_truth ? (
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

          {/* SOTA Benchmarks Table */}
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
            <CardContent>
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

                  <TableRow>
                    <TableCell className="font-medium text-slate-800">
                      AlphaMissense (Google DeepMind)
                    </TableCell>
                    <TableCell className="font-mono text-slate-700">
                      {currentResult.sota.alphamissense.toFixed(3)}
                    </TableCell>
                    <TableCell>
                      <Badge variant={currentResult.sota.alphamissense >= 0.56 ? "destructive" : "success"}>
                        {currentResult.sota.alphamissense >= 0.56 ? "Pathogenic" : "Benign"}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right text-slate-600 font-mono">
                      {(100 - Math.abs(currentResult.results.rb_net.probability - currentResult.sota.alphamissense) * 100).toFixed(1)}%
                    </TableCell>
                  </TableRow>

                  <TableRow>
                    <TableCell className="font-medium text-slate-800">
                      REVEL (Ensemble Predictor)
                    </TableCell>
                    <TableCell className="font-mono text-slate-700">
                      {currentResult.sota.revel.toFixed(3)}
                    </TableCell>
                    <TableCell>
                      <Badge variant={currentResult.sota.revel >= 0.5 ? "destructive" : "success"}>
                        {currentResult.sota.revel >= 0.5 ? "Pathogenic" : "Benign"}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right text-slate-600 font-mono">
                      {(100 - Math.abs(currentResult.results.rb_net.probability - currentResult.sota.revel) * 100).toFixed(1)}%
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}