import React from "react"
import { useNavigate } from "react-router-dom"
import { 
  Activity, 
  FileSpreadsheet, 
  BarChart3, 
  ArrowRight, 
  Dna, 
  ShieldCheck, 
  Cpu, 
  Layers, 
  Sparkles 
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

export const Home: React.FC = () => {
  const navigate = useNavigate()

  const features = [
    {
      title: "Single Variant Lookup",
      subtitle: "On-the-Fly Variant Inspection",
      description:
        "Input GRCh38 genomic coordinates to trigger real-time pathogenicity classification by the proposed RB-Net Hybrid model with dynamic Ablation Toggles and SOTA benchmarks.",
      icon: Activity,
      path: "/single",
      badge: "Real-time Inference",
      badgeVariant: "default" as const,
      color: "from-blue-600 to-indigo-600",
      accentBg: "bg-blue-50 text-blue-600 border-blue-200",
      details: ["4-Block Feature Ablation Study", "ClinVar Ground Truth Validation", "SOTA Benchmarks (AlphaMissense, REVEL)"],
    },
    {
      title: "Batch VCF Screening",
      subtitle: "High-Throughput VCF Stream",
      description:
        "Upload standard VCF (Variant Call Format) files from Next-Generation Sequencing. System streams variants through non-blocking SSE pipelines to prioritize pathogenic drivers.",
      icon: FileSpreadsheet,
      path: "/batch",
      badge: "Streaming SSE",
      badgeVariant: "default" as const,
      color: "from-blue-600 to-sky-600",
      accentBg: "bg-blue-50 text-blue-600 border-blue-200",
      details: ["Non-blocking 50-Variant Queue Stream", "Live Progress & Auto-Scroll", "ClinVar Ground Truth Benchmarking"],
    },
    {
      title: "XAI Feature Attribution",
      subtitle: "Explainable AI (SHAP Waterfall)",
      description:
        "Visualize the exact contribution of each biological and spatial feature (LVD, SpliceAI, Spatial Structural Perturbation, Allele Frequency) to explain classification decisions.",
      icon: BarChart3,
      path: "/xai",
      badge: "Clinical Explainability",
      badgeVariant: "success" as const,
      color: "from-emerald-500 to-teal-600",
      accentBg: "bg-emerald-50 text-emerald-600 border-emerald-200",
      details: ["Diverging / Waterfall Attribution Chart", "Positive/Negative Impact Decomposition", "Deep-linked Variant Context"],
    },
  ]

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 text-white p-8 sm:p-12 shadow-xl border border-slate-800">
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-96 h-96 rounded-full bg-blue-500/10 blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/3 -mb-20 w-80 h-80 rounded-full bg-indigo-500/10 blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl space-y-5 text-left">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 backdrop-blur border border-white/20 text-xs font-medium text-blue-200">
            <Sparkles className="h-3.5 w-3.5 text-blue-400" />
            <span>Deep Learning &amp; Bioinformatics Hybrid Framework</span>
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white leading-tight">
            Variant Pathogenicity <br />
            <span className="bg-gradient-to-r from-blue-400 via-indigo-300 to-teal-300 bg-clip-text text-transparent">
              Classification System (RB-Net)
            </span>
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed max-w-2xl">
            AI-powered decision support system designed to accurately classify genomic missense and non-coding variants. 
            Integrates deep neural sequence representations with spatial geometry LVD embeddings and multi-modal biological functional annotations.
          </p>

        </div>
      </section>

      {/* 3 Main Action Cards Grid */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2 text-left">
          <div>
            <h2 className="text-2xl font-bold tracking-tight text-slate-900">
              Core Functional Modules
            </h2>
            <p className="text-sm text-slate-500">
              Select an analytical pipeline tailored for clinical research and genomic variant interpretation
            </p>
          </div>
          <span className="text-xs text-slate-400 hidden sm:inline-block">
            Click on any card to launch module
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {features.map((feature, idx) => {
            const Icon = feature.icon
            return (
              <Card
                key={idx}
                onClick={() => navigate(feature.path)}
                className="group relative cursor-pointer border-slate-200/80 hover:border-blue-400 hover:shadow-xl hover:-translate-y-1 transition-all duration-300 flex flex-col justify-between overflow-hidden text-left"
              >
                <div className={`h-1.5 w-full bg-gradient-to-r ${feature.color}`} />
                
                <div>
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between mb-3">
                      <div className={`p-3 rounded-xl border ${feature.accentBg} transition-transform group-hover:scale-110 duration-200`}>
                        <Icon className="h-6 w-6" />
                      </div>
                      <Badge variant={feature.badgeVariant}>{feature.badge}</Badge>
                    </div>

                    <CardTitle className="text-xl group-hover:text-blue-600 transition-colors">
                      {feature.title}
                    </CardTitle>
                    <span className="text-xs font-mono text-slate-400 block -mt-1">
                      {feature.subtitle}
                    </span>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    <CardDescription className="text-sm text-slate-600 leading-relaxed">
                      {feature.description}
                    </CardDescription>

                    <div className="space-y-1.5 pt-2 border-t border-slate-100">
                      {feature.details.map((detail, dIdx) => (
                        <div key={dIdx} className="flex items-center text-xs text-slate-500 gap-2">
                          <span className="h-1.5 w-1.5 rounded-full bg-blue-500 shrink-0" />
                          <span>{detail}</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </div>

                <div className="p-6 pt-0 mt-4">
                  <Button
                    variant="outline"
                    className="w-full justify-between group-hover:bg-blue-600 group-hover:text-white group-hover:border-blue-600 transition-all font-medium"
                  >
                    <span>Launch Module</span>
                    <ArrowRight className="h-4 w-4 transform group-hover:translate-x-1 transition-transform" />
                  </Button>
                </div>
              </Card>
            )
          })}
        </div>
      </section>

      {/* Architecture Highlights Section */}
      <section className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 text-left space-y-6">
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-indigo-600" />
          <h3 className="text-lg font-bold text-slate-900">
            RB-Net Hybrid Architecture Highlights
          </h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-sm">
          <div className="space-y-2 p-4 rounded-xl bg-slate-50 border border-slate-100">
            <div className="flex items-center gap-2 font-semibold text-slate-800">
              <Cpu className="h-4 w-4 text-blue-600" />
              <span>1. Multi-modal Feature Extraction</span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Synchronously extracts 4 feature representations: nucleotide sequences, protein physicochemical properties, biological splicing annotations, and spatial geometry LVD vectors.
            </p>
          </div>

          <div className="space-y-2 p-4 rounded-xl bg-slate-50 border border-slate-100">
            <div className="flex items-center gap-2 font-semibold text-slate-800">
              <ShieldCheck className="h-4 w-4 text-emerald-600" />
              <span>2. Strict Binary Classification</span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Provides decisive binary predictions (Pathogenic vs. Benign) calibrated directly against official ClinVar ground truth standards.
            </p>
          </div>

          <div className="space-y-2 p-4 rounded-xl bg-slate-50 border border-slate-100">
            <div className="flex items-center gap-2 font-semibold text-slate-800">
              <Dna className="h-4 w-4 text-purple-600" />
              <span>3. Explainable AI (XAI) Verification</span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Quantifies exact SHAP attribution scores to enable clinical investigators to inspect why a variant is predicted as pathogenic or benign.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
