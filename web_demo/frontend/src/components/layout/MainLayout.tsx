import React from "react"
import { Link, useLocation, useNavigate } from "react-router-dom"
import { ArrowLeft, Dna, Activity, FileSpreadsheet, BarChart3 } from "lucide-react"
import { Button } from "@/components/ui/button"

interface MainLayoutProps {
  children: React.ReactNode
}

export const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const location = useLocation()
  const navigate = useNavigate()
  const isHomePage = location.pathname === "/"

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 selection:bg-blue-100 selection:text-blue-900">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-50 w-full border-b border-slate-200/80 bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/80 shadow-2xs">
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Left: Brand & Title */}
          <div className="flex items-center space-x-3">
            <Link to="/" className="flex items-center space-x-3 group">
              <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-sky-500 flex items-center justify-center text-white shadow-md shadow-blue-500/20 group-hover:scale-105 transition-transform duration-200">
                <Dna className="h-5 w-5 animate-pulse" />
              </div>
              <div className="flex flex-col text-left">
                <span className="text-base sm:text-lg font-bold tracking-tight text-slate-900 group-hover:text-blue-600 transition-colors">
                  Missense Variant Pathogenicity Classification System
                </span>
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-semibold px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">
                    RB-Net Hybrid
                  </span>
                </div>
              </div>
            </Link>
          </div>

          {/* Right: Navigation items and Back Button */}
          <div className="flex items-center space-x-2 sm:space-x-3">
            {/* Quick nav links */}
            <nav className="hidden md:flex items-center space-x-1 mr-2 text-sm font-medium">
              <Link
                to="/single"
                className={`px-3 py-1.5 rounded-md transition-colors flex items-center gap-1.5 ${
                  location.pathname === "/single"
                    ? "bg-blue-50 text-blue-700 font-semibold"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
                }`}
              >
                <Activity className="h-4 w-4" />
                Single Variant
              </Link>
              <Link
                to="/batch"
                className={`px-3 py-1.5 rounded-md transition-colors flex items-center gap-1.5 ${
                  location.pathname === "/batch"
                    ? "bg-blue-50 text-blue-700 font-semibold"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
                }`}
              >
                <FileSpreadsheet className="h-4 w-4" />
                Batch Screening
              </Link>
              <Link
                to="/xai"
                className={`px-3 py-1.5 rounded-md transition-colors flex items-center gap-1.5 ${
                  location.pathname === "/xai"
                    ? "bg-blue-50 text-blue-700 font-semibold"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
                }`}
              >
                <BarChart3 className="h-4 w-4" />
                XAI Attribution
              </Link>
            </nav>

            {/* Mandatory Back to Home button when not on Home */}
            {!isHomePage && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => navigate("/")}
                className="flex items-center gap-1.5 border-slate-300 hover:bg-blue-50 hover:text-blue-700 hover:border-blue-300 font-medium transition-all"
              >
                <ArrowLeft className="h-4 w-4" />
                <span>Back to Home</span>
              </Button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-500">
        <div className="max-w-[1440px] mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <p>
            © 2026 RB-Net Variant Classifier • Genomic Pathogenicity Assessment • AIP491 Project
          </p>
          <div className="flex items-center gap-3">
            <span className="inline-flex items-center gap-1 text-slate-400">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
              Inference Engine Ready
            </span>
            <span>FastAPI Integration (Phase 2)</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
