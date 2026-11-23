import React from "react"
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { MainLayout } from "@/components/layout/MainLayout"
import { Home } from "@/pages/Home"
import { Single } from "@/pages/Single"
import { Batch } from "@/pages/Batch"
import { Xai } from "@/pages/Xai"

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <MainLayout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/single" element={<Single />} />
          <Route path="/batch" element={<Batch />} />
          <Route path="/xai" element={<Xai />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </MainLayout>
    </BrowserRouter>
  )
}

export default App
