import React from "react"
import { Navigate, Outlet, RouterProvider, createBrowserRouter } from "react-router-dom"
import { MainLayout } from "@/components/layout/MainLayout"
import { Home } from "@/pages/Home"
import { Single } from "@/pages/Single"
import { Batch } from "@/pages/Batch"
import { Xai } from "@/pages/Xai"

// Data router (createBrowserRouter) để trang Batch dùng được useBlocker: chặn rời trang khi đang chạy.
const router = createBrowserRouter([
  {
    element: (
      <MainLayout>
        <Outlet />
      </MainLayout>
    ),
    children: [
      { path: "/", element: <Home /> },
      { path: "/single", element: <Single /> },
      { path: "/batch", element: <Batch /> },
      { path: "/xai", element: <Xai /> },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
])

const App: React.FC = () => <RouterProvider router={router} />

export default App
