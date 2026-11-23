import * as React from "react"
import { cn } from "@/lib/utils"

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary" | "destructive" | "outline" | "success" | "warning"
}

function Badge({ className, variant = "default", ...props }: BadgeProps) {
  const variants = {
    default: "border-transparent bg-blue-600 text-white shadow hover:bg-blue-700",
    secondary: "border-transparent bg-slate-100 text-slate-800 hover:bg-slate-200",
    destructive: "border-transparent bg-rose-500 text-white shadow hover:bg-rose-600",
    success: "border-transparent bg-emerald-500 text-white shadow hover:bg-emerald-600",
    warning: "border-transparent bg-amber-500 text-white shadow hover:bg-amber-600",
    outline: "text-slate-700 border-slate-300",
  }

  return (
    <div
      className={cn(
        "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2",
        variants[variant],
        className
      )}
      {...props}
    />
  )
}

export { Badge }
