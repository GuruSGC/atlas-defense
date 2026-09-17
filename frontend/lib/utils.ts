import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { MemoryStatus } from "./types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatScore(score: number | null | undefined): string {
  if (score === null || score === undefined) return "N/A";
  return (score * 100).toFixed(1) + "%";
}

export function getStatusBadgeClasses(status: MemoryStatus | string): string {
  switch (status) {
    case "ACCEPTED":
      return "bg-emerald-950/60 text-emerald-400 border-emerald-800/60 ring-emerald-500/20";
    case "QUARANTINED":
      return "bg-rose-950/60 text-rose-400 border-rose-800/60 ring-rose-500/20";
    case "PENDING_ANALYSIS":
    default:
      return "bg-amber-950/60 text-amber-400 border-amber-800/60 ring-amber-500/20";
  }
}

export function getScoreTextColor(score: number): string {
  if (score >= 0.75) return "text-emerald-400";
  if (score >= 0.60) return "text-amber-400";
  return "text-rose-400";
}

export function getScoreBgColor(score: number): string {
  if (score >= 0.75) return "bg-emerald-500";
  if (score >= 0.60) return "bg-amber-500";
  return "bg-rose-500";
}

export function formatDate(isoString: string): string {
  try {
    const d = new Date(isoString);
    return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" }) +
      " (" + d.toLocaleDateString("en-US", { month: "short", day: "numeric" }) + ")";
  } catch {
    return isoString;
  }
}
