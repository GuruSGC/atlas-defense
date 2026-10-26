"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  MessageSquare,
  Database,
  ShieldAlert,
  Flame,
  BarChart3,
  ShieldCheck,
  Cpu
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  {
    label: "Dashboard",
    href: "/",
    icon: LayoutDashboard,
    description: "Overview & metrics",
  },
  {
    label: "Chat",
    href: "/chat",
    icon: MessageSquare,
    description: "Defended agent interface",
  },
  {
    label: "Memory Monitor",
    href: "/memories",
    icon: Database,
    description: "Persistent & quarantined",
  },
  {
    label: "ATLAS Trust Analysis",
    href: "/atlas",
    icon: ShieldCheck,
    description: "Six-signal diagnostics",
  },
  {
    label: "Attack Lab",
    href: "/attack-lab",
    icon: Flame,
    description: "Poisoning demonstrations",
  },
  {
    label: "Evaluation",
    href: "/evaluation",
    icon: BarChart3,
    description: "Empirical research benchmarks",
  },
];

export const Sidebar: React.FC = () => {
  const pathname = usePathname();

  return (
    <aside className="w-64 bg-slate-950 border-r border-slate-800 flex flex-col h-screen sticky top-0 select-none">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800/80 flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-blue-600/20 border border-blue-500/40 flex items-center justify-center text-blue-400 font-bold">
          <ShieldAlert className="w-5 h-5" />
        </div>
        <div>
          <div className="font-bold text-sm text-slate-100 tracking-wider flex items-center gap-1.5">
            ATLAS
            <span className="text-[10px] px-1.5 py-0.2 rounded bg-blue-950 text-blue-400 border border-blue-800 font-mono">
              v1.0
            </span>
          </div>
          <p className="text-[10px] text-slate-400 leading-tight">
            Agent Memory Security
          </p>
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <div className="px-3 pb-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
          Research Modules
        </div>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-medium transition-all group",
                isActive
                  ? "bg-blue-600 text-white shadow-sm font-semibold"
                  : "text-slate-400 hover:text-slate-100 hover:bg-slate-900"
              )}
            >
              <Icon
                className={cn(
                  "w-4 h-4 shrink-0 transition-colors",
                  isActive ? "text-white" : "text-slate-500 group-hover:text-slate-300"
                )}
              />
              <div className="flex flex-col">
                <span>{item.label}</span>
              </div>
            </Link>
          );
        })}
      </nav>

      {/* System Status Footer */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-950/80">
        <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-[11px] space-y-1.5">
          <div className="flex items-center justify-between text-slate-300">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              ATLAS Active
            </span>
            <span className="font-mono text-slate-400 text-[10px]">T = 0.65</span>
          </div>
          <div className="flex items-center gap-1 text-[10px] text-slate-500">
            <Cpu className="w-3 h-3" /> Learned + rule-based, 6 signals
          </div>
        </div>
      </div>
    </aside>
  );
};
