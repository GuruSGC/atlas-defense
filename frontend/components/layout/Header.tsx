"use client";

import React, { useEffect, useState } from "react";
import { ShieldCheck, RefreshCw, AlertCircle, Database } from "lucide-react";
import { api } from "@/lib/api";
import { HealthResponse } from "@/lib/types";

export const Header: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isOnline, setIsOnline] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchStatus = async () => {
    setLoading(true);
    try {
      const data = await api.getHealth();
      setHealth(data);
      setIsOnline(true);
    } catch {
      setIsOnline(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const timer = setInterval(fetchStatus, 15000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header className="h-16 border-b border-slate-800 bg-slate-950/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <h1 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-blue-400" />
          ATLAS
          <span className="text-slate-500 font-normal hidden sm:inline text-xs">
            : Adaptive Trust Management for Long-term Agent Storage
          </span>
        </h1>
      </div>

      <div className="flex items-center gap-4 text-xs">
        {/* Memory telemetry counters */}
        {health && (
          <div className="hidden md:flex items-center gap-3 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 font-mono text-[11px]">
            <span className="flex items-center gap-1 text-slate-400">
              <Database className="w-3 h-3 text-slate-500" />
              Store: <strong className="text-slate-200">{health.total_memories}</strong>
            </span>
            <span className="text-slate-600">|</span>
            <span className="text-emerald-400">
              Accepted: <strong>{health.accepted_memories}</strong>
            </span>
            <span className="text-slate-600">|</span>
            <span className="text-rose-400">
              Quarantined: <strong>{health.quarantined_memories}</strong>
            </span>
          </div>
        )}

        {/* Backend Connectivity Status */}
        <div className="flex items-center gap-2">
          {isOnline === true ? (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-950/80 border border-emerald-800/80 text-emerald-400 font-medium text-[11px]">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              Backend Connected
            </div>
          ) : isOnline === false ? (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-rose-950/80 border border-rose-800/80 text-rose-400 font-medium text-[11px]">
              <AlertCircle className="w-3 h-3" />
              Backend Disconnected
            </div>
          ) : (
            <div className="text-slate-500 text-[11px]">Checking...</div>
          )}

          <button
            onClick={fetchStatus}
            title="Refresh backend telemetry"
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-blue-400" : ""}`} />
          </button>
        </div>
      </div>
    </header>
  );
};
