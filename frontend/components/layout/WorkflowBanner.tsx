import React from "react";
import {
  MessageSquare,
  FileSearch,
  ShieldCheck,
  CheckCircle2,
  XCircle,
  Database,
  ArrowRight
} from "lucide-react";

export const WorkflowBanner: React.FC = () => {
  const steps = [
    {
      title: "1. Dialogue Turn",
      desc: "User / Agent Chat",
      icon: MessageSquare,
      color: "text-blue-400 border-blue-500/30 bg-blue-950/40",
    },
    {
      title: "2. Extraction",
      desc: "Candidate Memory",
      icon: FileSearch,
      color: "text-purple-400 border-purple-500/30 bg-purple-950/40",
    },
    {
      title: "3. ATLAS Signals",
      desc: "6 Security Signals",
      icon: ShieldCheck,
      color: "text-amber-400 border-amber-500/30 bg-amber-950/40",
    },
    {
      title: "4. Trust Verdict",
      desc: "Accept / Quarantine",
      icon: CheckCircle2,
      color: "text-emerald-400 border-emerald-500/30 bg-emerald-950/40",
    },
    {
      title: "5. Persistent Store",
      desc: "Safe Long-Term Memory",
      icon: Database,
      color: "text-cyan-400 border-cyan-500/30 bg-cyan-950/40",
    },
  ];

  return (
    <div className="w-full rounded-xl bg-slate-900/80 border border-slate-800 p-4 mb-6 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-3">
        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
          ATLAS Security Defense Pipeline
        </span>
        <span className="text-[10px] text-slate-500 font-mono">
          Threshold: 0.65 | kNN / isolation forest anomaly signal
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5 items-center">
        {steps.map((step, idx) => {
          const Icon = step.icon;
          return (
            <div key={idx} className="relative flex items-center">
              <div
                className={`w-full flex items-center gap-2.5 p-2.5 rounded-lg border ${step.color} transition-all`}
              >
                <div className="p-1.5 rounded bg-slate-900/80 shrink-0">
                  <Icon className="w-4 h-4" />
                </div>
                <div className="min-w-0">
                  <div className="text-xs font-semibold text-slate-200 truncate">
                    {step.title}
                  </div>
                  <div className="text-[10px] text-slate-400 truncate">{step.desc}</div>
                </div>
              </div>

              {idx < steps.length - 1 && (
                <div className="hidden lg:flex absolute -right-2 top-1/2 -translate-y-1/2 z-10 text-slate-600">
                  <ArrowRight className="w-3.5 h-3.5" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
