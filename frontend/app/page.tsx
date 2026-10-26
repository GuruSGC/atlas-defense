"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  ShieldAlert,
  Database,
  CheckCircle,
  XCircle,
  Activity,
  ArrowUpRight,
  RefreshCw,
  Flame,
  ShieldCheck,
  Cpu
} from "lucide-react";
import { WorkflowBanner } from "@/components/layout/WorkflowBanner";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/Card";
import { StatCard } from "@/components/ui/StatCard";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { api } from "@/lib/api";
import {
  HealthResponse,
  MemoryItem,
  EvaluationSummary
} from "@/lib/types";
import { formatDate, formatScore, getScoreTextColor } from "@/lib/utils";

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [recentMemories, setRecentMemories] = useState<MemoryItem[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);
  const [loading, setLoading] = useState(true);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const [healthData, memoryData, evalData] = await Promise.all([
        api.getHealth(),
        api.getMemories(),
        api.getEvaluationSummary()
      ]);
      setHealth(healthData);
      setRecentMemories(memoryData.memories.slice(0, 5));
      setEvaluation(evalData);
    } catch (err) {
      console.error("Dashboard data load error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, []);

  return (
    <div className="space-y-6">
      {/* Top Title & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            Research Dashboard Overview
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Adaptive conversational memory defense monitoring and adversarial metrics
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={loadDashboardData}
            loading={loading}
          >
            <RefreshCw className="w-3.5 h-3.5 mr-1" /> Refresh
          </Button>
          <Link href="/chat">
            <Button size="sm" variant="primary">
              Open Chat Agent <ArrowUpRight className="w-3.5 h-3.5 ml-1" />
            </Button>
          </Link>
        </div>
      </div>

      {/* Visual Pipeline Workflow */}
      <WorkflowBanner />

      {/* Primary Telemetry Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Memories"
          value={health?.total_memories ?? 0}
          subtitle="Persistent storage items"
          icon={<Database className="w-4 h-4" />}
        />
        <StatCard
          title="Accepted Memories"
          value={health?.accepted_memories ?? 0}
          subtitle="Passed all 6 ATLAS signals"
          icon={<CheckCircle className="w-4 h-4 text-emerald-400" />}
          className="border-emerald-900/40 bg-emerald-950/20"
        />
        <StatCard
          title="Quarantined"
          value={health?.quarantined_memories ?? 0}
          subtitle="Blocked adversarial candidate facts"
          icon={<XCircle className="w-4 h-4 text-rose-400" />}
          className="border-rose-900/40 bg-rose-950/20"
        />
        <StatCard
          title="ATLAS Trust Status"
          value="ACTIVE"
          subtitle="Threshold τ = 0.65"
          icon={<ShieldCheck className="w-4 h-4 text-blue-400" />}
          className="border-blue-900/40 bg-blue-950/20"
        />
      </div>

      {/* Research Performance Radar & Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Empirical Metrics Card */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>
                  <Activity className="w-4 h-4 text-blue-400" />
                  Security & Defense Performance Metrics
                </CardTitle>
                <CardDescription>
                  Evaluated across benchmark suites of benign user statements and memory poisoning attacks
                </CardDescription>
              </div>
              <Link href="/evaluation" className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1 font-medium">
                Full Benchmark <ArrowUpRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </CardHeader>
          <CardContent>
            {evaluation ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                  <div className="text-[11px] text-slate-400 font-medium">Attack Success Rate (Without ATLAS)</div>
                  <div className="text-xl font-bold text-rose-400 mt-1">
                    {formatScore(evaluation.attack_success_rate_without_atlas)}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">Vulnerable agent baseline</div>
                </div>

                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                  <div className="text-[11px] text-slate-400 font-medium">Attack Success Rate (With ATLAS)</div>
                  <div className="text-xl font-bold text-emerald-400 mt-1">
                    {formatScore(evaluation.attack_success_rate_with_atlas)}
                  </div>
                  <div className="text-[10px] text-emerald-500 mt-0.5">Significant risk reduction</div>
                </div>

                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                  <div className="text-[11px] text-slate-400 font-medium">Detection Rate (Recall)</div>
                  <div className="text-xl font-bold text-blue-400 mt-1">
                    {formatScore(evaluation.detection_rate)}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">True positive poison capture</div>
                </div>

                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                  <div className="text-[11px] text-slate-400 font-medium">Precision</div>
                  <div className="text-xl font-bold text-slate-200 mt-1">
                    {formatScore(evaluation.precision)}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">TP / (TP + FP)</div>
                </div>

                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                  <div className="text-[11px] text-slate-400 font-medium">F1-Score</div>
                  <div className="text-xl font-bold text-indigo-400 mt-1">
                    {formatScore(evaluation.f1_score)}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">Harmonic mean precision-recall</div>
                </div>

                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                  <div className="text-[11px] text-slate-400 font-medium">Retrieval Accuracy</div>
                  <div className="text-xl font-bold text-teal-400 mt-1">
                    {formatScore(evaluation.retrieval_accuracy)}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">Safe context augmentation</div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-slate-500 text-xs">
                Loading evaluation benchmark telemetry...
              </div>
            )}
          </CardContent>
        </Card>

        {/* Quick Attack Lab Launch Card */}
        <Card>
          <CardHeader>
            <CardTitle>
              <Flame className="w-4 h-4 text-amber-400" />
              Attack Simulation Lab
            </CardTitle>
            <CardDescription>
              Demonstrate live memory poisoning attacks with ATLAS defense on vs off
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-3 rounded-lg bg-amber-950/20 border border-amber-900/40 text-xs space-y-2">
              <div className="font-semibold text-amber-300">Active Test Scenarios:</div>
              <ul className="space-y-1.5 text-slate-300 text-[11px]">
                <li className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                  Direct Fact Overwrite (Database Hijack)
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                  Covert Instruction Injection (Backdoor)
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                  Subtle Preference Drift (three steps)
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                  Sybil Persona (authority claim)
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                  Context Splicing (two fragments)
                </li>
              </ul>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              Verify how ATLAS intercepts poisoned conversational turns before they can corrupt downstream vector retrieval.
            </p>

            <Link href="/attack-lab" className="block">
              <Button variant="secondary" size="sm" className="w-full">
                Launch Attack Lab <ArrowUpRight className="w-3.5 h-3.5 ml-1" />
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>

      {/* Recent Memory Activity Feed */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>
                <Database className="w-4 h-4 text-purple-400" />
                Recent Memory Ingestion Activity
              </CardTitle>
              <CardDescription>
                Latest candidate statements evaluated by ATLAS Trust Engine
              </CardDescription>
            </div>
            <Link href="/memories" className="text-xs text-blue-400 hover:text-blue-300 font-medium flex items-center gap-1">
              View All Memories <ArrowUpRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-[11px] text-slate-400 border-b border-slate-800 uppercase tracking-wider font-semibold">
                <tr>
                  <th className="pb-3 font-medium">Memory Content</th>
                  <th className="pb-3 font-medium">Category</th>
                  <th className="pb-3 font-medium">Trust Score</th>
                  <th className="pb-3 font-medium">Status</th>
                  <th className="pb-3 font-medium">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                {recentMemories.length > 0 ? (
                  recentMemories.map((mem) => (
                    <tr key={mem.id} className="hover:bg-slate-900/40 transition-colors">
                      <td className="py-3 pr-4 font-sans text-slate-200 max-w-md truncate">
                        {mem.content}
                      </td>
                      <td className="py-3 pr-4 text-slate-400 font-sans">
                        <span className="px-2 py-0.5 rounded bg-slate-800 text-[10px] text-slate-300">
                          {mem.category}
                        </span>
                      </td>
                      <td className="py-3 pr-4">
                        <span className={`font-bold ${getScoreTextColor(mem.trust_score)}`}>
                          {formatScore(mem.trust_score)}
                        </span>
                      </td>
                      <td className="py-3 pr-4">
                        <Badge status={mem.status}>{mem.status}</Badge>
                      </td>
                      <td className="py-3 text-slate-500 font-sans text-[11px]">
                        {formatDate(mem.timestamp)}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} className="py-8 text-center text-slate-500 font-sans">
                      No memories recorded yet. Start a chat session or run an attack scenario.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
