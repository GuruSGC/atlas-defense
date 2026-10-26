"use client";

import React, { useEffect, useState } from "react";
import {
  BarChart3,
  RefreshCw,
  Activity,
  ShieldCheck,
  ShieldAlert,
  Play,
  Layers,
  Table,
  CheckCircle2,
  XCircle,
  HelpCircle,
  BookOpen
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { StatCard } from "@/components/ui/StatCard";
import { Badge } from "@/components/ui/Badge";
import { api } from "@/lib/api";
import { EvaluationSummary, EvaluationReport, BenchmarkV4 } from "@/lib/types";
import { ReportSections } from "@/components/evaluation/ReportSections";
import { V4Sections } from "@/components/evaluation/V4Sections";
import { formatDate, formatScore } from "@/lib/utils";

export default function EvaluationPage() {
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);
  const [report, setReport] = useState<EvaluationReport | null>(null);
  const [v4, setV4] = useState<BenchmarkV4 | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [runningBenchmark, setRunningBenchmark] = useState<boolean>(false);

  const loadEvaluation = async () => {
    setLoading(true);
    try {
      const data = await api.getEvaluationSummary();
      setEvaluation(data);
      api.getEvaluationReport().then(setReport).catch(() => setReport(null));
      api.getBenchmarkV4().then(setV4).catch(() => setV4(null));
    } catch (err) {
      console.error("Failed to load evaluation metrics:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleRunFreshBenchmark = async () => {
    setRunningBenchmark(true);
    try {
      const data = await api.runEvaluationBenchmark();
      setEvaluation(data);
    } catch (err) {
      console.error("Failed to execute benchmark:", err);
    } finally {
      setRunningBenchmark(false);
    }
  };

  useEffect(() => {
    loadEvaluation();
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-blue-400" />
            Empirical Security & Retrieval Benchmarks
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Quantitative evaluation metrics generated across synthetic poisoning scenarios and benign memory baselines
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={loadEvaluation}
            loading={loading}
          >
            <RefreshCw className="w-3.5 h-3.5 mr-1" /> Refresh Data
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={handleRunFreshBenchmark}
            loading={runningBenchmark}
          >
            <Play className="w-3.5 h-3.5 mr-1" /> Run Fresh Benchmark
          </Button>
        </div>
      </div>

      {/* Dataset Provenance */}
      {evaluation && (
        <Card className="border-indigo-900/50 bg-indigo-950/10">
          <CardContent className="py-3.5">
            <div className="flex items-start gap-3">
              <BookOpen className="w-4 h-4 text-indigo-400 mt-0.5 shrink-0" />
              <div className="space-y-1 text-xs">
                <div className="font-semibold text-slate-200">
                  {evaluation.dataset_info.name}{" "}
                  <span className="text-slate-500 font-mono text-[10px]">v{evaluation.dataset_info.version}</span>
                </div>
                <div className="text-slate-400">
                  <span className="text-slate-500">Benign class:</span> {evaluation.dataset_info.benign_source}
                </div>
                <div className="text-slate-400">
                  <span className="text-slate-500">Adversarial class:</span> {evaluation.dataset_info.adversarial_source}
                </div>
                <div className="text-[10px] text-slate-500">
                  Deterministic seed: <span className="font-mono">{evaluation.dataset_info.seed}</span> (full provenance in evaluation/data/README.md)
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Metric Cards Grid */}
      {evaluation && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <StatCard
            title="Attack Block Rate"
            value={formatScore(1.0 - evaluation.attack_success_rate_with_atlas)}
            subtitle="Adversarial defense"
            className="border-emerald-900/50 bg-emerald-950/20"
          />
          <StatCard
            title="ASR (Protected)"
            value={formatScore(evaluation.attack_success_rate_with_atlas)}
            subtitle={`vs ${formatScore(evaluation.attack_success_rate_without_atlas)} baseline`}
            className="border-blue-900/50 bg-blue-950/20"
          />
          <StatCard
            title="Precision"
            value={formatScore(evaluation.precision)}
            subtitle="TP / (TP + FP)"
          />
          <StatCard
            title="Recall / TPR"
            value={formatScore(evaluation.recall)}
            subtitle={evaluation.recall_ci ? `95% CI ${formatScore(evaluation.recall_ci.low)} to ${formatScore(evaluation.recall_ci.high)}` : "Detection Rate"}
          />
          <StatCard
            title="F1-Score"
            value={formatScore(evaluation.f1_score)}
            subtitle={evaluation.f1_ci ? `95% CI ${formatScore(evaluation.f1_ci.low)} to ${formatScore(evaluation.f1_ci.high)}` : "Harmonic mean"}
          />
          <StatCard
            title="Retrieval Acc"
            value={formatScore(evaluation.retrieval_accuracy)}
            subtitle="Safe context"
          />
        </div>
      )}

      {/* Confusion Matrix and Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Confusion Matrix (6 cols) */}
        <div className="lg:col-span-6 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>
                <Table className="w-4 h-4 text-purple-400" />
                Confusion Matrix & Classification Telemetry
              </CardTitle>
              <CardDescription>
                Classification distribution for memory trust verification decisions
              </CardDescription>
            </CardHeader>
            <CardContent>
              {evaluation ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    {/* True Positive */}
                    <div className="p-3.5 rounded-lg bg-emerald-950/30 border border-emerald-800/60">
                      <div className="text-[10px] text-emerald-400 font-bold uppercase tracking-wider">
                        True Positive (TP)
                      </div>
                      <div className="text-2xl font-bold text-emerald-300 mt-1">
                        {evaluation.confusion_matrix.true_positives}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        Malicious attacks correctly <strong>QUARANTINED</strong>
                      </div>
                    </div>

                    {/* False Positive */}
                    <div className="p-3.5 rounded-lg bg-amber-950/30 border border-amber-800/60">
                      <div className="text-[10px] text-amber-400 font-bold uppercase tracking-wider">
                        False Positive (FP)
                      </div>
                      <div className="text-2xl font-bold text-amber-300 mt-1">
                        {evaluation.confusion_matrix.false_positives}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        Benign memories incorrectly <strong>QUARANTINED</strong>
                      </div>
                    </div>

                    {/* False Negative */}
                    <div className="p-3.5 rounded-lg bg-rose-950/30 border border-rose-800/60">
                      <div className="text-[10px] text-rose-400 font-bold uppercase tracking-wider">
                        False Negative (FN)
                      </div>
                      <div className="text-2xl font-bold text-rose-300 mt-1">
                        {evaluation.confusion_matrix.false_negatives}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        Malicious attacks erroneously <strong>ACCEPTED</strong>
                      </div>
                    </div>

                    {/* True Negative */}
                    <div className="p-3.5 rounded-lg bg-blue-950/30 border border-blue-800/60">
                      <div className="text-[10px] text-blue-400 font-bold uppercase tracking-wider">
                        True Negative (TN)
                      </div>
                      <div className="text-2xl font-bold text-blue-300 mt-1">
                        {evaluation.confusion_matrix.true_negatives}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        Benign memories correctly <strong>ACCEPTED</strong>
                      </div>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
                    <span>Total Evaluated Samples: <strong>{evaluation.total_samples}</strong></span>
                    <span>False Positive Rate (FPR): <strong className="text-slate-200">{formatScore(evaluation.false_positive_rate)}</strong></span>
                  </div>
                </div>
              ) : (
                <div className="py-12 text-center text-slate-500 text-xs">
                  Loading classification telemetry...
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Defense Comparison (6 cols) */}
        <div className="lg:col-span-6 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>
                <Activity className="w-4 h-4 text-emerald-400" />
                Vulnerability vs Defense Comparison
              </CardTitle>
              <CardDescription>
                Empirical comparison between standard undefended memory vs ATLAS trust management
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {evaluation ? (
                <div className="space-y-4">
                  {/* Attack Success Rate Bar Comparison */}
                  <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-3">
                    <div className="text-xs font-semibold text-slate-300">
                      Share of attacks written to memory (persistence only):
                    </div>

                    {/* Undefended */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs text-slate-400">
                        <span>Undefended Agent (No Filter)</span>
                        <span className="text-rose-400 font-mono font-bold">
                          {formatScore(evaluation.attack_success_rate_without_atlas)}
                        </span>
                      </div>
                      <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-rose-500 rounded-full"
                          style={{ width: `${evaluation.attack_success_rate_without_atlas * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* Defended by ATLAS */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs text-slate-400">
                        <span>Defended by ATLAS Trust Engine</span>
                        <span className="text-emerald-400 font-mono font-bold">
                          {formatScore(evaluation.attack_success_rate_with_atlas)}
                        </span>
                      </div>
                      <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-emerald-500 rounded-full transition-all duration-700"
                          style={{ width: `${evaluation.attack_success_rate_with_atlas * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Benchmark Execution Metadata */}
                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-400 space-y-1">
                    <div>
                      Last Execution Timestamp:{" "}
                      <span className="text-slate-200 font-mono">
                        {formatDate(evaluation.execution_timestamp)}
                      </span>
                    </div>
                    <div className="text-[10px] text-slate-500">
                      * Results derived dynamically from execution across empirical test suites.
                    </div>
                  </div>
                </div>
              ) : (
                <div className="py-12 text-center text-slate-500 text-xs">
                  Loading defense comparison...
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Scenario Category Breakdown Table */}
      <Card>
        <CardHeader>
          <CardTitle>Scenario Category Performance Breakdown</CardTitle>
          <CardDescription>
            Defense performance across distinct adversarial memory poisoning categories
          </CardDescription>
        </CardHeader>
        <CardContent>
          {evaluation && evaluation.scenario_breakdown.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-[11px] text-slate-400 border-b border-slate-800 uppercase tracking-wider font-semibold">
                  <tr>
                    <th className="pb-3 font-medium">Attack Category</th>
                    <th className="pb-3 font-medium">Total Test Trials</th>
                    <th className="pb-3 font-medium">Quarantined (Blocked)</th>
                    <th className="pb-3 font-medium">Accepted</th>
                    <th className="pb-3 font-medium">Defense Rate</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-sans text-xs">
                  {evaluation.scenario_breakdown.map((item, idx) => (
                    <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                      <td className="py-3 font-medium text-slate-200">{item.category}</td>
                      <td className="py-3 text-slate-300 font-mono">{item.total_trials}</td>
                      <td className="py-3 text-emerald-400 font-mono">{item.quarantined_count}</td>
                      <td className="py-3 text-slate-400 font-mono">{item.accepted_count}</td>
                      <td className="py-3">
                        <span className="font-mono font-bold text-blue-400">
                          {formatScore(item.defense_rate)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="py-8 text-center text-slate-500 text-xs">
              No scenario breakdown available.
            </div>
          )}
        </CardContent>
      </Card>

      {v4 && <V4Sections data={v4} />}
      {report && <ReportSections report={report} />}
    </div>
  );
}
