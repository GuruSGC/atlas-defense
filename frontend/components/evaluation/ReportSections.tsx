"use client";

import React from "react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/Card";
import { EvaluationReport, Interval } from "@/lib/types";
import { formatScore } from "@/lib/utils";

const PROTOCOL_LABELS: Record<string, string> = {
  label_informed_legacy: "Label-informed (leaks the answer)",
  pipeline_static: "Pipeline, static (primary)",
  pipeline_streaming: "Pipeline, streaming (store grows, detector refits)",
  pipeline_unauth: "Pipeline, attacker on an untrusted channel",
};

const SET_LABELS: Record<string, string> = {
  dev_adversarial: "Dev adversarial",
  heldout_adversarial: "Held-out adversarial",
  evasion_adversarial: "Paraphrase (evasion) adversarial",
  benign_personachat: "Benign, PersonaChat",
  benign_technical: "Benign, technical",
  legitimate_updates: "Legitimate updates",
};

const SEED_LABELS: Record<string, string> = {
  static_precision: "Static precision",
  static_recall: "Static recall",
  static_f1: "Static F1",
  static_fpr: "Static false-positive rate",
  streaming_precision: "Streaming precision",
  streaming_recall: "Streaming recall",
  streaming_f1: "Streaming F1",
  streaming_fpr: "Streaming false-positive rate",
};

function ci(iv?: Interval | null): string {
  return iv ? `${formatScore(iv.low)} to ${formatScore(iv.high)}` : "";
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="pb-3 pr-4 font-medium">{children}</th>;
}

function Table({ head, children }: { head: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead className="text-[11px] text-slate-400 border-b border-slate-800 uppercase tracking-wider font-semibold">
          <tr>{head}</tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60 text-xs">{children}</tbody>
      </table>
    </div>
  );
}

export function ReportSections({ report }: { report: EvaluationReport }) {
  const protocols = Object.values(report.protocols);
  const sets = Object.values(report.sets);
  const sweep = report.threshold_sweep;
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Evaluation protocols</CardTitle>
          <CardDescription>
            The same 120 items scored four ways. The label-informed row copies the answer into the candidate and is shown only
            to measure how much that inflated the first version of this benchmark.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table head={<><Th>Protocol</Th><Th>TP / FN / FP / TN</Th><Th>Precision</Th><Th>Recall (95% CI)</Th><Th>F1 (95% CI)</Th><Th>FPR</Th><Th>AUC</Th></>}>
            {protocols.map((p) => {
              const c = p.confusion_matrix;
              return (
                <tr key={p.protocol} className="hover:bg-slate-900/40">
                  <td className="py-3 pr-4 text-slate-200 font-medium">{PROTOCOL_LABELS[p.protocol] ?? p.protocol}</td>
                  <td className="py-3 pr-4 font-mono text-slate-300">{c.true_positives} / {c.false_negatives} / {c.false_positives} / {c.true_negatives}</td>
                  <td className="py-3 pr-4 font-mono text-slate-300">{formatScore(p.precision)}</td>
                  <td className="py-3 pr-4 font-mono text-slate-300">{formatScore(p.recall)} <span className="text-slate-500">({ci(p.recall_ci)})</span></td>
                  <td className="py-3 pr-4 font-mono text-slate-300">{formatScore(p.f1_score)} <span className="text-slate-500">({ci(p.f1_ci)})</span></td>
                  <td className="py-3 pr-4 font-mono text-slate-300">{formatScore(p.false_positive_rate)}</td>
                  <td className="py-3 font-mono text-slate-300">{p.auc != null ? p.auc.toFixed(3) : ""}</td>
                </tr>
              );
            })}
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Other datasets</CardTitle>
          <CardDescription>Recall on attack sets and false-positive rate on benign sets, with 95% Wilson intervals.</CardDescription>
        </CardHeader>
        <CardContent>
          <Table head={<><Th>Set</Th><Th>Items</Th><Th>Flagged</Th><Th>Rate (95% CI)</Th><Th>What it is</Th></>}>
            {sets.map((s) => (
              <tr key={s.name} className="hover:bg-slate-900/40">
                <td className="py-3 pr-4 text-slate-200 font-medium">{SET_LABELS[s.name] ?? s.name}</td>
                <td className="py-3 pr-4 font-mono text-slate-300">{s.n}</td>
                <td className="py-3 pr-4 font-mono text-slate-300">{s.flagged}</td>
                <td className="py-3 pr-4 font-mono">
                  <span className={s.kind === "attack" ? "text-emerald-400" : "text-amber-400"}>{formatScore(s.rate)}</span>{" "}
                  <span className="text-slate-500">({ci(s.rate_ci)})</span>
                </td>
                <td className="py-3 text-slate-400">{s.description}</td>
              </tr>
            ))}
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Comparison with other defenses</CardTitle>
          <CardDescription>
            Plain re-implementations of the published ideas, scored on identical items. An authenticated attacker uses the same
            signed channel as the user.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table head={<><Th>Defense</Th><Th>Attacker</Th><Th>Dev recall</Th><Th>Held-out recall</Th><Th>FPR casual</Th><Th>FPR technical</Th><Th>FPR updates</Th></>}>
            {report.baselines.map((b, i) => (
              <tr key={i} className="hover:bg-slate-900/40">
                <td className="py-3 pr-4 text-slate-200 font-medium">{b.defense}</td>
                <td className="py-3 pr-4 text-slate-400">{b.attacker}</td>
                <td className="py-3 pr-4 font-mono text-slate-300">{formatScore(b.dev_recall)}</td>
                <td className="py-3 pr-4 font-mono text-slate-300">{formatScore(b.heldout_recall)}</td>
                <td className="py-3 pr-4 font-mono text-slate-300">{formatScore(b.false_positive_rate_personachat)}</td>
                <td className="py-3 pr-4 font-mono text-slate-300">{formatScore(b.false_positive_rate_technical)}</td>
                <td className="py-3 font-mono text-slate-300">{formatScore(b.false_positive_rate_updates)}</td>
              </tr>
            ))}
          </Table>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Stability across {report.seed_sweep.seeds.length} seeds</CardTitle>
            <CardDescription>The seed changes the isolation forest, the smoothing noise, and the streaming order.</CardDescription>
          </CardHeader>
          <CardContent>
            <Table head={<><Th>Metric</Th><Th>Mean</Th><Th>SD</Th><Th>Range</Th></>}>
              {Object.entries(report.seed_sweep.metrics).map(([k, v]) => (
                <tr key={k} className="hover:bg-slate-900/40">
                  <td className="py-2.5 pr-4 text-slate-200">{SEED_LABELS[k] ?? k}</td>
                  <td className="py-2.5 pr-4 font-mono text-slate-300">{formatScore(v.mean)}</td>
                  <td className="py-2.5 pr-4 font-mono text-slate-300">{(v.sd * 100).toFixed(2)}</td>
                  <td className="py-2.5 font-mono text-slate-400">{formatScore(v.min)} to {formatScore(v.max)}</td>
                </tr>
              ))}
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Trust threshold sweep</CardTitle>
            <CardDescription>Primary protocol. A veto quarantines an item at every threshold.</CardDescription>
          </CardHeader>
          <CardContent>
            <Table head={<><Th>Threshold</Th><Th>Precision</Th><Th>Recall</Th><Th>FPR</Th></>}>
              {sweep.map((r) => (
                <tr key={r.threshold} className={r.threshold === 0.65 ? "bg-blue-950/20" : "hover:bg-slate-900/40"}>
                  <td className="py-2 pr-4 font-mono text-slate-200">{r.threshold.toFixed(2)}</td>
                  <td className="py-2 pr-4 font-mono text-slate-300">{formatScore(r.precision)}</td>
                  <td className="py-2 pr-4 font-mono text-slate-300">{formatScore(r.recall)}</td>
                  <td className="py-2 font-mono text-slate-300">{formatScore(r.false_positive_rate)}</td>
                </tr>
              ))}
            </Table>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Attack Lab (persistence, retrieval, and behavior)</CardTitle>
          <CardDescription>
            Attack success needs all three stages. Gradual drift and split payloads get through because each memory is scored on its own.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table head={<><Th>Scenario</Th><Th>No defense</Th><Th>With ATLAS</Th><Th>Persisted</Th><Th>Retrieved</Th><Th>Behavior changed</Th><Th>Trust</Th></>}>
            {report.attack_lab.map((r) => (
              <tr key={r.scenario_id} className="hover:bg-slate-900/40">
                <td className="py-3 pr-4 text-slate-200 font-medium">{r.name}</td>
                <td className="py-3 pr-4 text-rose-400">{r.attack_succeeded_without_atlas ? "attack succeeds" : "blocked"}</td>
                <td className={`py-3 pr-4 ${r.attack_succeeded_with_atlas ? "text-rose-400" : "text-emerald-400"}`}>
                  {r.attack_succeeded_with_atlas ? "attack succeeds" : "blocked"}
                </td>
                <td className="py-3 pr-4 text-slate-400">{r.persisted_with_atlas ? "yes" : "no"}</td>
                <td className="py-3 pr-4 text-slate-400">{r.retrieved_with_atlas ? "yes" : "no"}</td>
                <td className="py-3 pr-4 text-slate-400">{r.behavior_compromised_with_atlas ? "yes" : "no"}</td>
                <td className="py-3 font-mono text-slate-300">{r.trust_score_with_atlas.toFixed(3)}</td>
              </tr>
            ))}
          </Table>
        </CardContent>
      </Card>

      <Card className="border-amber-900/40 bg-amber-950/10">
        <CardContent className="py-3.5">
          <div className="text-xs font-semibold text-slate-200 mb-1.5">Read these results with care</div>
          <ul className="list-disc pl-4 space-y-1 text-[11px] text-slate-400">
            {report.notes.map((n, i) => <li key={i}>{n}</li>)}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
