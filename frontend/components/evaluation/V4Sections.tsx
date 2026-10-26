"use client";

import React from "react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/Card";
import { BenchmarkV4, MetricRow } from "@/lib/types";
import { formatScore } from "@/lib/utils";

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

function pair(lo?: number[] | null): string {
  return lo && lo.length === 2 ? `${formatScore(lo[0])} to ${formatScore(lo[1])}` : "";
}

function Cell({ v, ci }: { v?: number | null; ci?: number[] | null }) {
  return (
    <td className="py-2.5 pr-4 font-mono text-slate-300">
      {formatScore(v)} {ci ? <span className="text-slate-500">({pair(ci)})</span> : null}
    </td>
  );
}

function MetricTable({ rows, extra }: { rows: Record<string, MetricRow>; extra?: (r: MetricRow) => React.ReactNode }) {
  return (
    <Table head={<><Th>System</Th><Th>Recall (95% CI)</Th><Th>False positives (95% CI)</Th><Th>F1</Th><Th>AUC</Th></>}>
      {Object.entries(rows).map(([name, r]) => (
        <tr key={name} className="hover:bg-slate-900/40">
          <td className="py-2.5 pr-4 text-slate-200 font-medium">{name}{extra ? extra(r) : null}</td>
          <Cell v={r.recall} ci={r.recall_ci} />
          <Cell v={r.fpr} ci={r.fpr_ci} />
          <td className="py-2.5 pr-4 font-mono text-slate-300">{formatScore(r.f1)}</td>
          <td className="py-2.5 font-mono text-slate-300">{r.auc != null ? r.auc.toFixed(3) : ""}</td>
        </tr>
      ))}
    </Table>
  );
}

export function V4Sections({ data }: { data: BenchmarkV4 }) {
  const ladder = data.ladder?.configs;
  const base = data.baselines?.rows;
  const anomaly = data.anomaly?.detectors;
  const adaptive = data.adaptive;
  const prb = data.prb?.conditions;
  const sens = data.sensitivity;
  const frozen = data.frozen_original_on_all_items?.per_source;
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Benchmark v4: {data.totals.cases.toLocaleString()} cases</CardTitle>
          <CardDescription>
            {data.totals.attacks} attacks and {data.totals.benign} benign items; {data.totals.public_cases} come from public sources.
            Learned parts are trained on the train split only, and every table below is scored on the held-out test split.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table head={<><Th>Source</Th><Th>Kind</Th><Th>Items</Th><Th>Train / test</Th><Th>Origin</Th><Th>License</Th>{frozen ? <Th>Original rules flagged</Th> : null}</>}>
            {data.composition.map((c) => (
              <tr key={c.source} className="hover:bg-slate-900/40">
                <td className="py-2.5 pr-4 text-slate-200 font-medium">{c.source}</td>
                <td className="py-2.5 pr-4 text-slate-400">{c.kind}</td>
                <td className="py-2.5 pr-4 font-mono text-slate-300">{c.n}</td>
                <td className="py-2.5 pr-4 font-mono text-slate-300">{c.train} / {c.test}</td>
                <td className="py-2.5 pr-4 text-slate-400">{c.origin}</td>
                <td className="py-2.5 pr-4 text-slate-400">{c.license}</td>
                {frozen ? <td className="py-2.5 font-mono text-slate-300">{frozen[c.source] ? formatScore(frozen[c.source].rate) : ""}</td> : null}
              </tr>
            ))}
          </Table>
          <p className="mt-3 text-[11px] text-slate-500">
            The last column is how often the original rule-based prototype flagged each source. It was frozen before any public item was inspected.
          </p>
        </CardContent>
      </Card>

      {ladder && (
        <Card>
          <CardHeader>
            <CardTitle>Ablation ladder (test split)</CardTitle>
            <CardDescription>The same items scored by each configuration, from the original prototype to the full learned one.</CardDescription>
          </CardHeader>
          <CardContent>
            <MetricTable rows={ladder} extra={(r) => <div className="text-[11px] text-slate-500 font-normal">{r.description}</div>} />
          </CardContent>
        </Card>
      )}

      {base && (
        <Card>
          <CardHeader>
            <CardTitle>Baselines (test split)</CardTitle>
            <CardDescription>Unsupervised detectors use a threshold that would flag about 5% of held-out benign data.</CardDescription>
          </CardHeader>
          <CardContent>
            <MetricTable rows={{ ...base, ...(data.llm_moderator ? { "LLM moderator (200-item sample)": data.llm_moderator } : {}) }} />
          </CardContent>
        </Card>
      )}

      {anomaly && (
        <Card>
          <CardHeader>
            <CardTitle>Anomaly detectors on their own</CardTitle>
            <CardDescription>Fitted on benign train embeddings only. AUC on the test split.</CardDescription>
          </CardHeader>
          <CardContent>
            <Table head={<><Th>Detector</Th><Th>AUC (95% CI)</Th><Th>Attacks caught at 1% FPR</Th><Th>at 5% FPR</Th></>}>
              {Object.entries(anomaly).map(([name, r]) => (
                <tr key={name} className="hover:bg-slate-900/40">
                  <td className="py-2.5 pr-4 text-slate-200 font-medium">{name}</td>
                  <td className="py-2.5 pr-4 font-mono text-slate-300">{r.auc.toFixed(3)} <span className="text-slate-500">({r.auc_ci?.map((x) => x.toFixed(3)).join(" to ")})</span></td>
                  <td className="py-2.5 pr-4 font-mono text-slate-300">{formatScore(r.tpr_at_1pct_fpr)}</td>
                  <td className="py-2.5 font-mono text-slate-300">{formatScore(r.tpr_at_5pct_fpr)}</td>
                </tr>
              ))}
            </Table>
          </CardContent>
        </Card>
      )}

      {adaptive?.search && (
        <Card>
          <CardHeader>
            <CardTitle>Adaptive adversary: black-box search</CardTitle>
            <CardDescription>Share of originally flagged attacks that a greedy attacker gets accepted within a number of queries.</CardDescription>
          </CardHeader>
          <CardContent>
            <Table head={<><Th>Defense</Th>{adaptive.search.budgets.map((b) => <Th key={b}>{b} queries</Th>)}</>}>
              {Object.entries(adaptive.search.systems).map(([name, s]) => (
                <tr key={name} className="hover:bg-slate-900/40">
                  <td className="py-2.5 pr-4 text-slate-200 font-medium">{name}</td>
                  {adaptive.search!.budgets.map((b) => (
                    <td key={b} className="py-2.5 pr-4 font-mono text-slate-300">{formatScore(s.evasion_rate[String(b)]?.rate)}</td>
                  ))}
                </tr>
              ))}
            </Table>
          </CardContent>
        </Card>
      )}

      {prb && (
        <Card>
          <CardHeader>
            <CardTitle>Attack success: persistence, retrieval, behaviour</CardTitle>
            <CardDescription>{data.prb?.n} poisoned passages with natural probe questions, answered by the live model.</CardDescription>
          </CardHeader>
          <CardContent>
            <Table head={<><Th>Condition</Th><Th>Persisted</Th><Th>Retrieved</Th><Th>Wrong answer given</Th><Th>Attack success</Th></>}>
              {Object.entries(prb).map(([name, r]) => (
                <tr key={name} className="hover:bg-slate-900/40">
                  <td className="py-2.5 pr-4 text-slate-200 font-medium">{name.replace(/_/g, " ")}</td>
                  <Cell v={r.persisted.rate} ci={r.persisted.ci} />
                  <Cell v={r.retrieved_given_persisted_or_all.rate} ci={r.retrieved_given_persisted_or_all.ci} />
                  <Cell v={r.behaviour_changed?.rate} ci={r.behaviour_changed?.ci} />
                  <Cell v={r.attack_success?.rate} ci={r.attack_success?.ci} />
                </tr>
              ))}
            </Table>
          </CardContent>
        </Card>
      )}

      {sens && (
        <Card>
          <CardHeader>
            <CardTitle>Sensitivity</CardTitle>
            <CardDescription>
              Over {sens.random_weights.draws} random signal-weight vectors, recall at the default threshold was
              {" "}{formatScore(sens.random_weights.recall_p5_p50_p95[0])} to {formatScore(sens.random_weights.recall_p5_p50_p95[2])} (5th to 95th percentile),
              with false positives {formatScore(sens.random_weights.fpr_p5_p50_p95[0])} to {formatScore(sens.random_weights.fpr_p5_p50_p95[2])}.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table head={<><Th>Trust threshold</Th><Th>Recall</Th><Th>False positives</Th><Th>F1</Th></>}>
              {Object.entries(sens.threshold).map(([t, r]) => (
                <tr key={t} className={t === "0.65" ? "bg-blue-950/20" : "hover:bg-slate-900/40"}>
                  <td className="py-2 pr-4 font-mono text-slate-200">{t}</td>
                  <td className="py-2 pr-4 font-mono text-slate-300">{formatScore(r.recall)}</td>
                  <td className="py-2 pr-4 font-mono text-slate-300">{formatScore(r.fpr)}</td>
                  <td className="py-2 font-mono text-slate-300">{formatScore(r.f1)}</td>
                </tr>
              ))}
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
