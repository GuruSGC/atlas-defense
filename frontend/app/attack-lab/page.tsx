"use client";

import React, { useEffect, useState } from "react";
import {
  Flame,
  ShieldCheck,
  ShieldAlert,
  Play,
  RotateCcw,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Database,
  ArrowRight,
  HelpCircle,
  Bug,
  Lock,
  Unlock
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { TrustGauge } from "@/components/ui/TrustGauge";
import { SignalBar } from "@/components/ui/SignalBar";
import { api } from "@/lib/api";
import { AttackScenario, AttackSimulationResult } from "@/lib/types";
import { formatScore } from "@/lib/utils";

export default function AttackLabPage() {
  const [scenarios, setScenarios] = useState<AttackScenario[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>("");
  const [enableAtlas, setEnableAtlas] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);
  const [resetting, setResetting] = useState<boolean>(false);
  const [simulationResult, setSimulationResult] = useState<AttackSimulationResult | null>(null);

  useEffect(() => {
    const fetchScenarios = async () => {
      try {
        const list = await api.getAttackScenarios();
        setScenarios(list);
        if (list.length > 0) {
          setSelectedScenarioId(list[0].id);
        }
      } catch (err) {
        console.error("Failed to load attack scenarios:", err);
      }
    };
    fetchScenarios();
  }, []);

  const selectedScenario = scenarios.find((s) => s.id === selectedScenarioId);

  const handleRunAttack = async () => {
    if (!selectedScenarioId) return;
    setLoading(true);
    try {
      const res = await api.runAttackScenario(selectedScenarioId, enableAtlas);
      setSimulationResult(res);
    } catch (err) {
      console.error("Attack simulation failed:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    setResetting(true);
    try {
      await api.resetAttackLab();
      setSimulationResult(null);
    } catch (err) {
      console.error("Failed to reset attack lab:", err);
    } finally {
      setResetting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Title Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <Flame className="w-5 h-5 text-rose-500" />
            Adversarial Attack Laboratory
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Demonstrate Conversational Memory Poisoning and compare agent behavior with ATLAS ON vs OFF
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={handleReset}
            loading={resetting}
            title="Reset testbed memories to clean baseline"
          >
            <RotateCcw className="w-3.5 h-3.5 mr-1" /> Reset Testbed
          </Button>
        </div>
      </div>

      {/* Attack Configuration Control Deck */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Scenario Selector & Controls (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>
                <Bug className="w-4 h-4 text-rose-400" />
                Select Attack Scenario
              </CardTitle>
              <CardDescription>
                Choose an adversarial memory poisoning vector for academic demonstration
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                {scenarios.map((scen) => (
                  <div
                    key={scen.id}
                    onClick={() => setSelectedScenarioId(scen.id)}
                    className={`p-3 rounded-lg border cursor-pointer transition-all text-xs ${
                      selectedScenarioId === scen.id
                        ? "bg-blue-950/40 border-blue-500 shadow-sm"
                        : "bg-slate-950/60 border-slate-800/80 hover:bg-slate-900"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-slate-100">{scen.name}</span>
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`text-[10px] font-mono px-1.5 py-0.2 rounded border ${
                            scen.severity === "CRITICAL"
                              ? "bg-rose-950 text-rose-400 border-rose-800"
                              : scen.severity === "HIGH"
                              ? "bg-amber-950 text-amber-400 border-amber-800"
                              : "bg-slate-900 text-slate-400 border-slate-800"
                          }`}
                        >
                          {scen.severity}
                        </span>
                        {!scen.is_active && (
                          <span className="text-[10px] text-slate-500 font-mono">
                            (Placeholder)
                          </span>
                        )}
                      </div>
                    </div>
                    <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                      {scen.description}
                    </p>
                  </div>
                ))}
              </div>

              {/* Defense Toggle Mode */}
              <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
                <div className="text-xs font-semibold text-slate-300 flex items-center justify-between">
                  <span>ATLAS Defense Mode</span>
                  {enableAtlas ? (
                    <span className="flex items-center gap-1 text-emerald-400 text-xs font-bold">
                      <Lock className="w-3.5 h-3.5" /> PROTECTED
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-rose-400 text-xs font-bold">
                      <Unlock className="w-3.5 h-3.5" /> VULNERABLE
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setEnableAtlas(true)}
                    className={`py-2 px-3 rounded-lg text-xs font-medium border flex items-center justify-center gap-1.5 transition-all ${
                      enableAtlas
                        ? "bg-emerald-950 text-emerald-300 border-emerald-700 shadow-sm"
                        : "bg-slate-900 text-slate-400 border-slate-800 hover:bg-slate-800"
                    }`}
                  >
                    <ShieldCheck className="w-3.5 h-3.5" /> ATLAS ON
                  </button>
                  <button
                    type="button"
                    onClick={() => setEnableAtlas(false)}
                    className={`py-2 px-3 rounded-lg text-xs font-medium border flex items-center justify-center gap-1.5 transition-all ${
                      !enableAtlas
                        ? "bg-rose-950 text-rose-300 border-rose-700 shadow-sm"
                        : "bg-slate-900 text-slate-400 border-slate-800 hover:bg-slate-800"
                    }`}
                  >
                    <ShieldAlert className="w-3.5 h-3.5" /> ATLAS OFF
                  </button>
                </div>
              </div>

              <Button
                onClick={handleRunAttack}
                loading={loading}
                className="w-full"
                variant={enableAtlas ? "primary" : "danger"}
                size="md"
              >
                <Play className="w-4 h-4 mr-1.5" /> Execute Attack Simulation
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Attack Progression & Execution Output (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <Card className="h-full flex flex-col">
            <CardHeader>
              <CardTitle>
                <ShieldAlert className="w-4 h-4 text-blue-400" />
                Demonstration Trace & Verification Outcome
              </CardTitle>
              <CardDescription>
                Step-by-step adversarial injection lifecycle and downstream exploitation probe
              </CardDescription>
            </CardHeader>

            <CardContent className="flex-1 space-y-5">
              {simulationResult ? (
                <div className="space-y-5">
                  {/* Overall Defense Banner */}
                  <div
                    className={`p-4 rounded-xl border flex items-center justify-between ${
                      simulationResult.defense_successful
                        ? "bg-emerald-950/40 border-emerald-800 text-emerald-200"
                        : "bg-rose-950/40 border-rose-800 text-rose-200"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      {simulationResult.defense_successful ? (
                        <div className="p-2 rounded-lg bg-emerald-900/60 text-emerald-400">
                          <CheckCircle2 className="w-6 h-6" />
                        </div>
                      ) : (
                        <div className="p-2 rounded-lg bg-rose-900/60 text-rose-400">
                          <XCircle className="w-6 h-6" />
                        </div>
                      )}
                      <div>
                        <div className="text-sm font-bold">
                          {simulationResult.defense_successful
                            ? "ATTACK BLOCKED & MEMORY QUARANTINED"
                            : "ATTACK SUCCEEDED & AGENT MEMORY POISONED"}
                        </div>
                        <div className="text-xs opacity-80 mt-0.5">
                          {simulationResult.defense_successful
                            ? "ATLAS detected anomalous trust parameters and prevented persistent memory poisoning."
                            : simulationResult.enable_atlas
                              ? "The payload passed the trust check because each step looks harmless on its own, then persisted and changed the downstream answer."
                              : "Vulnerable mode: Malicious payload was persisted and corrupted downstream retrieval."}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Step 1: Attacker Injection Payload */}
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
                    <div className="text-[10px] font-bold text-rose-400 uppercase tracking-wider flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
                      Step 1: Attacker Dialogue Turn
                    </div>
                    <p className="text-xs text-slate-200 font-mono bg-slate-900 p-2 rounded border border-slate-800/80">
                      "{simulationResult.attacker_prompt}"
                    </p>
                    <div className="text-[11px] text-slate-400">
                      Candidate Memory Extracted:{" "}
                      <strong className="text-slate-200">
                        "{simulationResult.candidate_poisoned_memory.content}"
                      </strong>
                    </div>
                  </div>

                  {/* Step 2: ATLAS Trust Decision */}
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-3">
                    <div className="text-[10px] font-bold text-blue-400 uppercase tracking-wider flex items-center justify-between">
                      <span className="flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                        Step 2: ATLAS Trust Verdict
                      </span>
                      <Badge status={simulationResult.memory_decision}>
                        {simulationResult.memory_decision}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 items-center">
                      <div className="flex justify-center p-2">
                        <TrustGauge
                          score={simulationResult.trust_score}
                          threshold={0.65}
                          size="sm"
                        />
                      </div>
                      <div className="text-xs text-slate-300 space-y-1 leading-relaxed">
                        <div>
                          <strong>Decision Note:</strong>
                        </div>
                        <p className="text-[11px] text-slate-400">
                          {simulationResult.decision_reason}
                        </p>
                      </div>
                    </div>

                    {/* Six signals */}
                    {simulationResult.signals && (
                      <div className="space-y-1.5 pt-2 border-t border-slate-800">
                        <SignalBar
                          name="Historical Consistency"
                          score={simulationResult.signals.historical_consistency}
                        />
                        <SignalBar
                          name="Behavioral Consistency"
                          score={simulationResult.signals.behavioral_consistency}
                        />
                        <SignalBar
                          name="Embedding Anomaly"
                          score={simulationResult.signals.embedding_anomaly}
                        />
                      </div>
                    )}
                  </div>

                  {/* Step 3: Downstream Exploitation Test */}
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
                    <div className="text-[10px] font-bold text-purple-400 uppercase tracking-wider flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-purple-500" />
                      Step 3: Downstream Agent Exploitation Test
                    </div>
                    <div className="text-xs text-slate-300">
                      Probe Query: <strong>"{simulationResult.probe_query}"</strong>
                    </div>
                    <div className="p-2.5 rounded bg-slate-900 border border-slate-800 text-xs text-slate-200">
                      <div className="text-[10px] text-slate-400 font-semibold mb-1">
                        Agent Downstream Response:
                      </div>
                      "{simulationResult.probe_response}"
                    </div>
                  </div>
                </div>
              ) : (
                <div className="py-20 text-center text-slate-500 text-xs space-y-2">
                  <HelpCircle className="w-8 h-8 text-slate-600 mx-auto stroke-[1.5]" />
                  <p>Select an attack scenario and click "Execute Attack Simulation" to observe the demonstration.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
