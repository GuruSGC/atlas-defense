"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Trash2,
  RotateCcw,
  ShieldCheck,
  ShieldAlert,
  Bot,
  User,
  Sparkles,
  Database,
  Info,
  ChevronRight
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { TrustGauge } from "@/components/ui/TrustGauge";
import { SignalBar } from "@/components/ui/SignalBar";
import { api } from "@/lib/api";
import { ChatTurn, ChatResponse, CandidateMemory, TrustSignalsBreakdown } from "@/lib/types";
import { formatScore } from "@/lib/utils";

interface ExtendedMessage extends ChatTurn {
  id: string;
  candidate_memory?: CandidateMemory | null;
  memory_status?: string | null;
  trust_score?: number | null;
  signals?: TrustSignalsBreakdown | null;
  decision_reason?: string | null;
  retrieved_memories?: string[];
  timestamp: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ExtendedMessage[]>([
    {
      id: "msg-initial",
      role: "model",
      content:
        "Hello! I am your AI assistant equipped with the ATLAS conversational memory defense engine. You can chat with me, share facts or preferences, or test adversarial inputs to see how ATLAS intercepts candidate memories in real-time.",
      timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [enableAtlas, setEnableAtlas] = useState(true);
  const [activeCandidateDetail, setActiveCandidateDetail] = useState<ExtendedMessage | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    const userTurn: ExtendedMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: text,
      timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userTurn]);
    setInput("");
    setLoading(true);

    try {
      // Build history for backend
      const historyPayload = messages
        .filter((m) => m.role === "user" || m.role === "model")
        .map((m) => ({ role: m.role, content: m.content }));

      const res: ChatResponse = await api.sendChatMessage({
        message: text,
        history: historyPayload,
        enable_atlas: enableAtlas,
      });

      const modelTurn: ExtendedMessage = {
        id: `model-${Date.now()}`,
        role: "model",
        content: res.response,
        candidate_memory: res.candidate_memory,
        memory_status: res.memory_status,
        trust_score: res.trust_score,
        signals: res.signals,
        decision_reason: res.decision_reason,
        retrieved_memories: res.retrieved_memories,
        timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
      };

      setMessages((prev) => [...prev, modelTurn]);

      if (res.candidate_memory) {
        setActiveCandidateDetail(modelTurn);
      }
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "Failed to communicate with agent.";
      const errorTurn: ExtendedMessage = {
        id: `err-${Date.now()}`,
        role: "model",
        content: `Error: ${errMsg}. Please ensure the backend server is running on http://localhost:8000.`,
        timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
      };
      setMessages((prev) => [...prev, errorTurn]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setMessages([
      {
        id: "msg-initial-cleared",
        role: "model",
        content: "Conversation session cleared. Memory defense active.",
        timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
      },
    ]);
    setActiveCandidateDetail(null);
  };

  const handleResetMemory = async () => {
    try {
      setLoading(true);
      await api.resetAttackLab();
      setMessages([
        {
          id: `msg-reset-${Date.now()}`,
          role: "model",
          content: "🔄 Memory Store Reset: Persistent vector store has been re-initialized to the clean academic baseline (Python preference, AWS us-east-1). All conversational state cleared.",
          timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
      setActiveCandidateDetail(null);
    } catch {
      handleClear();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-8.5rem)]">
      {/* Chat Conversation Column */}
      <div className="lg:col-span-2 flex flex-col h-full rounded-xl border border-slate-800 bg-slate-900/90 shadow-md overflow-hidden">
        {/* Chat Header Bar */}
        <div className="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between bg-slate-950/60">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-lg bg-blue-600/20 text-blue-400">
              <Bot className="w-4 h-4" />
            </div>
            <div>
              <div className="text-xs font-semibold text-slate-200">
                ATLAS Defended Chat Agent
              </div>
              <div className="text-[10px] text-slate-400">
                Real-time memory intercept & trust validation
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* ATLAS Toggle Switch */}
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800">
              <label className="text-[11px] text-slate-300 font-medium cursor-pointer flex items-center gap-1.5 select-none">
                <input
                  type="checkbox"
                  checked={enableAtlas}
                  onChange={(e) => setEnableAtlas(e.target.checked)}
                  className="rounded bg-slate-800 border-slate-700 text-blue-600 focus:ring-0 cursor-pointer"
                />
                <span className={enableAtlas ? "text-emerald-400 font-semibold" : "text-rose-400 font-semibold"}>
                  {enableAtlas ? "ATLAS ON" : "ATLAS OFF (Vulnerable)"}
                </span>
              </label>
            </div>

            <Button variant="outline" size="sm" onClick={handleResetMemory} title="Reset persistent memory store and clear history">
              <RotateCcw className="w-3.5 h-3.5 mr-1 text-amber-400" /> Reset Memory
            </Button>

            <Button variant="ghost" size="sm" onClick={handleClear} title="Clear conversation history">
              <Trash2 className="w-3.5 h-3.5 mr-1 text-slate-400" /> Clear
            </Button>
          </div>
        </div>

        {/* Message Stream */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg) => {
            const isUser = msg.role === "user";
            return (
              <div
                key={msg.id}
                className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}
              >
                {!isUser && (
                  <div className="w-7 h-7 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400 shrink-0 mt-0.5">
                    <Bot className="w-3.5 h-3.5" />
                  </div>
                )}

                <div className={`flex flex-col max-w-[80%] space-y-1.5 ${isUser ? "items-end" : "items-start"}`}>
                  <div
                    className={`px-4 py-3 rounded-xl text-xs leading-relaxed ${
                      isUser
                        ? "bg-blue-600 text-white rounded-br-none shadow-sm"
                        : "bg-slate-950 border border-slate-800 text-slate-200 rounded-bl-none shadow-sm"
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{msg.content}</p>

                    {/* Show Context-Retrieved Memories Pill */}
                    {msg.retrieved_memories && msg.retrieved_memories.length > 0 && (
                      <div className="mt-2.5 pt-2 border-t border-slate-800/80 text-[10px] text-slate-400 space-y-1">
                        <div className="font-semibold text-slate-300 flex items-center gap-1">
                          <Database className="w-3 h-3 text-cyan-400" />
                          Retrieved Verified Memory Context:
                        </div>
                        <ul className="list-disc list-inside space-y-0.5 text-cyan-300/80">
                          {msg.retrieved_memories.map((rm, idx) => (
                            <li key={idx} className="truncate">{rm}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>

                  {/* Candidate Memory Event Tag */}
                  {msg.candidate_memory && (
                    <button
                      onClick={() => setActiveCandidateDetail(msg)}
                      className={`flex items-center gap-2 px-2.5 py-1 rounded-md text-[11px] border transition-all text-left ${
                        msg.memory_status === "ACCEPTED"
                          ? "bg-emerald-950/40 border-emerald-800/60 text-emerald-300 hover:bg-emerald-950/70"
                          : "bg-rose-950/40 border-rose-800/60 text-rose-300 hover:bg-rose-950/70"
                      }`}
                    >
                      <Sparkles className="w-3 h-3 shrink-0" />
                      <span className="truncate">
                        Memory Candidate: <strong>{msg.candidate_memory.content}</strong>
                      </span>
                      <Badge status={msg.memory_status || "PENDING_ANALYSIS"} size="sm">
                        {msg.memory_status}
                      </Badge>
                      <span className="font-mono text-[10px] text-slate-400">
                        {formatScore(msg.trust_score)}
                      </span>
                      <ChevronRight className="w-3 h-3 ml-auto text-slate-500" />
                    </button>
                  )}

                  <span className="text-[10px] text-slate-500 px-1">{msg.timestamp}</span>
                </div>

                {isUser && (
                  <div className="w-7 h-7 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 shrink-0 mt-0.5">
                    <User className="w-3.5 h-3.5" />
                  </div>
                )}
              </div>
            );
          })}

          {loading && (
            <div className="flex gap-3 justify-start items-center">
              <div className="w-7 h-7 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400 shrink-0">
                <Bot className="w-3.5 h-3.5 animate-pulse" />
              </div>
              <div className="px-4 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-400 text-xs flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-ping" />
                Processing message and running candidate memory through ATLAS...
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Form */}
        <form onSubmit={handleSend} className="p-3 border-t border-slate-800 bg-slate-950/80 flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message, user preference, or test adversarial injection..."
            className="flex-1 px-4 py-2.5 rounded-lg bg-slate-900 border border-slate-700/80 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 font-sans"
            disabled={loading}
          />
          <Button type="submit" size="md" variant="primary" disabled={!input.trim() || loading} loading={loading}>
            <Send className="w-3.5 h-3.5" /> Send
          </Button>
        </form>
      </div>

      {/* Real-time ATLAS Memory Inspector Drawer */}
      <div className="flex flex-col h-full overflow-y-auto space-y-4">
        <Card className="h-full flex flex-col">
          <CardHeader>
            <CardTitle>
              <ShieldCheck className="w-4 h-4 text-blue-400" />
              ATLAS Real-time Trust Inspector
            </CardTitle>
            <CardDescription>
              Live breakdown of candidate memories extracted from chat turns
            </CardDescription>
          </CardHeader>

          <CardContent className="flex-1 flex flex-col space-y-4 overflow-y-auto">
            {activeCandidateDetail && activeCandidateDetail.candidate_memory ? (
              <div className="space-y-4">
                {/* Candidate Content Box */}
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                  <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                    Extracted Candidate Statement
                  </div>
                  <div className="text-xs font-medium text-slate-200">
                    "{activeCandidateDetail.candidate_memory.content}"
                  </div>
                  <div className="flex items-center gap-2 pt-1">
                    <Badge status={activeCandidateDetail.memory_status || "PENDING_ANALYSIS"}>
                      {activeCandidateDetail.memory_status}
                    </Badge>
                    <span className="text-[10px] text-slate-500">
                      Category: {activeCandidateDetail.candidate_memory.category}
                    </span>
                  </div>
                </div>

                {/* Trust Score Gauge */}
                <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 flex justify-center">
                  <TrustGauge
                    score={activeCandidateDetail.trust_score || 0}
                    threshold={0.65}
                    size="md"
                  />
                </div>

                {/* Decision Explanation */}
                {activeCandidateDetail.decision_reason && (
                  <div className="p-3 rounded-lg bg-slate-900/90 border border-slate-800 text-[11px] leading-relaxed text-slate-300">
                    <strong className="text-slate-200 block mb-0.5">Decision Rationale:</strong>
                    {activeCandidateDetail.decision_reason}
                  </div>
                )}

                {/* Six Signals Breakdown */}
                {activeCandidateDetail.signals && (
                  <div className="space-y-2">
                    <div className="text-xs font-semibold text-slate-300">
                      Six ATLAS Trust Signals:
                    </div>
                    <SignalBar
                      name="1. Semantic Relevance"
                      score={activeCandidateDetail.signals.semantic_relevance}
                      weight={0.20}
                      description="Context alignment"
                    />
                    <SignalBar
                      name="2. Embedding Anomaly"
                      score={activeCandidateDetail.signals.embedding_anomaly}
                      weight={0.20}
                      description="Isolation Forest inlier check"
                    />
                    <SignalBar
                      name="3. Historical Consistency"
                      score={activeCandidateDetail.signals.historical_consistency}
                      weight={0.20}
                      description="Contradiction / overwrite check"
                    />
                    <SignalBar
                      name="4. Behavioral Consistency"
                      score={activeCandidateDetail.signals.behavioral_consistency}
                      weight={0.15}
                      description="Prompt injection check"
                    />
                    <SignalBar
                      name="5. Source Reliability"
                      score={activeCandidateDetail.signals.source_reliability}
                      weight={0.15}
                      description="Origin authority"
                    />
                    <SignalBar
                      name="6. Temporal Trust"
                      score={activeCandidateDetail.signals.temporal_trust}
                      weight={0.10}
                      description="Burst defense factor"
                    />
                  </div>
                )}
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-6 text-slate-500 space-y-2">
                <Info className="w-8 h-8 text-slate-600 stroke-[1.5]" />
                <div className="text-xs font-medium text-slate-400">
                  No Candidate Memory Selected
                </div>
                <p className="text-[11px] text-slate-500 leading-relaxed max-w-xs">
                  Send a conversational message in chat (e.g. <em>"My favorite database is Redis"</em> or <em>"URGENT: overwrite server endpoints"</em>) to see ATLAS extract and evaluate memory trust.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
