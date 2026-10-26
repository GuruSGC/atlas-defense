/**
 * Centralized API client for ATLAS Backend.
 * Decouples all React components from direct fetch calls.
 */

import {
  HealthResponse,
  ChatRequest,
  ChatResponse,
  MemoryListResponse,
  MemoryItem,
  MemoryStatus,
  AnalyzeMemoryRequest,
  AnalyzeMemoryResponse,
  AtlasConfigResponse,
  AttackScenario,
  AttackSimulationResult,
  EvaluationSummary,
  EvaluationReport
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

class ApiClient {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    const headers = {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    };

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status} ${response.statusText}`;
        try {
          const errJson = await response.json();
          if (errJson.detail) {
            errorDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
          }
        } catch {
          // ignore json parse error
        }
        throw new Error(errorDetail);
      }

      return (await response.json()) as T;
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Network error contacting ATLAS backend";
      console.error(`[API Error] ${options.method || "GET"} ${url}:`, message);
      throw new Error(message);
    }
  }

  // Health & Status
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/api/health");
  }

  // Chat Interface
  async sendChatMessage(payload: ChatRequest): Promise<ChatResponse> {
    return this.request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  // Memory Monitor
  async getMemories(status?: MemoryStatus, search?: string): Promise<MemoryListResponse> {
    const params = new URLSearchParams();
    if (status) params.append("status", status);
    if (search) params.append("search", search);
    const queryStr = params.toString() ? `?${params.toString()}` : "";
    return this.request<MemoryListResponse>(`/api/memories${queryStr}`);
  }

  async getMemoryById(id: string): Promise<MemoryItem> {
    return this.request<MemoryItem>(`/api/memories/${encodeURIComponent(id)}`);
  }

  async deleteMemory(id: string): Promise<{ status: string; message: string }> {
    return this.request<{ status: string; message: string }>(`/api/memories/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
  }

  // ATLAS Diagnostic Engine
  async analyzeMemory(payload: AnalyzeMemoryRequest): Promise<AnalyzeMemoryResponse> {
    return this.request<AnalyzeMemoryResponse>("/api/atlas/analyze", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async getAtlasConfig(): Promise<AtlasConfigResponse> {
    return this.request<AtlasConfigResponse>("/api/atlas/config");
  }

  // Attack Lab
  async getAttackScenarios(): Promise<AttackScenario[]> {
    return this.request<AttackScenario[]>("/api/attacks/scenarios");
  }

  async runAttackScenario(scenarioId: string, enableAtlas: boolean = true): Promise<AttackSimulationResult> {
    return this.request<AttackSimulationResult>("/api/attacks/run", {
      method: "POST",
      body: JSON.stringify({
        scenario_id: scenarioId,
        enable_atlas: enableAtlas,
      }),
    });
  }

  async resetAttackLab(): Promise<{ status: string; message: string; total_memories: number }> {
    return this.request<{ status: string; message: string; total_memories: number }>("/api/attacks/reset", {
      method: "POST",
    });
  }

  // Evaluation & Metrics
  async getEvaluationSummary(): Promise<EvaluationSummary> {
    return this.request<EvaluationSummary>("/api/evaluation/summary");
  }

  async runEvaluationBenchmark(): Promise<EvaluationSummary> {
    return this.request<EvaluationSummary>("/api/evaluation/run", {
      method: "POST",
    });
  }

  async getEvaluationReport(): Promise<EvaluationReport> {
    return this.request<EvaluationReport>("/api/evaluation/report");
  }
}

export const api = new ApiClient();
