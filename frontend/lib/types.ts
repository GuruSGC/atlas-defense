/**
 * ATLAS TypeScript Definitions.
 * Mirrors FastAPI Pydantic request/response models.
 */

export type MemoryStatus = "ACCEPTED" | "QUARANTINED" | "PENDING_ANALYSIS";

export type MemorySource =
  | "USER_CONVERSATION"
  | "AGENT_INFERENCE"
  | "SYSTEM_PROMPT"
  | "ATTACK_SIMULATOR";

export interface TrustSignalsBreakdown {
  semantic_relevance: number;
  embedding_anomaly: number;
  historical_consistency: number;
  behavioral_consistency: number;
  source_reliability: number;
  temporal_trust: number;
  /** Randomized-smoothing certified L2 radius for embedding_anomaly (diagnostic only). */
  embedding_anomaly_certified_radius?: number | null;
}

export interface CandidateMemory {
  id: string;
  content: string;
  category: string;
  extracted_from_turn: string;
  timestamp: string;
  source: MemorySource | string;
  embedding?: number[];
}

export interface MemoryItem {
  id: string;
  content: string;
  category: string;
  timestamp: string;
  source: string;
  status: MemoryStatus;
  trust_score: number;
  anomaly_score: number;
  signals: TrustSignalsBreakdown;
  decision_reason: string;
  embedding?: number[];
  metadata?: Record<string, unknown>;
}

export interface ChatTurn {
  role: "user" | "model" | "assistant";
  content: string;
}

export interface ChatRequest {
  message: string;
  history?: ChatTurn[];
  enable_atlas?: boolean;
}

export interface ChatResponse {
  response: string;
  candidate_memory?: CandidateMemory | null;
  memory_status?: MemoryStatus | null;
  trust_score?: number | null;
  signals?: TrustSignalsBreakdown | null;
  decision_reason?: string | null;
  retrieved_memories: string[];
}

export interface AnalyzeMemoryRequest {
  content: string;
  context?: string;
  source?: string;
  weights?: Record<string, number>;
}

export interface AnalyzeMemoryResponse {
  candidate_id: string;
  content: string;
  trust_score: number;
  status: MemoryStatus;
  signals: TrustSignalsBreakdown;
  weighted_contributions: Record<string, number>;
  decision_reason: string;
  timestamp: string;
}

export interface MemoryListResponse {
  memories: MemoryItem[];
  total: number;
  accepted_count: number;
  quarantined_count: number;
}

export interface AttackScenario {
  id: string;
  name: string;
  category: string;
  severity: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | string;
  description: string;
  attacker_prompt: string;
  candidate_poisoned_memory: string;
  probe_query: string;
  expected_compromise_indicator: string;
  safe_baseline_memory: string;
  is_active: boolean;
}

export interface AttackSimulationResult {
  id: string;
  scenario_id: string;
  scenario_name: string;
  category: string;
  severity: string;
  enable_atlas: boolean;
  attacker_prompt: string;
  candidate_poisoned_memory: CandidateMemory;
  signals: TrustSignalsBreakdown;
  trust_score: number;
  memory_decision: MemoryStatus;
  decision_reason: string;
  stored_in_vector_store: boolean;
  probe_query: string;
  probe_response: string;
  retrieved_memories_for_probe: string[];
  attack_succeeded: boolean;
  defense_successful: boolean;
  timestamp: string;
  is_retrieved_in_probe: boolean;
  is_behavior_compromised: boolean;
  steps?: AttackStep[];
}

export interface AttackStep {
  attacker_turn: string;
  candidate_text: string;
  category: string;
  memory_decision: MemoryStatus;
  trust_score: number;
}

export interface ConfusionMatrix {
  true_positives: number;
  false_positives: number;
  true_negatives: number;
  false_negatives: number;
}

export interface ScenarioBreakdownItem {
  category: string;
  total_trials: number;
  quarantined_count: number;
  accepted_count: number;
  defense_rate: number;
}

export interface DatasetInfo {
  name: string;
  version: string;
  benign_source: string;
  adversarial_source: string;
  seed: number;
}

export interface EvaluationSummary {
  total_samples: number;
  attack_samples: number;
  benign_samples: number;
  attack_success_rate_without_atlas: number;
  attack_success_rate_with_atlas: number;
  detection_rate: number;
  false_positive_rate: number;
  precision: number;
  recall: number;
  f1_score: number;
  retrieval_accuracy: number;
  confusion_matrix: ConfusionMatrix;
  scenario_breakdown: ScenarioBreakdownItem[];
  dataset_info: DatasetInfo;
  execution_timestamp: string;
  protocol?: string;
  precision_ci?: Interval | null;
  recall_ci?: Interval | null;
  f1_ci?: Interval | null;
  false_positive_rate_ci?: Interval | null;
}

export interface Interval {
  low: number;
  high: number;
}

export interface ProtocolResult {
  protocol: string;
  description: string;
  confusion_matrix: ConfusionMatrix;
  precision: number;
  recall: number;
  f1_score: number;
  false_positive_rate: number;
  precision_ci: Interval;
  recall_ci: Interval;
  f1_ci: Interval;
  false_positive_rate_ci: Interval;
  auc?: number | null;
}

export interface SetResult {
  name: string;
  description: string;
  kind: "attack" | "benign";
  n: number;
  flagged: number;
  rate: number;
  rate_ci: Interval;
  by_technique: Record<string, { total: number; flagged: number }>;
}

export interface BaselineRow {
  defense: string;
  attacker: "unauthenticated" | "authenticated";
  dev_recall: number;
  heldout_recall: number;
  false_positive_rate_personachat: number;
  false_positive_rate_technical: number;
  false_positive_rate_updates: number;
}

export interface SeedSweep {
  seeds: number[];
  metrics: Record<string, { mean: number; sd: number; min: number; max: number }>;
}

export interface AttackLabRow {
  scenario_id: string;
  name: string;
  category: string;
  attack_succeeded_without_atlas: boolean;
  attack_succeeded_with_atlas: boolean;
  persisted_with_atlas: boolean;
  retrieved_with_atlas: boolean;
  behavior_compromised_with_atlas: boolean;
  trust_score_with_atlas: number;
}

export interface EvaluationReport {
  generated_at: string;
  dataset_info: DatasetInfo;
  protocols: Record<string, ProtocolResult>;
  sets: Record<string, SetResult>;
  threshold_sweep: { threshold: number; precision: number; recall: number; false_positive_rate: number }[];
  baselines: BaselineRow[];
  seed_sweep: SeedSweep;
  attack_lab: AttackLabRow[];
  notes: string[];
}

export interface HealthResponse {
  status: string;
  system: string;
  version: string;
  timestamp: string;
  gemini_api_configured: boolean;
  gemini_model: string;
  trust_threshold: number;
  total_memories: number;
  accepted_memories: number;
  quarantined_memories: number;
}

export interface AtlasConfigResponse {
  threshold: number;
  weights: Record<string, number>;
  signals: {
    id: keyof TrustSignalsBreakdown;
    name: string;
    description: string;
    default_weight: number;
  }[];
}
