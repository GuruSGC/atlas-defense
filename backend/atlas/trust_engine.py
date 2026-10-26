"""
ATLAS Trust Engine Orchestrator.
Central security pipeline that coordinates embedding generation, six-signal evaluation,
anomaly detection, and scoring to protect persistent vector memory.
"""

import time
from typing import Optional, Dict, Tuple
from core.embeddings import embedding_service
from memory.memory_models import CandidateMemory, TrustSignalsBreakdown, MemoryStatus, MemorySource
from atlas.signals import trust_signal_evaluator
from atlas.scoring import scoring_engine
from atlas.temporal import TemporalTracker
from atlas.nli import build_nli_checker


class TrustEngine:
    """
    Core trust evaluation pipeline for candidate conversational memories.
    """

    def __init__(self):
        self.signal_evaluator = trust_signal_evaluator
        self.scoring = scoring_engine
        self.temporal = TemporalTracker()
        self.nli_checker = build_nli_checker()

    def evaluate_candidate(
        self,
        candidate: CandidateMemory,
        context_text: Optional[str] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        now: Optional[float] = None,
        record_event: bool = False
    ) -> Tuple[float, MemoryStatus, TrustSignalsBreakdown, Dict[str, float], str]:
        """
        Runs candidate memory through the six ATLAS trust signals.

        `now` is the submission time (seconds) used for the temporal signal; evaluation runs pass
        a simulated clock. `record_event` adds this submission to the insertion history.

        Returns:
            (trust_score, status, signals, weighted_contributions, explanation)
        """
        # Ensure candidate embedding is present
        if candidate.embedding is None or len(candidate.embedding) == 0:
            candidate.embedding = embedding_service.get_embedding(candidate.content)

        # 1. Semantic Relevance
        sem_rel = self.signal_evaluator.evaluate_semantic_relevance(
            candidate.content,
            context_text or candidate.extracted_from_turn
        )

        # 2. Embedding Anomaly (Isolation Forest, randomized-smoothing certified)
        emb_anom, emb_anom_radius = self.signal_evaluator.evaluate_embedding_anomaly(candidate.embedding)

        # 3. Historical Consistency
        hist_cons = self.signal_evaluator.evaluate_historical_consistency(
            candidate.content,
            candidate.embedding,
            nli_checker=self.nli_checker
        )

        # 4. Behavioral Consistency (User profile & preference alignment)
        behav_cons = self.signal_evaluator.evaluate_behavioral_consistency(
            candidate.content,
            candidate.embedding,
            candidate.category
        )

        # 5. Source Reliability
        source_rel = self.signal_evaluator.evaluate_source_reliability(candidate.source, candidate.content)

        # 6. Temporal Trust
        now_ts = time.time() if now is None else now
        recent = self.temporal.recent_count(now_ts) + 1
        temp_trust = self.signal_evaluator.evaluate_temporal_trust(recent)
        if record_event:
            self.temporal.record(now_ts)

        signals = TrustSignalsBreakdown(
            semantic_relevance=round(sem_rel, 4),
            embedding_anomaly=round(emb_anom, 4),
            historical_consistency=round(hist_cons, 4),
            behavioral_consistency=round(behav_cons, 4),
            source_reliability=round(source_rel, 4),
            temporal_trust=round(temp_trust, 4),
            embedding_anomaly_certified_radius=round(emb_anom_radius, 5)
        )

        trust_score, status, weighted_contribs, reason = self.scoring.compute_trust_score(
            signals,
            custom_weights=custom_weights
        )

        return trust_score, status, signals, weighted_contribs, reason


# Global trust engine instance
trust_engine = TrustEngine()
