"""
ATLAS Six Trust Signals.
Implements the mathematical and heuristic evaluation logic for each orthogonal
security signal in the ATLAS trust engine.

The six signals are:
1. semantic_relevance: Contextual alignment with active conversational turn.
2. embedding_anomaly: Isolation Forest outlier detection in latent embedding space.
3. historical_consistency: Contradiction and direct overwrite detection against persistent memories.
4. behavioral_consistency: Alignment with established user behavioral preferences and interaction history.
5. source_reliability: Configurable authority mapping for input origin channels.
6. temporal_trust: Insertion pacing and burst defense factor.
"""

import math
from typing import List, Optional, Tuple
import numpy as np
from core.settings import settings
from core.embeddings import embedding_service
from atlas.anomaly_detector import anomaly_detector
from atlas import patterns
from atlas.learned import get_poison_classifier
from atlas.nli import get_nli_checker
from memory.vector_store import get_active_store
from memory.memory_models import MemoryStatus, MemorySource

# Domain predicate clusters where asserting distinct items is a factual conflict, but only
# when the candidate and the stored memory describe the same slot (see patterns.shares_slot).
DOMAIN_CLUSTERS = [
    {"python", "java", "rust", "golang", "c++", "typescript", "javascript", "ruby", "c#", "php", "kotlin"},
    {"postgres", "postgresql", "mysql", "mongodb", "redis", "oracle", "sqlite", "dynamodb"},
    {"aws", "gcp", "azure", "kubernetes", "darknet", "malicious-db", "c2", "on-prem"},
    {"us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1", "ap-south-1"},
    {"pep 8", "4-space", "tabs", "camelcase", "snake_case"},
]

# Behavioral flags that justify driving the score to its floor. A conditional or standing
# directive is only decisive on its own; the rest describe a concrete harmful action.
CRITICAL_BEHAVIOR_FLAGS = {
    "conditional_directive", "standing_directive", "future_trigger", "passive_policy_change",
    "authority_with_risky_action", "exfiltration", "unsafe_code", "memory_override",
    "injection_marker", "destructive_command", "security_control_off",
}

# Temporal trust follows an exponential decay in the number of insertions seen inside the
# recent window. One insertion is the normal case; a burst pushes the score toward the floor.
TEMPORAL_CEILING = 0.90
TEMPORAL_FLOOR = 0.40
TEMPORAL_DECAY = 0.35


def _nli_gate(candidate: str, memory: str) -> bool:
    """Which stored memories the NLI model is asked about (see settings.NLI_GATE)."""
    mode = settings.NLI_GATE
    if mode == "none":
        return True
    if mode == "related":
        return patterns.is_related(candidate, memory)
    return patterns.shares_slot(candidate, memory)


def _cluster_values(text_lower: str, cluster: set) -> set:
    return {w for w in cluster if w in text_lower}


class TrustSignals:
    """
    Evaluator for the six ATLAS trust signals.
    Each signal produces a normalized score strictly bounded between 0.0 and 1.0.
    """

    @staticmethod
    def evaluate_semantic_relevance(candidate_text: str, context_text: Optional[str] = None) -> float:
        """
        Signal 1: Semantic Relevance (0.0 to 1.0)
        Evaluates how closely the candidate memory aligns with the active conversational context.
        """
        if not context_text or not context_text.strip():
            return 0.75

        emb_candidate = embedding_service.get_embedding(candidate_text)
        emb_context = embedding_service.get_embedding(context_text)

        similarity = embedding_service.cosine_similarity(emb_candidate, emb_context)
        score = 0.10 + 0.90 * similarity
        return float(np.clip(score, 0.05, 1.0))

    @staticmethod
    def evaluate_embedding_anomaly(candidate_embedding: List[float]) -> Tuple[float, float]:
        """
        Signal 2: Embedding Anomaly (0.0 to 1.0)
        Isolation Forest in latent embedding space, hardened via randomized-smoothing
        certification (see atlas/anomaly_detector.py for the exact scope of the guarantee).
        Higher score = normal/in-distribution, lower score = statistical outlier.

        Returns (score, certified_radius).
        """
        if not candidate_embedding:
            return 0.5, 0.0
        return anomaly_detector.compute_certified_anomaly_score(candidate_embedding)

    @staticmethod
    def evaluate_historical_consistency(
        candidate_text: str,
        candidate_embedding: List[float],
        nli_checker=None,
    ) -> float:
        """
        Signal 3: Historical Consistency (0.0 to 1.0)
        Detects factual contradictions or unauthorized overwrites against accepted memories.
        An optional NLI cross-encoder can add a contradiction probability on top of the rules.
        """
        store = get_active_store()
        accepted_memories = store.list_memories(status=MemoryStatus.ACCEPTED)
        if not accepted_memories:
            return 0.90

        top_similar = store.search(candidate_embedding, top_k=5, status_filter=MemoryStatus.ACCEPTED)

        mode = settings.HIST_MODE
        use_rules = mode in ("clusters", "hybrid")
        semantic_nli = mode in ("nli", "hybrid")
        if semantic_nli and nli_checker is None:
            nli_checker = get_nli_checker()
        legacy_nli = mode == "clusters" and nli_checker is not None

        cand_lower = candidate_text.lower()
        contradiction_penalty = 0.0

        # Semantic NLI mode: memories are picked by embedding similarity, not by keyword slots, and
        # the strongest contradiction among them sets the penalty.
        if semantic_nli and nli_checker is not None:
            near = [m for m, sim in top_similar if sim >= settings.NLI_MIN_SIMILARITY]
            probs = nli_checker.contradiction_probabilities([(m.content, candidate_text) for m in near]) if near else []
            probs = [p for p in probs if p is not None]
            if probs and max(probs) > settings.NLI_THRESHOLD:
                contradiction_penalty += 0.85 * max(probs)

        for mem, sim in top_similar:
            if not (use_rules or legacy_nli):
                break
            mem_lower = mem.content.lower()

            # 1. Same slot, different value: both texts talk about, say, the preferred language
            # and name items from one cluster that do not overlap.
            if use_rules and patterns.shares_slot(candidate_text, mem.content):
                for cluster in DOMAIN_CLUSTERS:
                    cand_vals = _cluster_values(cand_lower, cluster)
                    mem_vals = _cluster_values(mem_lower, cluster)
                    if cand_vals and mem_vals and cand_vals.isdisjoint(mem_vals):
                        contradiction_penalty += 0.85
                        break

            # 2. NLI contradiction probability (optional model). Only for memories about the same
            # thing: NLI models call two unrelated sentences a "contradiction", so asking about
            # unrelated pairs would flag most ordinary statements.
            if legacy_nli and _nli_gate(candidate_text, mem.content):
                p = nli_checker.contradiction_probability(mem.content, candidate_text)
                if p is not None and p > settings.NLI_THRESHOLD:
                    contradiction_penalty += 0.85 * p

        # 3. Revision language ("no longer", "was changed to", ...) about something an accepted
        # memory already covers is treated as a likely unauthorized overwrite.
        if use_rules and contradiction_penalty == 0.0 and patterns.has_revision_language(candidate_text):
            if any(patterns.is_related(candidate_text, mem.content) for mem, _ in top_similar):
                # 0.60, not 0.55: 0.95 - 0.55 lands exactly on 0.40, which misses the strict
                # "< 0.40" veto condition. Found when the live embedding model was tried.
                contradiction_penalty += 0.60

        return float(np.clip(0.95 - contradiction_penalty, 0.05, 1.0))

    @staticmethod
    def evaluate_behavioral_consistency(
        candidate_text: str,
        candidate_embedding: List[float],
        category: str = "fact"
    ) -> float:
        """
        Signal 4: Behavioral Consistency (0.0 to 1.0)
        Measures whether the candidate aligns with established user behavior, and penalizes
        text that tries to steer the assistant (conditional or standing directives), change
        policy in the passive voice, impersonate authority, exfiltrate data, or run unsafe code.
        """
        cand_lower = candidate_text.lower()
        accepted_memories = get_active_store().list_memories(status=MemoryStatus.ACCEPTED)

        behavioral_anomaly_penalty = 0.0

        findings = patterns.directive_findings(candidate_text)
        if findings.flags and any(f in CRITICAL_BEHAVIOR_FLAGS for f in findings.flags):
            behavioral_anomaly_penalty += 0.85

        # An instruction-type candidate that tries to bind future behavior
        if category in ["instruction", "directive"] and behavioral_anomaly_penalty == 0.0:
            if any(marker in cand_lower for marker in ["always", "never", "must", "whenever", "next time"]):
                behavioral_anomaly_penalty += 0.50

        learned_cap = TrustSignals._learned_cap(candidate_text)
        if not accepted_memories:
            return float(min(np.clip(0.85 - behavioral_anomaly_penalty, 0.05, 1.0), learned_cap))

        accepted_embeddings = [m.embedding for m in accepted_memories if m.embedding is not None]
        if accepted_embeddings:
            profile_centroid = np.mean(accepted_embeddings, axis=0)
            norm = np.linalg.norm(profile_centroid)
            if norm > 0:
                profile_centroid = (profile_centroid / norm).tolist()
            profile_sim = embedding_service.cosine_similarity(candidate_embedding, profile_centroid)
        else:
            profile_sim = 0.75

        if behavioral_anomaly_penalty > 0.0:
            base_score = 0.20 - behavioral_anomaly_penalty
        else:
            base_score = 0.20 + 0.80 * profile_sim

        return float(min(np.clip(base_score, 0.05, 1.0), learned_cap))

    @staticmethod
    def _learned_cap(candidate_text: str) -> float:
        """
        Upper bound on behavioural consistency from the learned classifier: 1 - P(attack). Rule-based
        detectors can only lower the score further. Returns 1.0 when the learned path is off or no
        classifier has been trained, so the rules alone decide.
        """
        if settings.BEHAV_MODE == "regex":
            return 1.0
        clf = get_poison_classifier()
        if clf is None:
            return 1.0
        p = clf.predict_proba_one(candidate_text)
        return float(np.clip(1.0 - p, 0.05, 1.0))

    @staticmethod
    def evaluate_source_reliability(source: MemorySource = MemorySource.USER_CONVERSATION, text: str = "") -> float:
        """
        Signal 5: Source Reliability (0.0 to 1.0)
        Trust weight by origin channel, lowered when the text itself claims privileged
        authority or issues a destructive directive that a conversation cannot verify.
        """
        source_key = source.value if hasattr(source, "value") else str(source)
        base_reliability = settings.SOURCE_RELIABILITY_WEIGHTS.get(source_key, 0.70)

        if text:
            findings = patterns.directive_findings(text)
            suspicious = {"authority_with_risky_action", "security_control_off", "injection_marker",
                          "destructive_command", "memory_override", "exfiltration", "unsafe_code"}
            if patterns.has_authority_claim(text) or any(f in suspicious for f in findings.flags):
                base_reliability = min(base_reliability, 0.25)

        return base_reliability

    @staticmethod
    def evaluate_temporal_trust(recent_insertion_count: int = 1) -> float:
        """
        Signal 6: Temporal Trust (0.0 to 1.0)
        Exponential decay in the number of insertions inside the recent window: a single
        insertion scores 0.90, and a burst approaches the 0.40 floor. The count comes from
        atlas.temporal.TemporalTracker.
        """
        n = max(int(recent_insertion_count), 1)
        if n == 1:
            return TEMPORAL_CEILING
        span = TEMPORAL_CEILING - TEMPORAL_FLOOR
        return float(TEMPORAL_FLOOR + span * math.exp(-TEMPORAL_DECAY * (n - 1)))


trust_signal_evaluator = TrustSignals()
