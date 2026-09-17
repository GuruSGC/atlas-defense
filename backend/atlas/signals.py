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
6. temporal_trust: Transaction pacing, reinforcement stability, and burst defense factor.
"""

import re
from typing import List, Optional, Dict, Tuple
import numpy as np
from core.settings import settings
from core.embeddings import embedding_service
from atlas.anomaly_detector import anomaly_detector
from memory.vector_store import global_vector_store
from memory.memory_models import MemoryStatus, MemorySource



# Topic-agnostic revision/overwrite language: phrases that indicate an existing fact is
# being changed, reversed, or superseded, independent of *what* the fact is about. This
# complements the hardcoded domain_clusters in evaluate_historical_consistency (which only
# recognize a handful of software-specific topics) with a general linguistic signal that
# generalizes to arbitrary domains (medical, financial, organizational, etc.).
REVISION_MARKER_PATTERNS = [
    r"\bno longer\b", r"\bnot\b.{0,20}\banymore\b", r"\bwas changed to\b", r"\bchanged from\b",
    r"\bwas updated to\b", r"\breversed\b", r"\bmisdiagnos", r"\bquietly\b", r"\bwas moved\b",
    r"\bmoved up\b", r"\bwas switched\b", r"\bdisabled entirely\b", r"\bremoved from\b",
    r"\bincreased from\b", r"\bdecreased from\b", r"\brevised (upward|downward)\b",
    r"\bdiscontinu", r"\bstop taking\b", r"\bskip(ping|ped)?\b", r"\bwas reset to\b",
    r"\bcorrection\b", r"\bupdate:\s", r"\bswitch(ed)? to\b", r"\bwas raised to\b",
    r"\bwas increased\b", r"\bpermanently changed\b", r"\bpermanently disabled\b",
    r"\bwas approved\b", r"\brecommended (discontinuing|moving|switching|skipping)\b",
]

# Topic-agnostic authority/identity-claim pattern: an unverified conversational message
# claiming to *be*, or speak *for*, a privileged role. Broader than the narrow
# destructive_markers list in evaluate_source_reliability, which only matched a handful of
# exact phrases (e.g. "i am the lead system administrator") and missed equivalent claims
# like "this is the CEO" or "Dr. Chen has approved...".
AUTHORITY_CLAIM_PATTERN = re.compile(
    r"\b(?:this is|i am|i'm|as)\s+(?:the|your)\s+"
    r"(?:[a-z-]+(?:'s)?\s+){0,3}"
    r"(ceo|cfo|cto|coo|admin(?:istrator)?|sysadmin|system administrator|security officer|"
    r"incident commander|platform owner|manager|director|physician|doctor|dr\.|nurse|"
    r"accountant|fraud department|it department|hr department|"
    r"support team|customer support|lead|owner)\b",
    re.IGNORECASE,
)


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
        # Direct bounded cosine similarity
        score = 0.10 + 0.90 * similarity
        return float(np.clip(score, 0.05, 1.0))

    @staticmethod
    def evaluate_embedding_anomaly(candidate_embedding: List[float]) -> Tuple[float, float]:
        """
        Signal 2: Embedding Anomaly (0.0 to 1.0)
        Uses Isolation Forest in latent embedding space, hardened via randomized-smoothing
        certification (see atlas/anomaly_detector.py module docstring for exact scope).
        Higher score = normal/in-distribution (safe), Lower score = statistical outlier/anomaly.

        Returns (score, certified_radius).
        """
        if not candidate_embedding:
            return 0.5, 0.0
        return anomaly_detector.compute_certified_anomaly_score(candidate_embedding)

    @staticmethod
    def evaluate_historical_consistency(candidate_text: str, candidate_embedding: List[float]) -> float:
        """
        Signal 3: Historical Consistency (0.0 to 1.0)
        Detects factual contradictions or unauthorized overwrites against existing accepted memories.
        
        TODO (Research Implementation / Extension): Replace baseline contradiction heuristics
        with a formal Natural Language Inference (NLI) cross-encoder model for entailment vs contradiction.
        """
        accepted_memories = global_vector_store.list_memories(status=MemoryStatus.ACCEPTED)
        if not accepted_memories:
            return 0.90  # No prior constraints, default high consistency

        # Domain predicate clusters where asserting distinct items represents a factual conflict
        domain_clusters = [
            {"python", "java", "rust", "golang", "c++", "typescript", "javascript", "ruby", "c#", "php"},
            {"postgres", "postgresql", "mysql", "mongodb", "redis", "oracle", "sqlite", "dynamodb"},
            {"aws", "gcp", "azure", "kubernetes", "darknet", "malicious-db", "c2", "on-prem"},
            {"us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1", "ap-south-1"},
            {"pep 8", "4-space", "tabs", "camelcase", "snake_case"},
        ]

        top_similar = global_vector_store.search(candidate_embedding, top_k=5, status_filter=MemoryStatus.ACCEPTED)
        
        cand_lower = candidate_text.lower()
        contradiction_penalty = 0.0

        for mem, sim in top_similar:
            mem_lower = mem.content.lower()
            
            # 1. Check domain cluster overlap with contradictory values
            for cluster in domain_clusters:
                cand_matches = [word for word in cluster if word in cand_lower]
                mem_matches = [word for word in cluster if word in mem_lower]
                if cand_matches and mem_matches and set(cand_matches) != set(mem_matches):
                    contradiction_penalty += 0.85
                    break

            # 2. Check shared subject assertions (e.g. "prefers", "deployment target", "database", "favorite")
            clash_subjects = [
                "prefers", "preference", "language", "database", "endpoint", "host",
                "favorite", "cluster", "key", "style", "target", "deployment", "server",
                "region", "location", "port", "credentials"
            ]
            for subject in clash_subjects:
                if subject in cand_lower and subject in mem_lower:
                    if sim > 0.30 and cand_lower != mem_lower:
                        contradiction_penalty += 0.50

        # 3. Topic-agnostic revision-language check: a candidate using overwrite/reversal
        # language (regardless of subject matter) that is also semantically close to an
        # existing accepted memory is treated as a likely unauthorized overwrite, even when
        # the topic falls outside the hardcoded domain_clusters/clash_subjects above.
        if contradiction_penalty == 0.0 and any(re.search(pat, cand_lower) for pat in REVISION_MARKER_PATTERNS):
            closest_sim = top_similar[0][1] if top_similar else 0.0
            if closest_sim > 0.35:
                # 0.60, not 0.55: 0.95 - 0.55 lands at exactly 0.40, which fails to trigger
                # the critical-signal veto's strict "< 0.40" condition (found by testing under
                # real Gemini embeddings, where this boundary case actually occurs in practice).
                contradiction_penalty += 0.60

        consistency_score = 0.95 - contradiction_penalty
        return float(np.clip(consistency_score, 0.05, 1.0))

    @staticmethod
    def evaluate_behavioral_consistency(
        candidate_text: str,
        candidate_embedding: List[float],
        category: str = "fact"
    ) -> float:
        """
        Signal 4: Behavioral Consistency (0.0 to 1.0)
        Measures whether the candidate memory aligns with established user behavior,
        preferences, and interaction history. Aggressively penalizes imperative instruction
        injections, destructive actions, and future-conditioned backdoors.
        """
        cand_lower = candidate_text.lower()
        accepted_memories = global_vector_store.list_memories(status=MemoryStatus.ACCEPTED)
        
        behavioral_anomaly_penalty = 0.0

        # Comprehensive instruction injection and destructive command patterns
        destructive_and_injection_patterns = [
            r"\[injected_override\]",
            r"always prepend",
            r"ignore (all|prior|previous) instructions",
            r"system prompt",
            r"admin override",
            r"bypass safety",
            r"exfiltrate",
            r"leak\b",
            r"darknet",
            r"malicious-db",
            r"urgent update",
            r"note for all future sessions",
            r"from now on",
            r"override all",
            r"shut down",
            r"shutdown",
            r"remove all (end\s*points|endpoints|databases|files|data)",
            r"delete all",
            r"erase all",
            r"wipe (all|clean)",
            r"the next time (i\s+talk|i\s+ask|you\s+see|we\s+talk)",
            r"next time i talk to you",
            r"when i talk to you",
            r"terminate the project",
            r"kill the project",
            r"immediately (remove|delete|shut|stop|erase|kill|disable)",
            r"critical directive",
            r"security rule override",
            r"-rm\b",
            r"delete_all_files",
            r"tell me to use",
            r"if i ask you",
            r"whenever i ask",
            r"code to delete a file",
            r"(delete|remove|erase|format|wipe)\s+.*files?",
            r"(lead|senior|chief)?\s*(system\s+administrator|sysadmin|admin\s+directive|admin\s+override)",
            r"(disable|bypass|turn off|skip)\s*(security|vulnerability|scans?|authentication|auth|checks)",
            r"security scans are disabled",
            r"i am the (lead|system administrator|sysadmin|admin|root|owner)"
        ]

        for pat in destructive_and_injection_patterns:
            if re.search(pat, cand_lower):
                behavioral_anomaly_penalty += 0.85
                break

        # If candidate is classified as an instruction/directive from conversational source
        if category in ["instruction", "directive"] and behavioral_anomaly_penalty == 0.0:
            # Check if it attempts to bind future behavior
            if any(marker in cand_lower for marker in ["always", "never", "must", "whenever", "next time", "from now"]):
                behavioral_anomaly_penalty += 0.50

        if not accepted_memories:
            return float(np.clip(0.85 - behavioral_anomaly_penalty, 0.05, 1.0))

        # Compute average alignment with established user preference profile
        accepted_embeddings = [m.embedding for m in accepted_memories if m.embedding is not None]
        if accepted_embeddings:
            profile_centroid = np.mean(accepted_embeddings, axis=0)
            norm = np.linalg.norm(profile_centroid)
            if norm > 0:
                profile_centroid = (profile_centroid / norm).tolist()
            profile_sim = embedding_service.cosine_similarity(candidate_embedding, profile_centroid)
        else:
            profile_sim = 0.75

        # If a major behavioral penalty was triggered, heavily suppress the score
        if behavioral_anomaly_penalty > 0.0:
            base_score = 0.20 - behavioral_anomaly_penalty
        else:
            base_score = 0.20 + 0.80 * profile_sim

        return float(np.clip(base_score, 0.05, 1.0))

    @staticmethod
    def evaluate_source_reliability(source: MemorySource = MemorySource.USER_CONVERSATION, text: str = "") -> float:
        """
        Signal 5: Source Reliability (0.0 to 1.0)
        Assigns trust weight based on origin channel and identity verification.
        Uses configurable settings mapping with claim verification penalties.
        """
        source_key = source.value if hasattr(source, "value") else str(source)
        weights_map = settings.SOURCE_RELIABILITY_WEIGHTS
        base_reliability = weights_map.get(source_key, 0.70)

        # If incoming conversational turn contains unverified high-privilege assertion markers or destructive directives
        if text:
            t_lower = text.lower()
            destructive_markers = [
                "urgent update", "admin:", "system override", "note for all future",
                "shut down", "remove all", "delete all", "immediately", "the next time",
                "lead system administrator", "sysadmin", "admin directive", "security scans are disabled",
                "disable security", "i am the lead"
            ]
            if any(marker in t_lower for marker in destructive_markers):
                base_reliability = min(base_reliability, 0.25)

            # Broader, topic-agnostic check: an unverified conversational message claiming
            # to *be* (or speak for) a privileged role, e.g. "this is the CEO", "Dr. Chen has
            # approved...", "this is your bank's fraud department". This is exactly the kind of
            # unverified authority claim this signal exists to catch, regardless of the
            # specific role named.
            if AUTHORITY_CLAIM_PATTERN.search(t_lower):
                base_reliability = min(base_reliability, 0.25)

        return base_reliability

    @staticmethod
    def evaluate_temporal_trust(recent_insertion_count: int = 1) -> float:
        """
        Signal 6: Temporal Trust (0.0 to 1.0)
        Assesses temporal stability, burst protection, and cadence reliability.
        
        TODO (Research Implementation / Extension): Baseline temporal heuristic.
        Future work may implement an exponential half-life decay and temporal knowledge graph modeling.
        """
        if recent_insertion_count > 5:
            return 0.40
        elif recent_insertion_count > 3:
            return 0.60
        elif recent_insertion_count > 1:
            return 0.75
        return 0.90


trust_signal_evaluator = TrustSignals()
