"""
Memory Manager for ATLAS.
Coordinates retrieval of verified persistent memories and manages candidate memory
lifecycle through the ATLAS Trust Engine.
"""

from typing import List, Optional, Tuple, Dict
from memory.memory_models import (
    CandidateMemory,
    MemoryItem,
    MemoryStatus,
    MemorySource,
    TrustSignalsBreakdown
)
from memory.vector_store import BaseVectorStore, global_vector_store
from core.embeddings import embedding_service
from atlas.trust_engine import trust_engine, TrustEngine
from atlas.anomaly_detector import anomaly_detector


class MemoryManager:
    """
    Manages storage, retrieval, and trust evaluation routing for long-term agent memories.
    """

    def __init__(self, vector_store: Optional[BaseVectorStore] = None, engine: Optional[TrustEngine] = None):
        self.vector_store = vector_store or global_vector_store
        self.trust_engine = engine or trust_engine

    def retrieve_relevant_memories(self, query: str, top_k: int = 3) -> List[MemoryItem]:
        """
        Retrieve only ACCEPTED persistent memories relevant to the query.
        Quarantined memories are never surfaced to the agent prompt context.
        """
        emb = embedding_service.get_embedding(query)
        scored_memories = self.vector_store.search(
            query_embedding=emb,
            top_k=top_k,
            status_filter=MemoryStatus.ACCEPTED
        )
        return [mem for mem, score in scored_memories if score > 0.40]

    def process_candidate_memory(
        self,
        candidate: CandidateMemory,
        context_text: Optional[str] = None,
        enable_atlas: bool = True,
        custom_weights: Optional[Dict[str, float]] = None,
        now: Optional[float] = None
    ) -> Tuple[MemoryItem, float, MemoryStatus, str]:
        """
        Process a candidate memory:
        - If enable_atlas=True: Evaluate through 6 ATLAS signals -> ACCEPT / QUARANTINE.
        - If enable_atlas=False: Bypass trust check -> directly ACCEPT into vector store (vulnerable mode).
        
        Returns:
            (saved_memory_item, trust_score, status, decision_reason)
        """
        if enable_atlas:
            trust_score, status, signals, _, reason = self.trust_engine.evaluate_candidate(
                candidate=candidate,
                context_text=context_text,
                custom_weights=custom_weights,
                now=now,
                record_event=True
            )
            anomaly_score = round(1.0 - signals.embedding_anomaly, 4)
        else:
            # Vulnerable baseline mode: Blindly accept unverified memory
            trust_score = 1.0
            status = MemoryStatus.ACCEPTED
            signals = TrustSignalsBreakdown(
                semantic_relevance=1.0,
                embedding_anomaly=1.0,
                historical_consistency=1.0,
                behavioral_consistency=1.0,
                source_reliability=1.0,
                temporal_trust=1.0
            )
            anomaly_score = 0.0
            reason = "ATLAS Defense Disabled: Memory accepted blindly without trust validation."

        memory_item = MemoryItem(
            id=candidate.id,
            content=candidate.content,
            category=candidate.category,
            timestamp=candidate.timestamp,
            source=str(candidate.source.value if hasattr(candidate.source, "value") else candidate.source),
            status=status,
            trust_score=trust_score,
            anomaly_score=anomaly_score,
            signals=signals,
            decision_reason=reason,
            embedding=candidate.embedding or embedding_service.get_embedding(candidate.content)
        )

        # Store in vector store
        self.vector_store.add_memory(memory_item)

        # Keep the anomaly detector's notion of "normal" tied to what this user has accepted
        if enable_atlas and status == MemoryStatus.ACCEPTED:
            anomaly_detector.maybe_refit(self.vector_store.get_all_embeddings(MemoryStatus.ACCEPTED))

        return memory_item, trust_score, status, reason


# Global memory manager instance
memory_manager = MemoryManager()
