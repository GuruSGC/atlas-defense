import os

# Tests run offline and deterministically: no API key, no network.
os.environ["GEMINI_API_KEY"] = ""

import pytest

from core.embeddings import embedding_service
from core.gemini_client import gemini_client
from memory.memory_models import CandidateMemory, MemorySource
from memory.vector_store import InMemoryVectorStore, global_vector_store, use_store
from atlas.anomaly_detector import anomaly_detector
from atlas.trust_engine import trust_engine

embedding_service.is_real_mode = False
gemini_client.is_real_mode = False


@pytest.fixture
def store():
    """A fresh seeded in-memory store, active for the trust signals during the test."""
    s = InMemoryVectorStore(seed_demo_data=True)
    with use_store(s):
        yield s


@pytest.fixture
def clean_state(store):
    """Baseline anomaly detector and empty insertion history, restored afterwards."""
    saved = (anomaly_detector.model, anomaly_detector._is_fitted, anomaly_detector.fit_count,
             anomaly_detector._accepted_at_last_fit, anomaly_detector.seed, anomaly_detector._baseline)
    saved_events = list(trust_engine.temporal._events)
    anomaly_detector.reset(42)
    trust_engine.temporal.reset()
    yield store
    (anomaly_detector.model, anomaly_detector._is_fitted, anomaly_detector.fit_count,
     anomaly_detector._accepted_at_last_fit, anomaly_detector.seed, anomaly_detector._baseline) = saved
    trust_engine.temporal._events.clear()
    trust_engine.temporal._events.extend(saved_events)


@pytest.fixture
def global_store_guard():
    """Restore the application's global store after a test that resets it (Attack Lab)."""
    saved = global_vector_store.list_memories()
    yield
    global_vector_store.clear()
    for m in saved:
        global_vector_store.add_memory(m)
    anomaly_detector.reset()
    trust_engine.temporal.reset()


def make_candidate(text, category="fact", source=MemorySource.USER_CONVERSATION, context=None):
    return CandidateMemory(
        id=f"t-{abs(hash(text)) % 10**8}", content=text, category=category,
        extracted_from_turn=context or text, source=source, confidence=0.9,
        embedding=embedding_service.get_embedding(text),
    )
