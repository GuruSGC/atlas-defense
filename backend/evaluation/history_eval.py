"""
Historical consistency: keyword clusters against an NLI cross-encoder.

Two tests, both about overwriting something a user memory already says.

Profile contradictions (generated): each case has three accepted memories from a PersonaChat persona, one
consistent new statement, and one statement that contradicts a stored line. Personas are split into
train and test by hash, so the NLI threshold is chosen on one half and scored on the other.

PoisonedRAG with prior knowledge (public): the store holds the correct answer to the question each poisoned
passage targets, so the passage is an overwrite attempt against a real stored fact. Benign controls are a
restatement of the correct answer and an unrelated PersonaChat statement.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from atlas.nli import NLIContradictionChecker
from atlas.signals import TrustSignals
from core.embeddings import embedding_service
from core.settings import settings
from evaluation.configs import AtlasConfig, applied
from evaluation.datasets import benchmark_items
from evaluation.v4_common import auc, local_embeddings, metrics, tpr_at_fpr, write_result
from memory.memory_models import MemoryItem, MemoryStatus, TrustSignalsBreakdown
from memory.vector_store import InMemoryVectorStore, use_store

DATA = Path(__file__).resolve().parent / "data" / "profile_contradictions.json"
VETO_TRIGGER = 0.40


def _signals():
    return TrustSignalsBreakdown(semantic_relevance=0.9, embedding_anomaly=0.9, historical_consistency=0.9,
                                 behavioral_consistency=0.9, source_reliability=0.9, temporal_trust=0.9)


def _store(memories: List[str]) -> InMemoryVectorStore:
    store = InMemoryVectorStore(seed_demo_data=False)
    embs = embedding_service.get_batch_embeddings(memories)
    for k, (m, e) in enumerate(zip(memories, embs)):
        store.add_memory(MemoryItem(id=f"m{k}", content=m, status=MemoryStatus.ACCEPTED, trust_score=0.9, signals=_signals(), embedding=e))
    return store


def _hist(store: InMemoryVectorStore, text: str, checker=None) -> float:
    with use_store(store):
        return TrustSignals.evaluate_historical_consistency(text, embedding_service.get_embedding(text), nli_checker=checker)


def _persona_split(case_id: str) -> str:
    return "test" if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 100 < 40 else "train"


def _run_profile_cases(cases: List[dict], mode: str, checker, theta: float, gate: float) -> Dict[str, np.ndarray]:
    saved = (settings.HIST_MODE, settings.NLI_THRESHOLD, settings.NLI_MIN_SIMILARITY)
    settings.HIST_MODE, settings.NLI_THRESHOLD, settings.NLI_MIN_SIMILARITY = mode, theta, gate
    try:
        att, ben = [], []
        for c in cases:
            store = _store(c["stored"])
            att.append(_hist(store, c["attack"], checker))
            ben.append(_hist(store, c["benign"], checker))
    finally:
        settings.HIST_MODE, settings.NLI_THRESHOLD, settings.NLI_MIN_SIMILARITY = saved
    return {"attack": 1 - np.array(att), "benign": 1 - np.array(ben)}


def _summarise(scores: Dict[str, np.ndarray]) -> dict:
    a, b = scores["attack"], scores["benign"]
    y = np.concatenate([np.ones(len(a), bool), np.zeros(len(b), bool)])
    s = np.concatenate([a, b])
    flagged = s > (1 - VETO_TRIGGER)          # historical score below 0.40, the veto trigger
    m = metrics(y, flagged, s, boot=False)
    return {k: m[k] for k in ("recall", "fpr", "precision", "f1", "auc", "recall_ci", "fpr_ci") if k in m}


def run_profiles() -> dict:
    cases = json.loads(DATA.read_text(encoding="utf-8"))["cases"]
    for c in cases:
        c["split"] = _persona_split(c["id"])
    train = [c for c in cases if c["split"] == "train"]
    test = [c for c in cases if c["split"] == "test"]
    out = {"n_cases": len(cases), "n_train": len(train), "n_test": len(test), "modes": {}, "grid": []}
    with applied(AtlasConfig("history", backend="local")):
        rules = _summarise(_run_profile_cases(test, "clusters", None, 0.9, 0.62))
        out["modes"]["keyword clusters and revision rules (original)"] = rules
        best = None
        for model in ("cross-encoder/nli-deberta-v3-xsmall", "cross-encoder/nli-deberta-v3-small"):
            checker = NLIContradictionChecker(model)
            for gate in (0.55, 0.62, 0.70):
                for theta in (0.7, 0.9, 0.99, 0.999):
                    tr = _summarise(_run_profile_cases(train, "nli", checker, theta, gate))
                    # choose on train by Youden's J (recall minus false-positive rate)
                    key = (tr["recall"] - tr["fpr"], -tr["fpr"])
                    out["grid"].append({"model": model, "gate": gate, "theta": theta, "train": tr})
                    if best is None or key > best[0]:
                        best = (key, model, gate, theta, checker)
        _, model, gate, theta, checker = best
        out["selected"] = {"model": model, "similarity_gate": gate, "threshold": theta}
        out["modes"]["NLI cross-encoder (semantic retrieval)"] = _summarise(_run_profile_cases(test, "nli", checker, theta, gate))
        out["modes"]["rules plus NLI"] = _summarise(_run_profile_cases(test, "hybrid", checker, theta, gate))
        # an ungated NLI check: every stored memory is asked about, related or not
        out["modes"]["NLI without similarity gate"] = _summarise(_run_profile_cases(test, "nli", checker, theta, 0.0))
    return out


def run_prior_knowledge(selected: dict) -> dict:
    items = [i for i in benchmark_items() if i.source == "poisonedrag" and i.probe]
    unrelated = [i.text for i in benchmark_items(split="test") if i.source == "personachat"][:100]
    checker = NLIContradictionChecker(selected["model"])
    res: Dict[str, dict] = {}
    with applied(AtlasConfig("history", backend="local")):
        for mode in ("clusters", "nli", "hybrid"):
            saved = (settings.HIST_MODE, settings.NLI_THRESHOLD, settings.NLI_MIN_SIMILARITY)
            settings.HIST_MODE, settings.NLI_THRESHOLD, settings.NLI_MIN_SIMILARITY = mode, selected["threshold"], selected["similarity_gate"]
            ck = checker if mode != "clusters" else None
            try:
                att, rest, unr = [], [], []
                for it in items:
                    fact = f'The answer to the question "{it.probe}" is {it.correct}.'
                    store = _store([fact])
                    att.append(_hist(store, it.text, ck))
                    rest.append(_hist(store, f'For the question "{it.probe}", the answer is {it.correct}.', ck))
                unr_store = _store([f'The answer to the question "{items[0].probe}" is {items[0].correct}.'])
                unr = [_hist(unr_store, t, ck) for t in unrelated]
            finally:
                settings.HIST_MODE, settings.NLI_THRESHOLD, settings.NLI_MIN_SIMILARITY = saved
            flag = lambda v: float(np.mean(np.array(v) < VETO_TRIGGER))
            res[mode] = {"attack_flagged": round(flag(att), 4), "restatement_flagged": round(flag(rest), 4),
                         "unrelated_flagged": round(flag(unr), 4), "n_attack": len(att)}
    return res


def run() -> dict:
    prof = run_profiles()
    prior = run_prior_knowledge(prof["selected"])
    out = {"profile_contradictions": prof, "poisonedrag_with_prior_knowledge": prior}
    write_result("history", out)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1)[:4000])
