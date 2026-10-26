"""
Cost of the defense: wall-clock time per candidate memory for each component and for the whole pipeline,
measured on 200 held-out items on this machine's CPU (no GPU), one item at a time.
"""

import time
from typing import Callable, Dict, List

import numpy as np

from atlas.anomaly_detector import anomaly_detector
from atlas.learned import get_poison_classifier, set_poison_classifier
from atlas.nli import NLIContradictionChecker
from core.embeddings import embedding_service
from evaluation.configs import BY_NAME, applied
from evaluation.datasets import Item, benchmark_items
from evaluation.experiment_runner import ExperimentRunner, isolated_state, offline_mode
from evaluation.ladder_eval import load_classifiers, profile_embeddings
from evaluation.v4_common import write_result

N = 200


def _time(fn: Callable[[str], object], texts: List[str]) -> dict:
    ts = []
    for t in texts:
        t0 = time.perf_counter()
        fn(t)
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return {"mean_ms": round(float(a.mean()), 2), "p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2)}


def run() -> dict:
    items = sorted(benchmark_items(split="test"), key=lambda i: i.id)[:N]
    texts = [i.text for i in items]
    profile = profile_embeddings()
    clfs = load_classifiers()
    out: Dict[str, dict] = {"n": len(texts), "device": "CPU", "components": {}, "pipelines": {}}
    with applied(BY_NAME["full"], profile, clfs), offline_mode(True):
        embs = {t: embedding_service.get_embedding(t) for t in texts}
        out["components"]["sentence embedding (MiniLM)"] = _time(embedding_service.get_embedding, texts)
        out["components"]["anomaly signal with certification (kNN, 150 noisy copies)"] = _time(
            lambda t: anomaly_detector.compute_certified_anomaly_score(embs[t], seed=1), texts)
        clf = clfs.get("finetuned")
        if clf is not None:
            out["components"]["fine-tuned classifier"] = _time(lambda t: clf.probability([t]), texts)
        checker = NLIContradictionChecker()
        if checker._load():
            out["components"]["NLI cross-encoder, 5 memory pairs"] = _time(
                lambda t: checker.contradiction_probabilities([("The user likes hiking.", t)] * 5), texts)
        runner = ExperimentRunner()
        for name in ("original", "full"):
            pass
    for name in ("original", "profile_knn", "learned_behaviour", "full"):
        cfg = BY_NAME[name]
        with applied(cfg, profile, clfs), offline_mode(True), isolated_state(42) as _:
            runner = ExperimentRunner()
            def one(t):
                it = Item("l", t, t, True, "latency")
                cand = runner._candidate(it, "pipeline_static")
                runner.engine.evaluate_candidate(cand, context_text=t, now=0.0, record_event=False)
            one(texts[0])        # warm up the models
            out["pipelines"][name] = _time(one, texts)
    write_result("latency", out)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1))
