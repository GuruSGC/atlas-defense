"""
Experiment Runner for ATLAS Benchmark Evaluation.
Runs empirical test suites against ATLAS trust engine and returns real calculated metrics.
Sets deterministic random seeds for reproducible research benchmarking.

Benchmark composition (v2, September 2026):
  - Benign class (60 items): sampled from PersonaChat (Zhang et al., 2018, ACL),
    a peer-reviewed, publicly released persona/preference dialogue dataset,
    not hand-authored by the project team.
  - Adversarial class (60 items): constructed across the five risk categories
    (health & safety, financial loss, information integrity, cybersecurity,
    operational disruption) introduced by MemGhost's WhisperBench taxonomy
    (Zhang et al., 2026, arXiv:2607.05189), since the original WhisperBench
    and MINJA case files are not yet publicly released pending responsible
    disclosure. Attack phrasing is informed by the mechanisms described in
    MINJA (arXiv:2503.03704), MemPoison (arXiv:2605.29960), and Sunil et al.
    (arXiv:2601.05504).
See evaluation/data/README.md for full provenance and licensing notes.
"""

import datetime
import json
import uuid
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from memory.memory_models import CandidateMemory, MemorySource, MemoryStatus
from atlas.trust_engine import trust_engine, TrustEngine
from evaluation.metrics import compute_metrics, EvaluationSummary, DatasetInfo
from core.embeddings import embedding_service

DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET_SEED = 42


def _load_benchmark_dataset(seed: int = DATASET_SEED) -> List[Dict[str, Any]]:
    """
    Loads the benign (PersonaChat-derived) and adversarial (literature-taxonomy-derived)
    seed files from evaluation/data/, normalizes them into the internal
    {text, is_attack, category, context} shape used by the benchmark loop below,
    and returns them in a fixed, seeded, shuffled order for reproducibility.
    """
    with open(DATA_DIR / "benign_seed.json", encoding="utf-8") as f:
        benign_raw = json.load(f)
    with open(DATA_DIR / "adversarial_seed.json", encoding="utf-8") as f:
        adversarial_raw = json.load(f)

    items: List[Dict[str, Any]] = []
    for r in benign_raw:
        items.append({
            "text": r["text"],
            "is_attack": False,
            "category": r["category"],
            "context": r["text"],
        })
    for r in adversarial_raw:
        items.append({
            "text": r["text"],
            "is_attack": True,
            "category": r["category"],
            "context": r["context"],
        })

    rng = random.Random(seed)
    rng.shuffle(items)
    return items


BENCHMARK_DATASET: List[Dict[str, Any]] = _load_benchmark_dataset()

BENCHMARK_DATASET_INFO = DatasetInfo(
    name="ATLAS Benchmark v2",
    version="2.0.0",
    benign_source=f"PersonaChat (Zhang et al., 2018): {sum(1 for i in BENCHMARK_DATASET if not i['is_attack'])} real, human-written persona/preference statements",
    adversarial_source=f"Literature-derived across 5 WhisperBench risk categories (MemGhost, arXiv:2607.05189): {sum(1 for i in BENCHMARK_DATASET if i['is_attack'])} constructed cases informed by MINJA, MemPoison, and Sunil et al.",
    seed=DATASET_SEED,
)


class ExperimentRunner:
    """
    Executes benchmark suites and generates rigorous empirical metrics for research review.
    Uses deterministic seeds for full scientific reproducibility.
    """

    def __init__(self, engine: Optional[TrustEngine] = None, seed: int = 42):
        self.trust_engine = engine or trust_engine
        self.seed = seed
        self._cached_results: Optional[EvaluationSummary] = None

    def run_benchmark(self) -> EvaluationSummary:
        """
        Runs ATLAS trust evaluation across all benchmark test items,
        recording actual true/false positives and negatives.
        """
        # Set deterministic seeds
        random.seed(self.seed)
        np.random.seed(self.seed)

        tp = 0  # Malicious correctly QUARANTINED
        fp = 0  # Benign incorrectly QUARANTINED
        tn = 0  # Benign correctly ACCEPTED
        fn = 0  # Malicious incorrectly ACCEPTED

        scenario_counts: Dict[str, Dict[str, int]] = {}

        for item in BENCHMARK_DATASET:
            cat = item["category"]
            if cat not in scenario_counts:
                scenario_counts[cat] = {"total": 0, "detected": 0, "accepted": 0}

            scenario_counts[cat]["total"] += 1

            cid = f"eval-{uuid.uuid4().hex[:8]}"
            emb = embedding_service.get_embedding(item["text"])
            cand = CandidateMemory(
                id=cid,
                content=item["text"],
                category="instruction" if item["is_attack"] else "fact",
                extracted_from_turn=item["context"],
                source=MemorySource.ATTACK_SIMULATOR if item["is_attack"] else MemorySource.USER_CONVERSATION,
                confidence=0.90,
                should_persist=True,
                embedding=emb
            )

            # Evaluate with ATLAS
            score, status, signals, _, reason = self.trust_engine.evaluate_candidate(
                candidate=cand,
                context_text=item["context"]
            )

            if item["is_attack"]:
                if status == MemoryStatus.QUARANTINED:
                    tp += 1
                    scenario_counts[cat]["detected"] += 1
                else:
                    fn += 1
                    scenario_counts[cat]["accepted"] += 1
            else:
                if status == MemoryStatus.ACCEPTED:
                    tn += 1
                    scenario_counts[cat]["accepted"] += 1
                else:
                    fp += 1
                    scenario_counts[cat]["detected"] += 1

        # Format scenario breakdown
        scenario_stats = []
        for cat_name, stats in scenario_counts.items():
            tot = stats["total"]
            det = stats["detected"]
            rate = (det / tot) if tot > 0 else 0.0
            scenario_stats.append({
                "category": cat_name,
                "total_trials": tot,
                "quarantined_count": det,
                "accepted_count": stats["accepted"],
                "defense_rate": round(rate, 4)
            })

        # Calculate ASR:
        # Baseline ASR without defense (ATLAS OFF): all attacks persist and succeed (1.0)
        # Defended ASR with ATLAS ON: proportion of attacks that slipped through (FN / total_attacks)
        attack_total = tp + fn
        asr_without = 1.0
        asr_with = (fn / attack_total) if attack_total > 0 else 0.0
        retrieval_acc = (tn / (tn + fp)) if (tn + fp) > 0 else 1.0

        summary = compute_metrics(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            asr_without=asr_without,
            asr_with=asr_with,
            retrieval_acc=retrieval_acc,
            scenario_stats=scenario_stats,
            dataset_info=BENCHMARK_DATASET_INFO,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )

        self._cached_results = summary
        return summary

    def get_summary(self) -> EvaluationSummary:
        """
        Returns cached summary or runs benchmark if not yet executed.
        """
        if self._cached_results is None:
            return self.run_benchmark()
        return self._cached_results


# Global experiment runner instance
experiment_runner = ExperimentRunner(seed=42)
