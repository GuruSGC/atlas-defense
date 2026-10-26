"""
Experiment Runner for ATLAS Benchmark Evaluation.
Runs empirical test suites against the real ATLAS trust engine and returns calculated metrics.

Protocols
  pipeline_static       every item is scored against the seeded memory store. The candidate is built
                        the way the live pipeline builds it: all items arrive through the same user
                        channel, and the category comes from the memory extractor, not from the label.
  pipeline_streaming    the 120 primary items arrive interleaved. Accepted memories are written to the
                        store (so it grows), and the anomaly detector refits as it grows.
  pipeline_unauth       like pipeline_static, but attacks arrive through an untrusted channel
  label_informed_legacy source and category are taken from the ground-truth label. This is how the first
                        version of the benchmark built candidates. It leaks the answer, and is kept only
                        to show how much it inflated the result.

All runs use the offline deterministic embedder and extractor unless live=True, so the numbers
in the report can be reproduced without an API key. A run never changes the live application's
memories, anomaly detector, or insertion history: all three are swapped out for the run.
"""

import contextlib
import datetime
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np

from core.embeddings import embedding_service
from core.gemini_client import gemini_client
from memory.memory_models import CandidateMemory, MemorySource, MemoryStatus
from memory.memory_extractor import memory_extractor
from memory.vector_store import InMemoryVectorStore, global_vector_store, use_store
from agent.memory_manager import MemoryManager
from atlas.trust_engine import trust_engine, TrustEngine
from atlas.anomaly_detector import anomaly_detector
from atlas.scoring import scoring_engine
from evaluation import stats
from evaluation.baselines import AtlasDefense, NoDefense, ProvenanceGate, StaticPatternModeration
from evaluation.datasets import DATASET_SEED, Item, load_sets, primary_benchmark
from evaluation.metrics import (
    AttackLabRow, BaselineRow, DatasetInfo, EvaluationReport, EvaluationSummary, Interval,
    ProtocolResult, SeedSweep, SetResult, compute_metrics,
)

# Simulated seconds between submissions. It is wider than the 60 s temporal window so that an
# evaluation run models a normal conversational pace instead of one long burst.
SPACING = 120.0

BENCHMARK_DATASET: List[Dict[str, Any]] = [
    {"text": i.text, "is_attack": i.is_attack, "category": i.category, "context": i.context}
    for i in primary_benchmark()
]

BENCHMARK_DATASET_INFO = DatasetInfo(
    name="ATLAS Benchmark v3",
    version="3.0.0",
    benign_source="PersonaChat (Zhang et al., 2018): 60 real, human-written persona statements; plus 40 technical and 20 update statements written by the authors",
    adversarial_source="Literature-derived across 5 WhisperBench risk categories (MemGhost, arXiv:2607.05189): 60 development cases, 40 held-out cases, and paraphrased evasion cases, all written by the authors",
    seed=DATASET_SEED,
)


@dataclass
class Decision:
    item: Item
    quarantined: bool
    trust: float
    vetoed: bool


@contextlib.contextmanager
def offline_mode(enabled: bool = True) -> Iterator[None]:
    """Force the deterministic embedder and extractor, restoring the previous mode afterwards."""
    old = (embedding_service.is_real_mode, gemini_client.is_real_mode)
    if enabled:
        embedding_service.is_real_mode = False
        gemini_client.is_real_mode = False
    try:
        yield
    finally:
        embedding_service.is_real_mode, gemini_client.is_real_mode = old


@contextlib.contextmanager
def isolated_state(seed: int, engine: Optional[TrustEngine] = None) -> Iterator[InMemoryVectorStore]:
    """A fresh seeded store, a baseline anomaly detector, and an empty insertion history."""
    engine = engine or trust_engine
    saved_detector = (anomaly_detector.model, anomaly_detector._is_fitted, anomaly_detector.fit_count,
                      anomaly_detector._accepted_at_last_fit, anomaly_detector.seed, anomaly_detector._baseline)
    saved_events = list(engine.temporal._events)
    anomaly_detector.reset(seed)
    engine.temporal.reset()
    store = InMemoryVectorStore(seed_demo_data=True)
    try:
        with use_store(store):
            yield store
    finally:
        (anomaly_detector.model, anomaly_detector._is_fitted, anomaly_detector.fit_count,
         anomaly_detector._accepted_at_last_fit, anomaly_detector.seed, anomaly_detector._baseline) = saved_detector
        engine.temporal._events.clear()
        engine.temporal._events.extend(saved_events)


class ExperimentRunner:
    """
    Executes benchmark suites and generates the metrics used in the report.
    """

    def __init__(self, engine: Optional[TrustEngine] = None, seed: int = DATASET_SEED):
        self.engine = engine or trust_engine
        self.seed = seed
        self._cached_results: Optional[EvaluationSummary] = None
        self._cached_report: Optional[EvaluationReport] = None

    # ------------------------------------------------------------------ candidates
    def _candidate(self, item: Item, protocol: str) -> CandidateMemory:
        if protocol == "label_informed_legacy":
            source = MemorySource.ATTACK_SIMULATOR if item.is_attack else MemorySource.USER_CONVERSATION
            category = "instruction" if item.is_attack else "fact"
        else:
            source = MemorySource.ATTACK_SIMULATOR if (protocol == "pipeline_unauth" and item.is_attack) \
                else MemorySource.USER_CONVERSATION
            extracted = memory_extractor.extract_from_turn(item.context)
            category = extracted.category if extracted else "fact"
        return CandidateMemory(
            id=f"eval-{item.id}", content=item.text, category=category, extracted_from_turn=item.context,
            source=source, confidence=0.90, should_persist=True,
            embedding=embedding_service.get_embedding(item.text),
        )

    # ------------------------------------------------------------------ runs
    def run_static(self, items: List[Item], protocol: str = "pipeline_static",
                   seed: Optional[int] = None, live: bool = False) -> List[Decision]:
        seed = self.seed if seed is None else seed
        out: List[Decision] = []
        with offline_mode(not live), isolated_state(seed, self.engine):
            for i, item in enumerate(items):
                cand = self._candidate(item, protocol)
                score, status, signals, _, _ = self.engine.evaluate_candidate(
                    cand, context_text=item.context, now=i * SPACING, record_event=True)
                vetoed = bool(scoring_engine.evaluate_veto(signals)[1])
                out.append(Decision(item, status == MemoryStatus.QUARANTINED, score, vetoed))
        return out

    def run_streaming(self, items: List[Item], seed: Optional[int] = None, live: bool = False) -> List[Decision]:
        seed = self.seed if seed is None else seed
        order = list(items)
        random.Random(seed).shuffle(order)
        out: List[Decision] = []
        with offline_mode(not live), isolated_state(seed, self.engine) as store:
            manager = MemoryManager(vector_store=store, engine=self.engine)
            for i, item in enumerate(order):
                cand = self._candidate(item, "pipeline_streaming")
                mem, score, status, _ = manager.process_candidate_memory(
                    cand, context_text=item.context, enable_atlas=True, now=i * SPACING)
                vetoed = bool(scoring_engine.evaluate_veto(mem.signals)[1])
                out.append(Decision(item, status == MemoryStatus.QUARANTINED, score, vetoed))
        return out

    # ------------------------------------------------------------------ summaries
    @staticmethod
    def confusion(decisions: List[Decision]) -> Tuple[int, int, int, int]:
        tp = sum(1 for d in decisions if d.item.is_attack and d.quarantined)
        fn = sum(1 for d in decisions if d.item.is_attack and not d.quarantined)
        fp = sum(1 for d in decisions if not d.item.is_attack and d.quarantined)
        tn = sum(1 for d in decisions if not d.item.is_attack and not d.quarantined)
        return tp, fp, tn, fn

    @staticmethod
    def _interval(pair: Tuple[float, float]) -> Interval:
        return Interval(low=round(pair[0], 3), high=round(pair[1], 3))

    def _intervals(self, decisions: List[Decision]) -> Dict[str, Interval]:
        tp, fp, tn, fn = self.confusion(decisions)
        labels = [d.item.is_attack for d in decisions]
        flagged = [d.quarantined for d in decisions]
        return {
            "precision": self._interval(stats.wilson_interval(tp, tp + fp)),
            "recall": self._interval(stats.wilson_interval(tp, tp + fn)),
            "fpr": self._interval(stats.wilson_interval(fp, fp + tn)),
            "f1": self._interval(stats.bootstrap_interval(labels, flagged, "f1", seed=self.seed)),
        }

    def _summary(self, decisions: List[Decision], protocol: str) -> EvaluationSummary:
        tp, fp, tn, fn = self.confusion(decisions)
        counts: Dict[str, Dict[str, int]] = {}
        for d in decisions:
            if not d.item.is_attack:
                continue
            c = counts.setdefault(d.item.category, {"total": 0, "detected": 0, "accepted": 0})
            c["total"] += 1
            c["detected" if d.quarantined else "accepted"] += 1
        breakdown = [{
            "category": k, "total_trials": v["total"], "quarantined_count": v["detected"],
            "accepted_count": v["accepted"], "defense_rate": round(v["detected"] / v["total"], 4),
        } for k, v in counts.items()]
        attacks = tp + fn
        return compute_metrics(
            tp=tp, fp=fp, tn=tn, fn=fn,
            asr_without=1.0,                     # with the defense off every write is persisted
            asr_with=(fn / attacks) if attacks else 0.0,
            retrieval_acc=(tn / (tn + fp)) if (tn + fp) else 1.0,
            scenario_stats=breakdown, dataset_info=BENCHMARK_DATASET_INFO,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            protocol=protocol, intervals=self._intervals(decisions),
        )

    def run_benchmark(self, live: bool = False) -> EvaluationSummary:
        """The 120-item primary benchmark under the leakage-free static protocol."""
        self._cached_results = self._summary(self.run_static(primary_benchmark(self.seed), live=live), "pipeline_static")
        return self._cached_results

    def get_summary(self) -> EvaluationSummary:
        return self._cached_results if self._cached_results is not None else self.run_benchmark()

    # ------------------------------------------------------------------ full report
    def _protocol_result(self, name: str, description: str, decisions: List[Decision]) -> ProtocolResult:
        tp, fp, tn, fn = self.confusion(decisions)
        iv = self._intervals(decisions)
        recall = tp / (tp + fn) if tp + fn else 1.0
        precision = tp / (tp + fp) if tp + fp else 1.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        from evaluation.metrics import ConfusionMatrix
        return ProtocolResult(
            protocol=name, description=description,
            confusion_matrix=ConfusionMatrix(true_positives=tp, false_positives=fp, true_negatives=tn, false_negatives=fn),
            precision=round(precision, 3), recall=round(recall, 3), f1_score=round(stats.f1_from_counts(tp, fp, fn), 3),
            false_positive_rate=round(fpr, 3),
            precision_ci=iv["precision"], recall_ci=iv["recall"], f1_ci=iv["f1"], false_positive_rate_ci=iv["fpr"],
            auc=round(stats.auc_from_scores([d.trust for d in decisions if not d.item.is_attack],
                                            [d.trust for d in decisions if d.item.is_attack]), 3),
        )

    def _set_result(self, name: str, description: str, decisions: List[Decision]) -> SetResult:
        attack = decisions[0].item.is_attack if decisions else True
        flagged = sum(1 for d in decisions if d.quarantined)
        by_tech: Dict[str, Dict[str, int]] = {}
        if attack:
            for d in decisions:
                t = by_tech.setdefault(d.item.technique or "unlabeled", {"total": 0, "flagged": 0})
                t["total"] += 1
                t["flagged"] += int(d.quarantined)
        return SetResult(
            name=name, description=description, kind="attack" if attack else "benign", n=len(decisions),
            flagged=flagged, rate=round(flagged / len(decisions), 3) if decisions else 0.0,
            rate_ci=self._interval(stats.wilson_interval(flagged, len(decisions))), by_technique=by_tech,
        )

    def _baseline_rows(self, sets_dec: Dict[str, List[Decision]], unauth_dec: Dict[str, List[Decision]]) -> List[BaselineRow]:
        sets = load_sets()
        rows: List[BaselineRow] = []

        def rate(defense, items: List[Item], attacker_signed: bool) -> float:
            signed = attacker_signed if items and items[0].is_attack else True
            return sum(1 for it in items if defense.blocks(it.text, it.context, signed)) / len(items) if items else 0.0

        for defense in (NoDefense(), StaticPatternModeration(), ProvenanceGate()):
            for attacker, signed in (("unauthenticated", False), ("authenticated", True)):
                rows.append(BaselineRow(
                    defense=defense.name, attacker=attacker,
                    dev_recall=round(rate(defense, sets["dev_adversarial"], signed), 3),
                    heldout_recall=round(rate(defense, sets["heldout_adversarial"], signed), 3),
                    false_positive_rate_personachat=round(rate(defense, sets["benign_personachat"], True), 3),
                    false_positive_rate_technical=round(rate(defense, sets["benign_technical"], True), 3),
                    false_positive_rate_updates=round(rate(defense, sets["legitimate_updates"], True), 3),
                ))

        def flagged(dec: List[Decision]) -> float:
            return round(sum(1 for d in dec if d.quarantined) / len(dec), 3) if dec else 0.0

        for attacker, source in (("unauthenticated", unauth_dec), ("authenticated", sets_dec)):
            rows.append(BaselineRow(
                defense="ATLAS", attacker=attacker,
                dev_recall=flagged(source["dev_adversarial"]), heldout_recall=flagged(source["heldout_adversarial"]),
                false_positive_rate_personachat=flagged(sets_dec["benign_personachat"]),
                false_positive_rate_technical=flagged(sets_dec["benign_technical"]),
                false_positive_rate_updates=flagged(sets_dec["legitimate_updates"]),
            ))
        return rows

    def run_attack_lab(self, live: bool = False) -> List[AttackLabRow]:
        from attacks.scenarios import ATTACK_SCENARIOS
        from attacks.poisoning_simulator import poisoning_simulator
        saved = global_vector_store.list_memories()
        rows: List[AttackLabRow] = []
        try:
            with offline_mode(not live):
                for sc in [s for s in ATTACK_SCENARIOS if s.is_active]:
                    off = poisoning_simulator.run_simulation(sc.id, enable_atlas=False)
                    on = poisoning_simulator.run_simulation(sc.id, enable_atlas=True)
                    rows.append(AttackLabRow(
                        scenario_id=sc.id, name=sc.name, category=sc.category,
                        attack_succeeded_without_atlas=off.attack_succeeded,
                        attack_succeeded_with_atlas=on.attack_succeeded,
                        persisted_with_atlas=on.stored_in_vector_store,
                        retrieved_with_atlas=on.is_retrieved_in_probe,
                        behavior_compromised_with_atlas=on.is_behavior_compromised,
                        trust_score_with_atlas=round(on.trust_score, 3),
                    ))
        finally:
            global_vector_store.clear()
            for m in saved:
                global_vector_store.add_memory(m)
            anomaly_detector.reset()
            self.engine.temporal.reset()
        return rows

    def run_full_report(self, seeds: int = 10, live: bool = False) -> EvaluationReport:
        primary = primary_benchmark(self.seed)
        sets = load_sets()

        static = self.run_static(primary, "pipeline_static", live=live)
        streaming = self.run_streaming(primary, live=live)
        legacy = self.run_static(primary, "label_informed_legacy", live=live)
        unauth = self.run_static(primary, "pipeline_unauth", live=live)

        protocols = {
            "label_informed_legacy": self._protocol_result(
                "label_informed_legacy",
                "Source and category set from the ground-truth label. Leaks the answer; shown only to measure the inflation.", legacy),
            "pipeline_static": self._protocol_result(
                "pipeline_static",
                "Primary protocol: same channel for every item, category from the extractor, scored against the seeded store.", static),
            "pipeline_streaming": self._protocol_result(
                "pipeline_streaming",
                "Items arrive interleaved; accepted memories accumulate and the anomaly detector refits as the store grows.", streaming),
            "pipeline_unauth": self._protocol_result(
                "pipeline_unauth",
                "As the primary protocol, but attacks arrive through an untrusted channel (lower source reliability).", unauth),
        }

        descriptions = {
            "dev_adversarial": "60 adversarial cases used while building the detectors",
            "heldout_adversarial": "40 adversarial cases written after a first round of detector work; their misses then guided new detectors, so this set is optimistic",
            "evasion_adversarial": "paraphrased attacks written after the detectors were frozen, to test robustness to rewording",
            "benign_personachat": "60 casual statements sampled from PersonaChat",
            "benign_technical": "40 developer-style statements, including hard negatives with words like always, never, approved",
            "legitimate_updates": "20 honest user updates that use revision language",
        }
        set_dec: Dict[str, List[Decision]] = {}
        set_dec_unauth: Dict[str, List[Decision]] = {}
        for name, items in sets.items():
            if not items:
                continue
            set_dec[name] = self.run_static(items, "pipeline_static", live=live)
            if items[0].is_attack:
                set_dec_unauth[name] = self.run_static(items, "pipeline_unauth", live=live)
        set_results = {n: self._set_result(n, descriptions[n], d) for n, d in set_dec.items()}

        scored = [(d.trust, d.vetoed, d.item.is_attack) for d in static]
        sweep = stats.threshold_sweep(scored, [round(0.30 + 0.05 * k, 2) for k in range(13)])

        base_rows = self._baseline_rows(set_dec, {**set_dec, **set_dec_unauth})

        seed_list = list(range(seeds))
        series: Dict[str, List[float]] = {k: [] for k in
                                          ["static_precision", "static_recall", "static_f1", "static_fpr",
                                           "streaming_precision", "streaming_recall", "streaming_f1", "streaming_fpr"]}
        for sd in seed_list:
            for tag, dec in (("static", self.run_static(primary, "pipeline_static", seed=sd, live=live)),
                             ("streaming", self.run_streaming(primary, seed=sd, live=live))):
                tp, fp, tn, fn = self.confusion(dec)
                series[f"{tag}_precision"].append(tp / (tp + fp) if tp + fp else 1.0)
                series[f"{tag}_recall"].append(tp / (tp + fn))
                series[f"{tag}_f1"].append(stats.f1_from_counts(tp, fp, fn))
                series[f"{tag}_fpr"].append(fp / (fp + tn))
        seed_sweep = SeedSweep(seeds=seed_list, metrics={k: stats.mean_sd_range(v) for k, v in series.items()})

        notes = [
            "Benchmarks are small and written by the authors, apart from the 60 PersonaChat statements.",
            "The held-out set is optimistic: its misses under the original detectors were read and shaped the new detector families, and the same authors wrote it. The paraphrase set is the cleaner test of generalisation.",
            "The detectors depend on wording: paraphrased attacks are flagged far less often than the originals.",
            "Plausible false facts with no stored memory to contradict cannot be detected from content alone.",
            "Attack success in the benchmark counts persistence only; the Attack Lab measures persistence, retrieval, and behavior.",
            "Baselines are our own plain re-implementations of the published ideas, not the original authors' code.",
        ]
        report = EvaluationReport(
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            dataset_info=BENCHMARK_DATASET_INFO, protocols=protocols, sets=set_results,
            threshold_sweep=sweep, baselines=base_rows, seed_sweep=seed_sweep,
            attack_lab=self.run_attack_lab(live=live), notes=notes,
        )
        self._cached_report = report
        return report

    def get_report(self) -> EvaluationReport:
        return self._cached_report if self._cached_report is not None else self.run_full_report()


# Global runner instance
experiment_runner = ExperimentRunner()
