"""
The five Attack Lab scenarios with and without the learned classifier.

The Attack Lab runs on the application's own store, which uses the default embedder, so the embedding backend
and the anomaly detector stay at their defaults. Only the behavioural signal changes: rules alone, or rules
capped by the fine-tuned classifier.
"""

import json
from typing import Dict, List

from atlas.learned import get_poison_classifier
from core.settings import settings
from evaluation.experiment_runner import ExperimentRunner
from evaluation.v4_common import write_result


def run() -> dict:
    runner = ExperimentRunner()
    out: Dict[str, List[dict]] = {}
    saved = settings.BEHAV_MODE
    try:
        for label, mode in (("rules only", "regex"), ("rules and fine-tuned classifier", "hybrid")):
            if mode == "hybrid" and get_poison_classifier() is None:
                continue
            settings.BEHAV_MODE = mode
            out[label] = [r.model_dump() for r in runner.run_attack_lab()]
    finally:
        settings.BEHAV_MODE = saved
    result = {"configs": out}
    write_result("attack_lab", result)
    return result


if __name__ == "__main__":
    for k, rows in run()["configs"].items():
        print(k)
        for r in rows:
            print("  ", r["scenario_id"], "undefended:", r["attack_succeeded_without_atlas"], "defended:", r["attack_succeeded_with_atlas"],
                  "P/R/B:", r["persisted_with_atlas"], r["retrieved_with_atlas"], r["behavior_compromised_with_atlas"], "trust", r["trust_score_with_atlas"])
