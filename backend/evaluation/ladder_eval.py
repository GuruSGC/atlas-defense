"""
The ablation ladder: the same held-out items scored by each ATLAS configuration, from the original
prototype to the full learned configuration. Per-item signal values are saved so the sensitivity
analyses can recombine them without running the pipeline again.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from atlas.learned import PoisonClassifier
from core.settings import BASE_DIR
from evaluation.configs import LADDER, AtlasConfig, applied
from evaluation.datasets import Item, benchmark_items
from evaluation.experiment_runner import ExperimentRunner
from evaluation.v4_common import local_embeddings, metrics, per_source, write_result


def load_classifiers() -> Dict[str, PoisonClassifier]:
    found = {}
    for key, folder in (("finetuned", "models/poison_clf"), ("logreg", "models/poison_logreg")):
        path = BASE_DIR / folder
        if (path / "meta.json").exists():
            found[key] = PoisonClassifier.load(str(path))
    return found


def profile_embeddings() -> np.ndarray:
    train = [i for i in benchmark_items(split="train") if not i.is_attack]
    return local_embeddings([i.text for i in train])


def run_config(cfg: AtlasConfig, items: List[Item], runner: ExperimentRunner, profile: np.ndarray,
               classifiers: Dict[str, PoisonClassifier], seed: int = 42):
    with applied(cfg, profile, classifiers):
        t0 = time.time()
        dec = runner.run_static(items, seed=seed)
        return dec, time.time() - t0


def summarise(items: List[Item], dec) -> dict:
    y = np.array([d.item.is_attack for d in dec])
    flagged = np.array([d.quarantined for d in dec])
    scores = 1.0 - np.array([d.trust for d in dec])
    out = metrics(y, flagged, scores)
    out["per_source"] = per_source([d.item for d in dec], flagged)
    out["vetoed_share"] = round(float(np.mean([d.vetoed for d in dec])), 4)
    return out


def run(configs: Optional[List[AtlasConfig]] = None, split: str = "test") -> dict:
    items = sorted(benchmark_items(split=split), key=lambda i: (i.source, i.id))
    runner = ExperimentRunner()
    profile = profile_embeddings()
    classifiers = load_classifiers()
    result = {"split": split, "n": len(items), "configs": {}}
    saved = {}
    for cfg in configs or LADDER:
        if cfg.classifier and cfg.classifier not in classifiers:
            print("skip", cfg.name, "(classifier not trained)")
            continue
        dec, secs = run_config(cfg, items, runner, profile, classifiers)
        row = summarise(items, dec)
        row.update({"description": cfg.description, "seconds_per_item": round(secs / len(items), 4)})
        result["configs"][cfg.name] = row
        saved[cfg.name] = [{"id": d.item.id, "source": d.item.source, "y": d.item.is_attack, "trust": d.trust,
                            "quarantined": d.quarantined, "signals": d.signals} for d in dec]
        print(f"{cfg.name:22s} recall={row['recall']} fpr={row['fpr']} f1={row['f1']} auc={row.get('auc')} ({secs / len(items):.3f}s/item)", flush=True)
    write_result("ladder", result)
    (Path(__file__).resolve().parent / "results" / "v4" / "ladder_items.json").write_text(json.dumps(saved), encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
