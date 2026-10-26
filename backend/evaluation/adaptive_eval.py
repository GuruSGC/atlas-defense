"""
Runs the adaptive-adversary experiments described in evaluation/adaptive.py.

  python -m evaluation.adaptive_eval sweep       rewrite families, one at a time
  python -m evaluation.adaptive_eval search      greedy black-box search at several query budgets
  python -m evaluation.adaptive_eval retrain     adversarial training, scored on families it did not see
  python -m evaluation.adaptive_eval all
"""

import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from atlas.learned import PoisonClassifier
from core.settings import BASE_DIR
from evaluation import adaptive
from evaluation.configs import BY_NAME, AtlasConfig, applied
from evaluation.datasets import Item, benchmark_items
from evaluation.experiment_runner import ExperimentRunner, isolated_state, offline_mode
from evaluation.ladder_eval import load_classifiers, profile_embeddings
from evaluation.v4_common import read_result, write_result
from evaluation.stats import wilson_interval
from memory.memory_models import MemoryStatus

SEARCH_BUDGETS = [1, 5, 10, 20, 50]
N_SEARCH = 100
N_SWEEP = 250


class PipelineDefense:
    """A defense that can be queried one text at a time with the state held fixed between queries."""

    def __init__(self, cfg: AtlasConfig, profile, classifiers):
        self.cfg, self.profile, self.classifiers = cfg, profile, classifiers
        self.runner = ExperimentRunner()
        self._stack = None

    def __enter__(self):
        import contextlib
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(applied(self.cfg, self.profile, self.classifiers))
        self._stack.enter_context(offline_mode(True))
        self._stack.enter_context(isolated_state(42, self.runner.engine))
        return self

    def __exit__(self, *exc):
        self._stack.close()

    def query(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        scores, flags = [], []
        for k, t in enumerate(texts):
            it = Item(f"q{k}", t, t, True, "adaptive")
            cand = self.runner._candidate(it, "pipeline_static")
            trust, status, _, _, _ = self.runner.engine.evaluate_candidate(cand, context_text=t, now=0.0, record_event=False)
            scores.append(1.0 - trust)
            flags.append(status == MemoryStatus.QUARANTINED)
        return np.array(scores), np.array(flags)


class ClassifierDefense:
    def __init__(self, clf: PoisonClassifier):
        self.clf = clf

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        p = self.clf.probability(texts)
        return p, p > 0.5


def attack_pool() -> List[Item]:
    items = [i for i in benchmark_items(split="test") if i.is_attack and i.source != "poisonedrag"]
    rng = random.Random(11)
    items.sort(key=lambda i: i.id)
    rng.shuffle(items)
    return items


def defenses() -> Dict[str, Callable]:
    profile = profile_embeddings()
    clfs = load_classifiers()
    out = {"ATLAS original (rules, hash embeddings)": lambda: PipelineDefense(BY_NAME["original"], profile, clfs)}
    if "finetuned" in clfs:
        out["ATLAS full (rules, classifier, NLI)"] = lambda: PipelineDefense(BY_NAME["full"], profile, clfs)
        out["classifier only (fine-tuned)"] = lambda: ClassifierDefense(clfs["finetuned"])
    return out


def sweep() -> dict:
    pool = attack_pool()[:N_SWEEP]
    res: Dict[str, dict] = {"n_attacks": len(pool), "systems": {}}
    for name, make in defenses().items():
        with make() as d:
            base_scores, base_flags = d.query([i.text for i in pool])
            row = {"flagged_original": round(float(base_flags.mean()), 4), "families": {}}
            keep = [k for k, f in enumerate(base_flags) if f]
            for fam in adaptive.FAMILIES:
                texts = [adaptive.rewrite(pool[k].text, [fam], 0) for k in keep]
                _, flags = d.query(texts)
                still = int(flags.sum())
                row["families"][fam] = {"still_flagged": still, "of": len(keep), "rate": round(still / max(len(keep), 1), 4),
                                         "ci": [round(v, 4) for v in wilson_interval(still, len(keep))]}
            res["systems"][name] = row
            print(name, {f: v["rate"] for f, v in row["families"].items()}, flush=True)
    return res


def search() -> dict:
    pool = attack_pool()
    res: Dict[str, dict] = {"budgets": SEARCH_BUDGETS, "n_per_system": N_SEARCH, "systems": {}}
    for name, make in defenses().items():
        with make() as d:
            _, flags = d.query([i.text for i in pool])
            flagged = [pool[k] for k, f in enumerate(flags) if f][:N_SEARCH]
            used = []
            for it in flagged:
                r = adaptive.greedy_search(it, d.query, max(SEARCH_BUDGETS), seed=0)
                used.append(r["queries"] if r["evaded"] else None)
            row = {"n_attacked": len(flagged), "evasion_rate": {}}
            for b in SEARCH_BUDGETS:
                k = sum(1 for u in used if u is not None and u <= b)
                row["evasion_rate"][str(b)] = {"evaded": k, "rate": round(k / max(len(flagged), 1), 4),
                                               "ci": [round(v, 4) for v in wilson_interval(k, len(flagged))]}
            res["systems"][name] = row
            print(name, {b: v["rate"] for b, v in row["evasion_rate"].items()}, flush=True)
    return res


def retrain() -> dict:
    items = benchmark_items()
    train = [i for i in items if i.split == "train" and i.source != "poisonedrag"]
    texts = [i.text for i in train]
    labels = [i.is_attack for i in train]
    aug_t, aug_l = [], []
    for k, i in enumerate(train):
        if i.is_attack:
            for fam in adaptive.TRAIN_FAMILIES:
                aug_t.append(adaptive.rewrite(i.text, [fam], k))
                aug_l.append(True)
    base = load_classifiers()["finetuned"]
    robust = PoisonClassifier("finetuned").fit(texts + aug_t, labels + aug_l, epochs=4, seed=0)
    robust.save(str(BASE_DIR / "models" / "poison_clf_robust"))
    test = [i for i in benchmark_items(split="test")]
    attacks = [i for i in test if i.is_attack and i.source != "poisonedrag"]
    benign = [i for i in test if not i.is_attack]
    out: Dict[str, dict] = {"train_families": adaptive.TRAIN_FAMILIES, "heldout_families": adaptive.HELDOUT_FAMILIES,
                            "n_augmented": len(aug_t), "recall": {}, "benign_fpr": {}}
    for name, clf in (("fine-tuned (original training)", base), ("fine-tuned (adversarial training)", robust)):
        out["benign_fpr"][name] = round(float((clf.probability([i.text for i in benign]) > 0.5).mean()), 4)
        rows = {"unmodified": float((clf.probability([i.text for i in attacks]) > 0.5).mean())}
        for fam in adaptive.FAMILIES:
            rw = [adaptive.rewrite(i.text, [fam], 0) for i in attacks]
            rows[fam] = float((clf.probability(rw) > 0.5).mean())
        out["recall"][name] = {k: round(v, 4) for k, v in rows.items()}
        print(name, out["recall"][name], flush=True)
    # greedy search against the retrained classifier
    flagged = [i for i in attacks if robust.probability([i.text])[0] > 0.5][:N_SEARCH]
    for name, clf in (("fine-tuned (original training)", base), ("fine-tuned (adversarial training)", robust)):
        d = ClassifierDefense(clf)
        used = []
        for it in flagged:
            if clf.probability([it.text])[0] <= 0.5:
                continue
            r = adaptive.greedy_search(it, d.query, max(SEARCH_BUDGETS), seed=0)
            used.append(r["queries"] if r["evaded"] else None)
        out.setdefault("search_evasion", {})[name] = {str(b): round(sum(1 for u in used if u is not None and u <= b) / max(len(used), 1), 4)
                                                       for b in SEARCH_BUDGETS}
    return out


def main(argv: List[str]) -> None:
    which = argv[0] if argv else "all"
    parts = read_result("adaptive") or {}
    if which in ("sweep", "all"):
        parts["sweep"] = sweep()
        write_result("adaptive", parts)
    if which in ("search", "all"):
        parts["search"] = search()
        write_result("adaptive", parts)
    if which in ("retrain", "all"):
        parts["retrain"] = retrain()
        write_result("adaptive", parts)


if __name__ == "__main__":
    main(sys.argv[1:])
