"""
Shared helpers for the v4 evaluation: cached embeddings, vectorised AUC and bootstrap intervals, and a
single function that turns scores and decisions into the metrics every table uses.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from core.embeddings import embedding_service
from evaluation.datasets import Item, benchmark_items
from evaluation.stats import wilson_interval

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "v4"
CACHE_DIR = Path(__file__).resolve().parent / "data" / "public_cache"


def write_result(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def read_result(name: str) -> Optional[dict]:
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------------------------------- embeddings
def local_embeddings(texts: Sequence[str]) -> np.ndarray:
    """Sentence-transformer embeddings for texts, cached on disk by content hash."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    store = CACHE_DIR / "local_embeddings.npz"
    cache: Dict[str, np.ndarray] = {}
    if store.exists():
        z = np.load(store, allow_pickle=False)
        cache = dict(zip(z["keys"].tolist(), z["vecs"]))
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    missing = sorted({k: t for k, t in zip(keys, texts) if k not in cache}.items())
    if missing:
        vecs = embedding_service._local.encode([t for _, t in missing])
        for (k, _), v in zip(missing, vecs):
            cache[k] = np.array(v, dtype=np.float32)
        np.savez(store, keys=np.array(list(cache.keys())), vecs=np.stack(list(cache.values())))
    return np.stack([cache[k] for k in keys]).astype(float)


# ---------------------------------------------------------------------------------------- statistics
def auc(attack_scores: np.ndarray, benign_scores: np.ndarray) -> float:
    """P(an attack scores higher than a benign item); ties count half. Higher score = more attack-like."""
    a, b = np.asarray(attack_scores, float), np.asarray(benign_scores, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    allv = np.concatenate([a, b])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_vals = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def bootstrap(stat_fn, n: int, iterations: int = 500, seed: int = 7) -> List[float]:
    rng = np.random.default_rng(seed)
    vals = [stat_fn(rng.integers(0, n, n)) for _ in range(iterations)]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def tpr_at_fpr(attack_scores: np.ndarray, benign_scores: np.ndarray, fpr: float) -> float:
    neg = np.sort(np.asarray(benign_scores, float))
    thr = neg[int(np.ceil((1 - fpr) * len(neg))) - 1]
    return float((np.asarray(attack_scores) > thr).mean())


def metrics(y: np.ndarray, flagged: np.ndarray, scores: Optional[np.ndarray] = None, boot: bool = True) -> dict:
    """Confusion counts, rates with Wilson intervals, F1 with a bootstrap interval, and AUC if scores are given."""
    y = np.asarray(y, bool)
    f = np.asarray(flagged, bool)
    tp, fn = int((y & f).sum()), int((y & ~f).sum())
    fp, tn = int((~y & f).sum()), int((~y & ~f).sum())
    out = {
        "n": int(len(y)), "n_attack": int(y.sum()), "n_benign": int((~y).sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "fpr": round(fp / (fp + tn), 4) if fp + tn else None,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else 0.0,
    }
    if tp + fn:
        out["recall_ci"] = [round(v, 4) for v in wilson_interval(tp, tp + fn)]
    if fp + tn:
        out["fpr_ci"] = [round(v, 4) for v in wilson_interval(fp, fp + tn)]
    if tp + fp:
        out["precision_ci"] = [round(v, 4) for v in wilson_interval(tp, tp + fp)]
    if boot and len(y) > 1:
        def f1_stat(idx):
            yy, ff = y[idx], f[idx]
            t, p, nn = (yy & ff).sum(), (~yy & ff).sum(), (yy & ~ff).sum()
            return 2 * t / max(2 * t + p + nn, 1)
        out["f1_ci"] = bootstrap(f1_stat, len(y))
    if scores is not None:
        s = np.asarray(scores, float)
        out["auc"] = round(auc(s[y], s[~y]), 4)
        if boot:
            def auc_stat(idx):
                yy, ss = y[idx], s[idx]
                return auc(ss[yy], ss[~yy])
            out["auc_ci"] = bootstrap(auc_stat, len(y), iterations=300)
    return out


def per_source(items: Sequence[Item], flagged: np.ndarray) -> Dict[str, dict]:
    """Share of each source's items that were flagged, with a Wilson interval."""
    rows: Dict[str, dict] = {}
    for src in sorted({i.source for i in items}):
        idx = [k for k, i in enumerate(items) if i.source == src]
        n, fl = len(idx), int(sum(flagged[k] for k in idx))
        rows[src] = {"n": n, "flagged": fl, "rate": round(fl / n, 4), "rate_ci": [round(v, 4) for v in wilson_interval(fl, n)],
                     "kind": "attack" if items[idx[0]].is_attack else "benign"}
    return rows


def split_items(split: str = "test"):
    items = benchmark_items(split=split)
    return items, np.array([i.is_attack for i in items])
