"""
Sensitivity analyses on the held-out split.

Most of these recombine the saved per-item signal values from the ablation ladder, so they cost nothing and
use the same decisions the pipeline made. Two (the smoothing sigma and the anomaly detector's own settings)
recompute the anomaly signal and then recombine.

  threshold      trust threshold from 0.30 to 0.90
  weights        500 random weight vectors (uniform on the simplex) against the equal weights used by default
  leave one out  drop one signal, and use each signal alone
  veto           veto triggers shifted by plus or minus 0.10, and the veto switched off
  sigma          smoothing noise from 0.01 to 0.20: certified radius, abstentions, decisions
  detector       kNN neighbours and quantile, forest contamination
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from evaluation.configs import BY_NAME, applied
from evaluation.datasets import benchmark_items
from evaluation.experiment_runner import ExperimentRunner
from evaluation.ladder_eval import load_classifiers, profile_embeddings
from evaluation.v4_common import auc, metrics, read_result, write_result
from atlas.anomaly_detector import anomaly_detector
from core.embeddings import embedding_service

SIGNALS = ["semantic_relevance", "embedding_anomaly", "historical_consistency", "behavioral_consistency", "source_reliability", "temporal_trust"]
CRITICAL = {"historical_consistency": (0.40, 0.15), "behavioral_consistency": (0.40, 0.20), "embedding_anomaly": (0.30, 0.40)}
ITEMS_PATH = Path(__file__).resolve().parent / "results" / "v4" / "ladder_items.json"


def trust(sig: Dict[str, float], weights: Optional[np.ndarray] = None, threshold: float = 0.65, veto: bool = True,
          shift: float = 0.0, drop: Optional[str] = None, only: Optional[str] = None):
    """Recompute (effective trust, flagged) from one item's signal values."""
    w = np.ones(len(SIGNALS)) / len(SIGNALS) if weights is None else np.asarray(weights, float)
    w = w.copy()
    if drop:
        w[SIGNALS.index(drop)] = 0.0
    if only:
        w = np.zeros(len(SIGNALS))
        w[SIGNALS.index(only)] = 1.0
    vals = np.array([sig[k] for k in SIGNALS])
    score = float(np.clip((w * vals).sum() / max(w.sum(), 1e-9), 0, 1))
    mult, violated = 1.0, False
    if veto:
        for k, (trig, floor) in CRITICAL.items():
            if drop == k or (only and only != k):
                continue
            t = trig + shift
            if sig[k] < t:
                violated = True
                mult *= max(floor, sig[k] / 0.50)
    eff = float(np.clip(score * mult, 0, 1))
    return eff, (eff < threshold) or violated


def _arrays(rows: List[dict], **kw):
    y = np.array([r["y"] for r in rows])
    out = [trust(r["signals"], **kw) for r in rows]
    t = np.array([o[0] for o in out])
    f = np.array([o[1] for o in out])
    return y, t, f


def _row(y, t, f) -> dict:
    m = metrics(y, f, 1 - t, boot=False)
    return {k: m.get(k) for k in ("recall", "fpr", "precision", "f1", "auc")}


def run(config: str = "learned_behaviour", draws: int = 500) -> dict:
    ladder = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    rows = ladder[config]
    out: Dict[str, object] = {"config": config, "n": len(rows)}

    y, t, f = _arrays(rows)
    out["default"] = _row(y, t, f)

    out["threshold"] = {f"{th:.2f}": _row(*_arrays(rows, threshold=th)) for th in np.round(np.arange(0.30, 0.91, 0.05), 2)}

    rng = np.random.default_rng(0)
    rec, fpr, auc_v = [], [], []
    for _ in range(draws):
        w = rng.dirichlet(np.ones(len(SIGNALS)))
        r = _row(*_arrays(rows, weights=w))
        rec.append(r["recall"]); fpr.append(r["fpr"]); auc_v.append(r["auc"])
    pct = lambda v: [round(float(np.percentile(v, q)), 4) for q in (5, 50, 95)]
    out["random_weights"] = {"draws": draws, "recall_p5_p50_p95": pct(rec), "fpr_p5_p50_p95": pct(fpr), "auc_p5_p50_p95": pct(auc_v)}

    out["leave_one_signal_out"] = {k: _row(*_arrays(rows, drop=k)) for k in SIGNALS}
    out["single_signal"] = {k: _row(*_arrays(rows, only=k)) for k in SIGNALS}
    out["veto"] = {"off": _row(*_arrays(rows, veto=False)), "triggers_minus_0.10": _row(*_arrays(rows, shift=-0.10)),
                   "triggers_plus_0.10": _row(*_arrays(rows, shift=0.10))}
    return out


def run_detector_sweeps(split: str = "test", config: str = "learned_behaviour") -> dict:
    """Recompute the anomaly signal under different sigma and detector settings, then recombine."""
    ladder = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    items = sorted(benchmark_items(split=split), key=lambda i: (i.source, i.id))
    profile = profile_embeddings()
    classifiers = load_classifiers()
    out: Dict[str, dict] = {"sigma": {}, "knn": {}, "forest_contamination": {}}
    base_rows = ladder[config]

    def anomaly_for(texts: List[str], sigma: float = 0.05):
        embs = embedding_service.get_batch_embeddings(texts)
        res = [anomaly_detector.compute_certified_anomaly_score(e, sigma=sigma, seed=42) for e in embs]
        return np.array([r[0] for r in res]), np.array([r[1] for r in res])

    def recombine(rows, anomaly):
        sub = [dict(r, signals=dict(r["signals"], embedding_anomaly=float(a))) for r, a in zip(rows, anomaly)]
        return _row(*_arrays(sub))

    with applied(BY_NAME[config], profile, classifiers):
        texts = [i.text for i in items]
        for sigma in (0.01, 0.02, 0.05, 0.10, 0.20):
            a, r = anomaly_for(texts, sigma)
            row = recombine(base_rows, a)
            row.update({"mean_radius": round(float(r.mean()), 5), "abstained": round(float((r == 0).mean()), 4),
                        "mean_anomaly_signal": round(float(a.mean()), 4)})
            out["sigma"][str(sigma)] = row
        for k in (3, 5, 10, 20):
            for q in (0.90, 0.95, 0.99):
                anomaly_detector.KNN_K, anomaly_detector.KNN_QUANTILE = k, q
                anomaly_detector.set_profile(profile.tolist())
                a, _ = anomaly_for(texts)
                out["knn"][f"k={k},q={q}"] = recombine(base_rows, a)
        anomaly_detector.KNN_K, anomaly_detector.KNN_QUANTILE = 5, 0.95
    with applied(BY_NAME["profile_forest"], profile, classifiers):
        forest_rows = ladder["profile_forest"]
        for c in (0.01, 0.05, 0.10, 0.20, 0.30):
            anomaly_detector.contamination = c
            anomaly_detector.set_profile(profile.tolist())
            a, _ = anomaly_for([i.text for i in items])
            out["forest_contamination"][str(c)] = recombine(forest_rows, a)
    return out


def main():
    res = run()
    res["detector_sweeps"] = run_detector_sweeps()
    write_result("sensitivity", res)
    print(json.dumps({k: res[k] for k in ("default", "random_weights", "veto")}, indent=1))


if __name__ == "__main__":
    main()
