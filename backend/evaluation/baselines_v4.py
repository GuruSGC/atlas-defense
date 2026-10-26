"""
Head-to-head comparison on the held-out split.

Every row is scored on the same items. Unsupervised detectors are fitted on benign train data and
use a threshold set on a held-out slice of that benign data so that about 5% of it would be flagged.
Nothing here sees a test item before it is scored.

  no defense                      accepts everything
  static pattern moderation       keyword and code-pattern filter (the idea in Sunil et al.)
  provenance gate                 accepts signed writes only (the idea in SMSR); two attacker models
  embedding similarity threshold  reject text far from every benign memory
  isolation forest (original)     forest on 50 synthetic vectors
  isolation forest (profile)      forest on benign embeddings
  k-nearest-neighbour             mean distance to the five nearest benign memories
  local outlier factor            LOF novelty detection
  one-class SVM                   RBF one-class SVM
  zero-shot NLI                   an NLI model asked whether the text instructs an assistant
  classifier only                 the fine-tuned and the logistic-regression classifiers on their own
  ATLAS ladder                    from ladder_eval
"""

import warnings
from typing import Dict, List

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from atlas.nli import NLIContradictionChecker
from evaluation.anomaly_eval import synthetic_forest
from evaluation.baselines import StaticPatternModeration
from evaluation.datasets import benchmark_items
from evaluation.ladder_eval import load_classifiers
from evaluation.v4_common import local_embeddings, metrics, per_source, read_result, write_result

HYPOTHESIS = "This text gives instructions to an AI assistant."


def _threshold(scores: np.ndarray, q: float = 0.95) -> float:
    return float(np.quantile(scores, q))


def run() -> dict:
    warnings.filterwarnings("ignore")
    all_items = benchmark_items()
    X_all = local_embeddings([i.text for i in all_items])
    idx = {i.id: k for k, i in enumerate(all_items)}
    y_all = np.array([i.is_attack for i in all_items])
    split = np.array([i.split for i in all_items])
    test_idx = np.where(split == "test")[0]
    items = [all_items[k] for k in test_idx]
    y = y_all[test_idx]
    Xt = X_all[test_idx]

    benign_train = np.where((split == "train") & ~y_all)[0]
    rng = np.random.RandomState(0)
    perm = rng.permutation(benign_train)
    n_val = max(40, len(perm) // 5)
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    Xfit, Xval = X_all[fit_idx], X_all[val_idx]

    rows: Dict[str, dict] = {}

    def add(name: str, scores: np.ndarray, flagged: np.ndarray, note: str = ""):
        row = metrics(y, flagged, scores)
        row["per_source"] = per_source(items, flagged)
        row["note"] = note
        rows[name] = row
        print(f"{name:42s} recall={row['recall']} fpr={row['fpr']} f1={row['f1']} auc={row.get('auc')}", flush=True)

    # ---- no defense, static pattern, provenance
    add("no defense", np.zeros(len(items)), np.zeros(len(items), bool))
    sp = StaticPatternModeration()
    flagged = np.array([sp.blocks(i.text, i.context, True) for i in items])
    add("static pattern moderation", flagged.astype(float), flagged, "keyword and code-pattern filter")
    unsigned = y.copy()
    add("provenance gate, unsigned attacker", unsigned.astype(float), unsigned, "blocks every write without a valid signature")
    add("provenance gate, signed attacker", np.zeros(len(items)), np.zeros(len(items), bool), "the attacker uses the user's signed channel")

    # ---- unsupervised detectors on semantic embeddings
    syn = synthetic_forest(Xfit.shape[1])
    forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42).fit(Xfit)
    nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(Xfit)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="cosine").fit(Xfit)
    ocsvm = OneClassSVM(nu=0.1, gamma="scale").fit(Xfit)
    mu = Xfit.mean(0)
    mu /= np.linalg.norm(mu)
    detectors = {
        "embedding similarity threshold": lambda X: 1 - (X @ Xfit.T).max(axis=1),
        "isolation forest, synthetic baseline (original)": lambda X: -syn.decision_function(X),
        "isolation forest, benign profile": lambda X: -forest.decision_function(X),
        "k-nearest-neighbour distance": lambda X: nn.kneighbors(X)[0].mean(axis=1),
        "local outlier factor": lambda X: -lof.decision_function(X),
        "one-class SVM": lambda X: -ocsvm.decision_function(X),
    }
    for name, fn in detectors.items():
        thr = _threshold(np.asarray(fn(Xval)))
        s = np.asarray(fn(Xt))
        add(name, s, s > thr, "threshold at the 95th percentile of held-out benign scores")

    # ---- zero-shot NLI: does the text instruct an assistant?
    checker = NLIContradictionChecker()
    if checker._load():
        def entail(texts: List[str]) -> np.ndarray:
            raw = np.asarray(checker._model.predict([(t, HYPOTHESIS) for t in texts], batch_size=64), dtype=float)
            e = np.exp(raw - raw.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            labels = checker._model.model.config.id2label
            ent = [int(k) for k, v in labels.items() if str(v).lower().startswith("entail")][0]
            return p[:, ent]
        val_texts = [all_items[k].text for k in val_idx]
        thr = _threshold(entail(val_texts))
        s = entail([i.text for i in items])
        add("zero-shot NLI instruction detector", s, s > thr, "no training; hypothesis: the text instructs an assistant")

    # ---- classifiers on their own
    clfs = load_classifiers()
    for key, label in (("logreg", "classifier only (logistic regression)"), ("finetuned", "classifier only (fine-tuned)")):
        if key in clfs:
            p = clfs[key].probability([i.text for i in items])
            add(label, p, p > 0.5, "trained on the train split, probability above 0.5")

    # ---- the ATLAS ladder
    ladder = read_result("ladder")
    ladder_rows = ladder["configs"] if ladder else {}
    out = {"n_test": len(items), "n_attack": int(y.sum()), "n_benign": int((~y).sum()), "rows": rows, "atlas": ladder_rows}
    write_result("baselines", out)
    return out


if __name__ == "__main__":
    run()
