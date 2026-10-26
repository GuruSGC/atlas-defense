"""
Anomaly detectors on the held-out split.

Every detector is fitted on benign embeddings only and has no access to any attack. Scores are higher for
more anomalous text. The synthetic forest is the detector the original prototype used (50 random
unit vectors); the others are fitted on the benign train split.
"""

import warnings
from typing import Dict

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from evaluation.datasets import benchmark_items
from evaluation.v4_common import auc, local_embeddings, tpr_at_fpr, write_result, bootstrap


def synthetic_forest(dim: int, seed: int = 42) -> IsolationForest:
    rng = np.random.RandomState(seed)
    syn = rng.normal(0, 0.3, size=(50, dim))
    syn /= np.linalg.norm(syn, axis=1, keepdims=True)
    return IsolationForest(n_estimators=100, contamination=0.1, random_state=seed).fit(syn)


def fit_detectors(benign_train: np.ndarray, seed: int = 42) -> Dict[str, callable]:
    """Name -> scoring function (embeddings -> anomaly score)."""
    dim = benign_train.shape[1]
    syn = synthetic_forest(dim, seed)
    forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=seed).fit(benign_train)
    nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(benign_train)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="cosine").fit(benign_train)
    ocsvm = OneClassSVM(nu=0.1, gamma="scale").fit(benign_train)
    mu = benign_train.mean(0)
    mu /= np.linalg.norm(mu)
    return {
        "isolation forest, synthetic baseline (original)": lambda X: -syn.decision_function(X),
        "isolation forest, benign profile": lambda X: -forest.decision_function(X),
        "k-nearest-neighbour distance (k=5)": lambda X: nn.kneighbors(X)[0].mean(axis=1),
        "local outlier factor (k=20)": lambda X: -lof.decision_function(X),
        "one-class SVM (RBF)": lambda X: -ocsvm.decision_function(X),
        "cosine distance to profile centroid": lambda X: 1 - X @ mu,
    }


def run() -> dict:
    warnings.filterwarnings("ignore")
    items = benchmark_items()
    X = local_embeddings([i.text for i in items])
    y = np.array([i.is_attack for i in items])
    split = np.array([i.split for i in items])
    src = np.array([i.source for i in items])
    benign_train = X[(split == "train") & ~y]
    test = split == "test"
    out = {"n_profile": int(len(benign_train)), "n_test_attack": int((test & y).sum()), "n_test_benign": int((test & ~y).sum()),
           "detectors": {}, "profile_size": {}}
    for name, fn in fit_detectors(benign_train).items():
        s = np.asarray(fn(X))
        a, b = s[test & y], s[test & ~y]
        row = {"auc": round(auc(a, b), 4), "tpr_at_1pct_fpr": round(tpr_at_fpr(a, b, 0.01), 4),
               "tpr_at_5pct_fpr": round(tpr_at_fpr(a, b, 0.05), 4), "per_source_auc": {}}
        ya, sa = y[test], s[test]
        row["auc_ci"] = bootstrap(lambda idx: auc(sa[idx][ya[idx]], sa[idx][~ya[idx]]), len(ya), iterations=300)
        for sname in sorted(set(src[test & y])):
            row["per_source_auc"][sname] = round(auc(s[test & (src == sname)], b), 4)
        out["detectors"][name] = row
    # how much real benign data does a profile need?
    rng = np.random.RandomState(0)
    for n in (25, 50, 100, 200, len(benign_train)):
        n = min(n, len(benign_train))
        sub = benign_train[rng.choice(len(benign_train), n, replace=False)]
        det = fit_detectors(sub)
        out["profile_size"][str(n)] = {k: round(auc(np.asarray(det[k](X))[test & y], np.asarray(det[k](X))[test & ~y]), 4)
                                         for k in ("isolation forest, benign profile", "k-nearest-neighbour distance (k=5)")}
    write_result("anomaly", out)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1)[:3000])
