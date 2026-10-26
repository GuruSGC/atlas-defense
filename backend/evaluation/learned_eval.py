"""
Training and evaluation of the learned poisoning classifier.

  python -m evaluation.learned_eval train [--kind finetuned|logreg] [--out models/poison_clf]
  python -m evaluation.learned_eval loso  [--kind finetuned|logreg]

Training uses the train split only and leaves PoisonedRAG out: those items are false facts written in
an encyclopedic style, and a text classifier would learn the style, not the falsehood.

The leave-one-source-out run is the generalisation test. For each attack source (the authored sets count
as one source) the classifier is trained without it and scored on all of it, against the benign test items.
One more run trains without OpenAssistant, to see how it behaves on a benign style it has never seen.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from atlas.learned import PoisonClassifier
from core.settings import BASE_DIR
from evaluation.datasets import benchmark_items
from evaluation.stats import auc_from_scores, wilson_interval

RESULTS = Path(__file__).resolve().parent / "results"
EXCLUDED_FROM_TRAINING = {"poisonedrag"}


def _group(source: str) -> str:
    return "authored" if source.startswith("authored:") and "benign" not in source else source


def _split_items():
    items = benchmark_items()
    train = [i for i in items if i.split == "train" and i.source not in EXCLUDED_FROM_TRAINING]
    test = [i for i in items if i.split == "test"]
    return items, train, test


def tpr_at_fpr(attack_scores: np.ndarray, benign_scores: np.ndarray, fpr: float = 0.01) -> float:
    neg = np.sort(benign_scores)
    thr = neg[int(np.ceil((1 - fpr) * len(neg))) - 1]
    return float((attack_scores > thr).mean())


def train(kind: str, out: str = None, epochs: int = 4, seed: int = 0) -> PoisonClassifier:
    _, train_items, _ = _split_items()
    clf = PoisonClassifier(kind).fit([i.text for i in train_items], [i.is_attack for i in train_items], epochs=epochs, seed=seed)
    if out:
        clf.save(str(Path(out) if Path(out).is_absolute() else BASE_DIR / out))
    return clf


def score_summary(clf: PoisonClassifier, attacks, benign) -> Dict:
    a = clf.margin([i.text for i in attacks]) if attacks else np.zeros(0)
    b = clf.margin([i.text for i in benign]) if benign else np.zeros(0)
    pa = clf.probability([i.text for i in attacks]) if attacks else np.zeros(0)
    pb = clf.probability([i.text for i in benign]) if benign else np.zeros(0)
    out = {"n_attack": len(attacks), "n_benign": len(benign)}
    if len(a) and len(b):
        out["auc"] = round(auc_from_scores(list(a), list(b)), 4)
        out["tpr_at_1pct_fpr"] = round(tpr_at_fpr(a, b, 0.01), 4)
        out["tpr_at_5pct_fpr"] = round(tpr_at_fpr(a, b, 0.05), 4)
    if len(a):
        out["recall_p50"] = round(float((pa > 0.5).mean()), 4)
    if len(b):
        fp = int((pb > 0.5).sum())
        out["fpr_p50"] = round(fp / len(b), 4)
        out["fpr_p50_ci"] = [round(x, 4) for x in wilson_interval(fp, len(b))]
    return out


def evaluate_test(clf: PoisonClassifier) -> Dict:
    """Test-split results overall and per source."""
    _, _, test = _split_items()
    attacks = [i for i in test if i.is_attack and i.source != "poisonedrag"]
    benign = [i for i in test if not i.is_attack]
    res = {"overall": score_summary(clf, attacks, benign), "per_source": {}}
    for src in sorted({i.source for i in attacks}):
        res["per_source"][src] = score_summary(clf, [i for i in attacks if i.source == src], benign)
    pr = [i for i in test if i.source == "poisonedrag"]
    res["poisonedrag_not_trained_on"] = score_summary(clf, pr, benign)
    for src in sorted({i.source for i in benign}):
        bs = [i for i in benign if i.source == src]
        res["per_source"][src] = {"n_benign": len(bs), "fpr_p50": round(float((clf.probability([i.text for i in bs]) > 0.5).mean()), 4)}
    return res


def loso(kind: str, epochs: int = 4) -> Dict:
    items, train_items, test_items = _split_items()
    benign_test = [i for i in test_items if not i.is_attack]
    groups = sorted({_group(i.source) for i in items if i.is_attack and i.source not in EXCLUDED_FROM_TRAINING})
    out: Dict[str, Dict] = {}
    for g in groups:
        t0 = time.time()
        tr = [i for i in train_items if _group(i.source) != g]
        clf = PoisonClassifier(kind).fit([i.text for i in tr], [i.is_attack for i in tr], epochs=epochs, seed=0)
        held = [i for i in items if i.is_attack and _group(i.source) == g]
        out[g] = score_summary(clf, held, benign_test)
        out[g]["seconds"] = round(time.time() - t0)
        print(f"hold {g:24s} {out[g]}", flush=True)
    t0 = time.time()
    tr = [i for i in train_items if i.source != "oasst1"]
    clf = PoisonClassifier(kind).fit([i.text for i in tr], [i.is_attack for i in tr], epochs=epochs, seed=0)
    unseen_benign = [i for i in items if i.source == "oasst1"]
    attacks_test = [i for i in test_items if i.is_attack and i.source not in EXCLUDED_FROM_TRAINING]
    out["benign:oasst1"] = score_summary(clf, attacks_test, unseen_benign)
    out["benign:oasst1"]["seconds"] = round(time.time() - t0)
    print("hold benign oasst1", out["benign:oasst1"], flush=True)
    return out


def style_control() -> Dict:
    """
    Shortcut check. A logistic regression that sees only surface style (length, punctuation, capitalisation,
    digits, quotes) and no meaning. If it separates attacks from benign text well, part of what any text
    classifier learns on this benchmark is style and not intent.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def feats(t: str):
        return [len(t), len(t.split()), t.strip()[-1:] in ".!?", sum(c.isupper() for c in t) / max(len(t), 1),
                sum(c.isdigit() for c in t), t.count(","), t.count('"'), t[:1].isupper()]

    _, train_items, test_items = _split_items()
    scaler = StandardScaler().fit([feats(i.text) for i in train_items])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        scaler.transform([feats(i.text) for i in train_items]), [i.is_attack for i in train_items])
    test = [i for i in test_items if i.source not in EXCLUDED_FROM_TRAINING]
    margin = clf.decision_function(scaler.transform([feats(i.text) for i in test]))
    y = np.array([i.is_attack for i in test])
    from evaluation.v4_common import auc as auc_fn
    out = {"auc": round(auc_fn(margin[y], margin[~y]), 4), "tpr_at_1pct_fpr": round(tpr_at_fpr(margin[y], margin[~y], 0.01), 4),
           "tpr_at_5pct_fpr": round(tpr_at_fpr(margin[y], margin[~y], 0.05), 4), "n_attack": int(y.sum()), "n_benign": int((~y).sum())}
    # the same check restricted to texts of 8 to 30 words, where length cannot do the work
    keep = np.array([8 <= len(i.text.split()) <= 30 for i in test])
    if keep.sum() and y[keep].any() and (~y[keep]).any():
        out["length_matched"] = {"n": int(keep.sum()), "auc": round(auc_fn(margin[keep & y], margin[keep & ~y]), 4)}
    return out


def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "loso", "test", "style"])
    ap.add_argument("--kind", default="finetuned")
    ap.add_argument("--out", default="models/poison_clf")
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        t0 = time.time()
        clf = train(args.kind, args.out, args.epochs)
        res = {"meta": clf.meta, "seconds": round(time.time() - t0), "test": evaluate_test(clf)}
        (RESULTS / f"classifier_{args.kind}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res["test"]["overall"], indent=1))
    elif args.command == "style":
        res = style_control()
        (RESULTS / "style_control.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res))
    elif args.command == "loso":
        res = loso(args.kind, args.epochs)
        (RESULTS / f"classifier_loso_{args.kind}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
