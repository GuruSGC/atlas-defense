"""
Learned memory-poisoning classifier.

The rule-based behavioural detectors only know the wordings someone thought of. This module adds a
supervised text classifier that is trained on labelled candidate memories and then used inside
behavioural consistency (see atlas/signals.py, BEHAV_MODE).

Two variants share one interface:
  logreg      logistic regression on frozen sentence embeddings. Seconds to train, tiny file.
  finetuned   the same sentence-transformer encoder fine-tuned end to end with a classification head.
              Minutes to train on a CPU, and clearly stronger.

The output is a calibrated probability that a candidate is an attack. Calibration is Platt scaling
on examples the classifier did not train on.
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


class PoisonClassifier:
    def __init__(self, kind: str = "finetuned"):
        if kind not in ("logreg", "finetuned"):
            raise ValueError(kind)
        self.kind = kind
        self._lr = None
        self._tok = None
        self._model = None
        self._platt = (1.0, 0.0)      # p = sigmoid(a * margin + b)
        self.meta: dict = {}

    # ------------------------------------------------------------------ scoring
    def margin(self, texts: Sequence[str]) -> np.ndarray:
        """Decision margin, positive for attack-like text."""
        texts = [_normalize(t) for t in texts]
        if not texts:
            return np.zeros(0)
        if self.kind == "logreg":
            from core.embeddings import embedding_service
            vecs = np.array(embedding_service._local.encode(texts), dtype=float)
            return self._lr.decision_function(vecs)
        import torch
        out: List[float] = []
        self._model.eval()
        with torch.no_grad():
            for b in range(0, len(texts), 64):
                enc = self._tok(texts[b:b + 64], padding=True, truncation=True, max_length=96, return_tensors="pt")
                lg = self._model(**enc).logits
                out.extend((lg[:, 1] - lg[:, 0]).tolist())
        return np.array(out)

    def probability(self, texts: Sequence[str]) -> np.ndarray:
        a, b = self._platt
        z = np.clip(a * self.margin(texts) + b, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba_one(self, text: str) -> float:
        return float(self.probability([text])[0])

    # ------------------------------------------------------------------ training
    def fit(self, texts: Sequence[str], labels: Sequence[bool], calibration_fraction: float = 0.15,
            epochs: int = 4, seed: int = 0, learning_rate: float = 5e-5, C: float = 1.0) -> "PoisonClassifier":
        texts = [_normalize(t) for t in texts]
        y = np.array(labels, dtype=bool)
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(texts))
        n_cal = int(len(texts) * calibration_fraction)
        cal_idx, fit_idx = order[:n_cal], order[n_cal:]
        if self.kind == "logreg":
            self._fit_logreg([texts[i] for i in fit_idx], y[fit_idx], C)
        else:
            self._fit_finetuned([texts[i] for i in fit_idx], y[fit_idx], epochs, seed, learning_rate)
        self._calibrate([texts[i] for i in cal_idx], y[cal_idx])
        self.meta = {"kind": self.kind, "n_train": int(len(fit_idx)), "n_calibration": int(n_cal), "epochs": epochs,
                     "seed": seed, "platt": list(self._platt)}
        return self

    def _fit_logreg(self, texts, y, C):
        from sklearn.linear_model import LogisticRegression
        from core.embeddings import embedding_service
        vecs = np.array(embedding_service._local.encode(texts), dtype=float)
        self._lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(vecs, y)

    def _fit_finetuned(self, texts, y, epochs, seed, lr):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        self._tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        opt = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=0.01)
        weight = torch.tensor([1.0, float((~y).sum()) / max(float(y.sum()), 1.0)], dtype=torch.float)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        idx = np.arange(len(texts))
        for _ in range(epochs):
            self._model.train()
            rng.shuffle(idx)
            for b in range(0, len(idx), 16):
                bi = idx[b:b + 16]
                enc = self._tok([texts[i] for i in bi], padding=True, truncation=True, max_length=96, return_tensors="pt")
                loss = loss_fn(self._model(**enc).logits, torch.tensor(y[bi].astype(int)))
                loss.backward()
                opt.step()
                opt.zero_grad()
        self._model.eval()

    def _calibrate(self, texts, y):
        from sklearn.linear_model import LogisticRegression
        if len(texts) < 20 or y.all() or (~y).all():
            return
        m = self.margin(texts).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(m, y)
        self._platt = (float(lr.coef_[0][0]), float(lr.intercept_[0]))

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self.kind == "logreg":
            import joblib
            joblib.dump(self._lr, d / "logreg.joblib")
        else:
            self._model.save_pretrained(d / "encoder")
            self._tok.save_pretrained(d / "encoder")
        (d / "meta.json").write_text(json.dumps(self.meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: str) -> "PoisonClassifier":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        clf = cls(meta["kind"])
        if clf.kind == "logreg":
            import joblib
            clf._lr = joblib.load(d / "logreg.joblib")
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            clf._tok = AutoTokenizer.from_pretrained(d / "encoder")
            clf._model = AutoModelForSequenceClassification.from_pretrained(d / "encoder")
            clf._model.eval()
        clf._platt = tuple(meta["platt"])
        clf.meta = meta
        return clf


_cached: Optional[PoisonClassifier] = None
_cached_path: Optional[str] = None
_forced = False


def get_poison_classifier(path: Optional[str] = None) -> Optional[PoisonClassifier]:
    """The classifier at settings.POISON_MODEL_PATH, or None when nothing has been trained yet."""
    global _cached, _cached_path
    from core.settings import settings, BASE_DIR
    if _forced:
        return _cached
    path = path or settings.POISON_MODEL_PATH
    full = Path(path) if Path(path).is_absolute() else BASE_DIR / path
    if _cached is not None and _cached_path == str(full):
        return _cached
    if not (full / "meta.json").exists():
        return None
    try:
        _cached, _cached_path = PoisonClassifier.load(str(full)), str(full)
    except Exception as exc:        # missing torch or a corrupt file: run on the rules alone
        print(f"[ATLAS] learned classifier not loaded ({str(exc)[:120]}); using rule-based behavioural checks only.")
        return None
    return _cached


def set_poison_classifier(clf: Optional[PoisonClassifier]) -> None:
    """Install a classifier for the current process (used by experiments)."""
    global _cached, _cached_path, _forced
    _cached, _cached_path, _forced = clf, "<installed>", True


def clear_installed_classifier() -> None:
    global _cached, _cached_path, _forced
    _cached, _cached_path, _forced = None, None, False
