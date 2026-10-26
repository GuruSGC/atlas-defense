"""
Optional natural-language-inference check for historical consistency.

The rule-based contradiction checks only know the topics listed in DOMAIN_CLUSTERS. An NLI
cross-encoder judges whether a stored memory (premise) and a candidate (hypothesis)
contradict each other whatever the topic is. It needs the optional sentence-transformers
package and a model download, so it is off by default (USE_NLI=false). When it cannot be
loaded the checker reports itself unavailable and the signal falls back to the rules.
"""

from typing import Callable, List, Optional
import numpy as np
from core.settings import settings


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


class NLIContradictionChecker:
    def __init__(self, model_name: Optional[str] = None, loader: Optional[Callable[[str], object]] = None):
        self.model_name = model_name or settings.NLI_MODEL
        self._loader = loader
        self._model = None
        self._contradiction_index = 0
        self.available: Optional[bool] = None
        self.error: str = ""

    def _load(self) -> bool:
        if self.available is not None:
            return self.available
        try:
            if self._loader is not None:
                self._model = self._loader(self.model_name)
            else:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            labels = getattr(getattr(getattr(self._model, "model", None), "config", None), "id2label", None) or {}
            for idx, name in labels.items():
                if str(name).lower().startswith("contradiction"):
                    self._contradiction_index = int(idx)
            self.available = True
        except Exception as exc:
            self.available = False
            self.error = str(exc)[:200]
        return self.available

    def contradiction_probability(self, premise: str, hypothesis: str) -> Optional[float]:
        """Probability that the hypothesis contradicts the premise, or None if no model is loaded."""
        if not self._load():
            return None
        raw = np.asarray(self._model.predict([(premise, hypothesis)]), dtype=float).reshape(1, -1)
        if not (np.all(raw >= 0.0) and np.allclose(raw.sum(axis=1), 1.0, atol=1e-3)):
            raw = _softmax(raw)
        return float(raw[0, self._contradiction_index])

    def contradiction_probabilities(self, pairs: List[tuple]) -> List[Optional[float]]:
        return [self.contradiction_probability(p, h) for p, h in pairs]


def build_nli_checker() -> Optional[NLIContradictionChecker]:
    """A checker when USE_NLI is on, otherwise None so the signal uses rules only."""
    return NLIContradictionChecker() if settings.USE_NLI else None
