"""
Embedding Anomaly Detector for ATLAS.
Utilizes scikit-learn's Isolation Forest on latent embedding vectors to detect
statistical outliers and out-of-distribution adversarial memory injections.

Certified hardening (September 2026): compute_certified_anomaly_score() applies
randomized smoothing (Cohen, Rosenfeld & Kolter, "Certified Adversarial Robustness
via Randomized Smoothing," ICML 2019) on top of the base Isolation Forest verdict.
This is a deliberate, honestly-scoped choice: it is in the same family of idea as
MemSAD's (Gowda et al., 2026, arXiv:2605.03482) certified detection radius.
Both produce a provable radius within which a classification cannot flip, but it
is NOT a reproduction of MemSAD's own gradient-coupling proof, which rests on that
paper's specific encoder-regularity assumptions and derivation that are not
reproduced here. Randomized smoothing is a separately well-established, correctly
implementable technique that gives a real (if narrower) certified guarantee:
robustness to small L2 perturbations of the embedding vector itself, not to an
adversary who rewrites the underlying text into an entirely different embedding
region. That distinction is stated plainly rather than glossed over.
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy.stats import norm, beta
from sklearn.ensemble import IsolationForest
from core.settings import settings


def _clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    """
    Exact one-sided Clopper-Pearson lower confidence bound on a binomial proportion.
    Standard statistical technique (Clopper & Pearson, 1934), not implementation-specific.
    """
    if successes <= 0:
        return 0.0
    if successes >= trials:
        return float(alpha ** (1.0 / trials))
    return float(beta.ppf(alpha, successes, trials - successes + 1))


class AnomalyDetector:
    """
    Isolation Forest anomaly detector for vector memory embeddings.
    """

    def __init__(self, contamination: float = 0.10, dimension: int = 128):
        self.contamination = contamination
        self.dimension = dimension
        self.model: Optional[IsolationForest] = None
        self._is_fitted = False
        self._initialize_baseline_model()

    def _initialize_baseline_model(self) -> None:
        """
        Initializes and fits an initial Isolation Forest on standard baseline embeddings.
        """
        # Create baseline distribution of benign unit sphere vectors
        np.random.seed(42)
        baseline_samples = np.random.normal(loc=0.0, scale=0.3, size=(50, self.dimension))
        # Normalize to unit sphere
        norms = np.linalg.norm(baseline_samples, axis=1, keepdims=True)
        baseline_samples = baseline_samples / np.maximum(norms, 1e-8)

        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=42
        )
        self.model.fit(baseline_samples)
        self._is_fitted = True

    def fit(self, embeddings: List[List[float]]) -> None:
        """
        Fit or re-train Isolation Forest on verified persistent memory embeddings.
        """
        if len(embeddings) < 5:
            # Not enough samples to reliably fit, keep baseline
            return

        arr = np.array(embeddings, dtype=float)
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=42
        )
        self.model.fit(arr)
        self._is_fitted = True

    def compute_anomaly_score(self, embedding: List[float]) -> float:
        """
        Computes anomaly score between 0.0 (extreme anomaly / outlier) and 1.0 (completely normal / inlier).
        Uses IsolationForest decision function with sigmoidal normalization.
        """
        if not self._is_fitted or self.model is None or not embedding:
            return 0.5

        arr = np.array(embedding, dtype=float).reshape(1, -1)
        # raw decision_function: typically in [-0.5, 0.5], positive is inlier, negative is outlier
        raw_score = self.model.decision_function(arr)[0]

        # Sigmoidal mapping to [0.0, 1.0]
        # Shifted so that 0.0 maps to ~0.5, positive values approach 1.0, negative approach 0.0
        normalized_score = 1.0 / (1.0 + np.exp(-6.0 * raw_score))
        return float(np.clip(normalized_score, 0.0, 1.0))

    def compute_certified_anomaly_score(
        self,
        embedding: List[float],
        n_samples: int = 100,
        sigma: float = 0.05,
        alpha: float = 0.05,
    ) -> Tuple[float, float]:
        """
        Randomized-smoothing certified hardening of compute_anomaly_score(). See module
        docstring for exactly what this does and does not guarantee.

        Procedure (Cohen et al., 2019):
          1. Draw n_samples noisy copies of the embedding, each perturbed by isotropic
             Gaussian noise of scale sigma, then re-normalized to the unit sphere (the
             embedding space this system operates in throughout).
          2. Run the base Isolation Forest's decision_function on all noisy copies and
             tally how many vote "normal" (positive decision) vs "anomalous".
          3. Take the majority vote as the smoothed classification, and compute a
             Clopper-Pearson lower confidence bound (95% by default) on the true
             probability that vote would recur.
          4. If that lower bound exceeds 0.5, derive a certified radius R = sigma * Phi^-1(p_lower):
             the guaranteed minimum L2 distance the embedding would need to move before
             the smoothed classification could possibly flip, regardless of which specific
             direction or attack strategy produced that movement.

        Returns (hardened_score, certified_radius):
          - hardened_score: the base compute_anomaly_score() value, pulled toward the
            uncertain midpoint (0.5) in proportion to how small/unreliable the certified
            radius is. A fragile, barely-a-majority smoothed vote should not be trusted as
            much as a robust, wide-margin one.
          - certified_radius: the raw radius in embedding L2-distance units, exposed for
            transparency (TrustSignalsBreakdown.embedding_anomaly_certified_radius) rather
            than folded invisibly into a single number.
        """
        raw_score = self.compute_anomaly_score(embedding)

        if not self._is_fitted or self.model is None or not embedding:
            return raw_score, 0.0

        arr = np.array(embedding, dtype=float)
        rng = np.random.RandomState(42)  # deterministic, reproducible per project convention
        noise = rng.normal(loc=0.0, scale=sigma, size=(n_samples, arr.shape[0]))
        noisy = arr[None, :] + noise
        norms = np.linalg.norm(noisy, axis=1, keepdims=True)
        noisy = noisy / np.maximum(norms, 1e-8)

        decisions = self.model.decision_function(noisy)
        votes_normal = int(np.sum(decisions > 0))
        majority_is_normal = votes_normal * 2 >= n_samples
        k = votes_normal if majority_is_normal else (n_samples - votes_normal)

        p_lower = _clopper_pearson_lower(k, n_samples, alpha)

        if p_lower > 0.5:
            certified_radius = sigma * float(norm.ppf(p_lower))
        else:
            certified_radius = 0.0

        # Confidence-based hardening: a certified radius at or beyond 2*sigma is treated as
        # a fully robust verdict (multiplier 1.0); anything smaller pulls the score toward
        # the uncertain midpoint proportionally, rather than reporting false confidence.
        reference_radius = 2.0 * sigma
        confidence_multiplier = float(np.clip(certified_radius / reference_radius, 0.0, 1.0))
        hardened_score = 0.5 + (raw_score - 0.5) * confidence_multiplier

        return float(np.clip(hardened_score, 0.0, 1.0)), float(certified_radius)


# Global anomaly detector instance
anomaly_detector = AnomalyDetector(
    contamination=settings.ISOLATION_FOREST_CONTAMINATION,
    dimension=settings.EMBEDDING_DIMENSION
)
