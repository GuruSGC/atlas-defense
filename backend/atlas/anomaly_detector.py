"""
Embedding Anomaly Detector for ATLAS.
Utilizes scikit-learn's Isolation Forest on latent embedding vectors to detect
statistical outliers and out-of-distribution adversarial memory injections.

Certified hardening: certify() applies randomized smoothing (Cohen, Rosenfeld & Kolter,
"Certified Adversarial Robustness via Randomized Smoothing," ICML 2019) on top of the base
Isolation Forest verdict, using their two-stage CERTIFY procedure. A first batch of noisy
samples (n0) selects the smoothed class, and a separate batch (n) estimates how often that
class wins. Reusing one batch for both steps biases the estimate upward, which is why the
two batches are kept apart.

It is in the same family of idea as MemSAD's (Gowda et al., 2026, arXiv:2605.03482)
certified detection radius, since both give a radius inside which a classification cannot
flip, but it is NOT a reproduction of MemSAD's gradient-coupling proof. The guarantee covers
small L2 perturbations of the embedding vector itself, not an adversary who rewrites the
underlying text into an entirely different embedding region.
"""

from dataclasses import dataclass
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


@dataclass
class Certificate:
    smoothed_normal: bool      # class chosen from the selection batch
    votes: int                 # estimation votes for that class
    n0: int
    n: int
    p_lower: float
    radius: float              # 0.0 when the procedure abstains
    abstained: bool


class AnomalyDetector:
    """
    Isolation Forest anomaly detector for vector memory embeddings.
    """

    BASELINE_SIZE = 50
    MIN_ACCEPTED_FOR_REFIT = 8
    REFIT_EVERY = 5

    def __init__(self, contamination: float = 0.10, dimension: int = 128, seed: int = 42):
        self.contamination = contamination
        self.dimension = dimension
        self.seed = seed
        self.model: Optional[IsolationForest] = None
        self._is_fitted = False
        self.fit_count = 0
        self._accepted_at_last_fit = 0
        self._baseline: Optional[np.ndarray] = None
        self._initialize_baseline_model()

    def _initialize_baseline_model(self) -> None:
        """
        Fits an initial Isolation Forest on a synthetic baseline of unit-sphere vectors.
        """
        rng = np.random.RandomState(self.seed)
        samples = rng.normal(loc=0.0, scale=0.3, size=(self.BASELINE_SIZE, self.dimension))
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        self._baseline = samples / np.maximum(norms, 1e-8)
        self._fit_forest(self._baseline)
        self.fit_count = 0
        self._accepted_at_last_fit = 0

    def _fit_forest(self, data: np.ndarray) -> None:
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.seed
        )
        self.model.fit(data)
        self._is_fitted = True

    def reset(self, seed: Optional[int] = None) -> None:
        """Return to the synthetic baseline, optionally with a different random seed."""
        if seed is not None:
            self.seed = seed
        self._initialize_baseline_model()

    def fit(self, embeddings: List[List[float]]) -> None:
        """
        Refit on verified persistent memory embeddings. The synthetic baseline stays in the
        training set so that a handful of accepted memories cannot collapse the model onto a
        tiny region of the space.
        """
        if len(embeddings) < 5:
            return
        accepted = np.array(embeddings, dtype=float)
        self._fit_forest(np.vstack([self._baseline, accepted]))
        self.fit_count += 1
        self._accepted_at_last_fit = len(embeddings)

    def maybe_refit(self, accepted_embeddings: List[List[float]]) -> bool:
        """
        Called by the memory manager after an accepted insertion. Refits once enough accepted
        memories exist and then again every few insertions. Returns True if a refit happened.
        """
        n = len(accepted_embeddings)
        if n < self.MIN_ACCEPTED_FOR_REFIT:
            return False
        if self.fit_count > 0 and n - self._accepted_at_last_fit < self.REFIT_EVERY:
            return False
        self.fit(accepted_embeddings)
        return True

    def compute_anomaly_score(self, embedding: List[float]) -> float:
        """
        Computes anomaly score between 0.0 (extreme anomaly / outlier) and 1.0 (completely normal / inlier).
        Uses IsolationForest decision function with sigmoidal normalization.
        """
        if not self._is_fitted or self.model is None or not embedding:
            return 0.5

        arr = np.array(embedding, dtype=float).reshape(1, -1)
        raw_score = self.model.decision_function(arr)[0]
        normalized_score = 1.0 / (1.0 + np.exp(-6.0 * raw_score))
        return float(np.clip(normalized_score, 0.0, 1.0))

    def _noisy_votes(self, arr: np.ndarray, count: int, sigma: float, rng: np.random.Generator) -> int:
        """Number of noisy copies the forest classifies as normal."""
        noise = rng.normal(loc=0.0, scale=sigma, size=(count, arr.shape[0]))
        noisy = arr[None, :] + noise
        norms = np.linalg.norm(noisy, axis=1, keepdims=True)
        noisy = noisy / np.maximum(norms, 1e-8)
        return int(np.sum(self.model.decision_function(noisy) > 0))

    def certify(
        self,
        embedding: List[float],
        n0: int = 50,
        n: int = 100,
        sigma: float = 0.05,
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ) -> Certificate:
        """
        Two-stage randomized-smoothing certification of the normal/anomalous decision.

        1. Draw n0 noisy copies, run the forest, and take the majority as the smoothed class.
        2. Draw n fresh copies and count how many vote for that class.
        3. Take a one-sided Clopper-Pearson lower bound p_lower on that class probability.
        4. If p_lower > 0.5 the radius is sigma * Phi^-1(p_lower); otherwise abstain (radius 0).
        """
        arr = np.array(embedding, dtype=float)
        base = self.seed if seed is None else seed
        rng_select = np.random.default_rng([base, 0])
        rng_estimate = np.random.default_rng([base, 1])

        normal_in_select = self._noisy_votes(arr, n0, sigma, rng_select)
        smoothed_normal = normal_in_select * 2 >= n0

        normal_in_estimate = self._noisy_votes(arr, n, sigma, rng_estimate)
        votes = normal_in_estimate if smoothed_normal else n - normal_in_estimate

        p_lower = _clopper_pearson_lower(votes, n, alpha)
        if p_lower > 0.5:
            return Certificate(smoothed_normal, votes, n0, n, p_lower, sigma * float(norm.ppf(p_lower)), False)
        return Certificate(smoothed_normal, votes, n0, n, p_lower, 0.0, True)

    def compute_certified_anomaly_score(
        self,
        embedding: List[float],
        n0: int = 50,
        n: int = 100,
        sigma: float = 0.05,
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Randomized-smoothing hardening of compute_anomaly_score().

        Returns (hardened_score, certified_radius):
          - hardened_score: the base score pulled toward the uncertain midpoint 0.5 in
            proportion to how small the certified radius is. A decision that barely wins the
            vote should not be trusted as much as one with a wide margin.
          - certified_radius: the radius in embedding L2 units, exposed for transparency
            (TrustSignalsBreakdown.embedding_anomaly_certified_radius).
        """
        raw_score = self.compute_anomaly_score(embedding)
        if not self._is_fitted or self.model is None or not embedding:
            return raw_score, 0.0

        cert = self.certify(embedding, n0=n0, n=n, sigma=sigma, alpha=alpha, seed=seed)

        # A radius at or beyond 2*sigma counts as a fully robust verdict (multiplier 1.0).
        reference_radius = 2.0 * sigma
        confidence_multiplier = float(np.clip(cert.radius / reference_radius, 0.0, 1.0))
        hardened_score = 0.5 + (raw_score - 0.5) * confidence_multiplier

        return float(np.clip(hardened_score, 0.0, 1.0)), float(cert.radius)


# Global anomaly detector instance
anomaly_detector = AnomalyDetector(
    contamination=settings.ISOLATION_FOREST_CONTAMINATION,
    dimension=settings.EMBEDDING_DIMENSION
)
