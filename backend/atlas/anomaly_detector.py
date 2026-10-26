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
from sklearn.neighbors import NearestNeighbors
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

    KNN_K = 5
    KNN_QUANTILE = 0.95

    def __init__(self, contamination: float = 0.10, dimension: int = 128, seed: int = 42, kind: Optional[str] = None):
        self.kind = kind or settings.ANOMALY_DETECTOR      # "iforest" or "knn"
        if self.kind not in ("iforest", "knn"):
            raise ValueError(f"unknown anomaly detector {self.kind!r}")
        self._knn: Optional[NearestNeighbors] = None
        self._knn_tau = 0.0
        self._knn_scale = 1.0
        self.contamination = contamination
        self.dimension = dimension
        self.seed = seed
        self.model: Optional[IsolationForest] = None
        self._is_fitted = False
        self.fit_count = 0
        self._accepted_at_last_fit = 0
        self._baseline: Optional[np.ndarray] = None
        self._profile: Optional[np.ndarray] = None
        self._initialize_baseline_model()

    def _initialize_baseline_model(self) -> None:
        """
        Fits the starting Isolation Forest. With a profile set (real benign embeddings) the forest
        starts from that profile; otherwise it starts from a synthetic baseline of unit-sphere vectors.
        """
        if self._profile is not None:
            self._baseline = self._profile
            self._fit_forest(self._baseline)
            self.fit_count = 0
            self._accepted_at_last_fit = 0
            return
        rng = np.random.RandomState(self.seed)
        samples = rng.normal(loc=0.0, scale=0.3, size=(self.BASELINE_SIZE, self.dimension))
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        self._baseline = samples / np.maximum(norms, 1e-8)
        self._fit_forest(self._baseline)
        self.fit_count = 0
        self._accepted_at_last_fit = 0

    def _fit_forest(self, data: np.ndarray) -> None:
        """Fit the base detector on `data`: an isolation forest, or a k-nearest-neighbour distance model."""
        if self.kind == "knn":
            data = data / np.maximum(np.linalg.norm(data, axis=1, keepdims=True), 1e-8)
            k = min(self.KNN_K, len(data) - 1)
            self._knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(data)
            dist, _ = self._knn.kneighbors(data)
            loo = dist[:, 1:].mean(axis=1)                       # distance to the k nearest others
            self._knn_tau = float(np.quantile(loo, self.KNN_QUANTILE))
            self._knn_scale = float(max(np.std(loo), 1e-3))
            self._knn_k = k
            self.model = None
            self._is_fitted = True
            return
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.seed
        )
        self.model.fit(data)
        self._is_fitted = True

    def _margins(self, batch: np.ndarray) -> np.ndarray:
        """Signed margin per row: positive means inside the learned distribution."""
        if self.kind == "knn":
            batch = batch / np.maximum(np.linalg.norm(batch, axis=1, keepdims=True), 1e-8)
            dist, _ = self._knn.kneighbors(batch, n_neighbors=self._knn_k)
            return self._knn_tau - dist.mean(axis=1)
        return self.model.decision_function(batch)

    def _sigmoid_gain(self) -> float:
        return 2.0 / self._knn_scale if self.kind == "knn" else 6.0

    def reset(self, seed: Optional[int] = None) -> None:
        """Return to the starting baseline (profile if one is set), optionally with a different random seed."""
        if seed is not None:
            self.seed = seed
        self._initialize_baseline_model()

    def set_dimension(self, dimension: int) -> None:
        """Follow the embedding backend. Drops any profile of a different dimension."""
        if dimension != self.dimension:
            self.dimension = dimension
            if self._profile is not None and self._profile.shape[1] != dimension:
                self._profile = None
            self._initialize_baseline_model()

    def set_profile(self, embeddings: List[List[float]]) -> None:
        """
        Use real benign embeddings as the starting distribution instead of synthetic vectors.
        The profile is what "normal" means for this user or deployment.
        """
        arr = np.array(embeddings, dtype=float)
        if arr.ndim != 2 or len(arr) < 20:
            raise ValueError("a profile needs at least 20 embeddings")
        self.dimension = arr.shape[1]
        self._profile = arr
        self._initialize_baseline_model()

    def clear_profile(self) -> None:
        self._profile = None
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
        if not self._is_fitted or not embedding:
            return 0.5

        arr = np.array(embedding, dtype=float).reshape(1, -1)
        raw_score = self._margins(arr)[0]
        normalized_score = 1.0 / (1.0 + np.exp(-np.clip(self._sigmoid_gain() * raw_score, -50.0, 50.0)))
        return float(np.clip(normalized_score, 0.0, 1.0))

    def _noisy_votes(self, arr: np.ndarray, count: int, sigma: float, rng: np.random.Generator) -> int:
        """Number of noisy copies the forest classifies as normal."""
        noise = rng.normal(loc=0.0, scale=sigma, size=(count, arr.shape[0]))
        noisy = arr[None, :] + noise
        norms = np.linalg.norm(noisy, axis=1, keepdims=True)
        noisy = noisy / np.maximum(norms, 1e-8)
        return int(np.sum(self._margins(noisy) > 0))

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
        if not self._is_fitted or not embedding:
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
