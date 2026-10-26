"""
Named ATLAS configurations for the v4 evaluation, and a context manager that applies one and then
restores the process exactly as it was.

A configuration chooses the embedding backend, which anomaly detector runs and whether it starts from
a real benign profile, how behavioural consistency and historical consistency are computed, and which
learned classifier (if any) is installed.
"""

import contextlib
import copy
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np

from atlas.anomaly_detector import anomaly_detector
from atlas.learned import PoisonClassifier, clear_installed_classifier, get_poison_classifier, set_poison_classifier
from core.embeddings import embedding_service
from core.settings import settings


@dataclass(frozen=True)
class AtlasConfig:
    name: str
    backend: str = "hash"              # hash or local
    anomaly: str = "iforest"           # iforest or knn
    profile: bool = False              # start the anomaly detector from real benign embeddings
    behav: str = "regex"               # regex, learned, or hybrid
    hist: str = "clusters"             # clusters, nli, or hybrid
    classifier: Optional[str] = None   # key into the classifiers passed to apply()
    description: str = ""


# The ladder used for the ablation, from the original prototype up to the full learned configuration.
LADDER = [
    AtlasConfig("original", description="The original prototype: hash embeddings, forest on 50 synthetic vectors, rule-based signals."),
    AtlasConfig("semantic_embeddings", backend="local", description="Original rules and forest, but on sentence-transformer embeddings."),
    AtlasConfig("profile_forest", backend="local", profile=True, description="Forest fitted on real benign embeddings."),
    AtlasConfig("profile_knn", backend="local", anomaly="knn", profile=True, description="k-nearest-neighbour anomaly detector on the benign profile."),
    AtlasConfig("learned_behaviour", backend="local", anomaly="knn", profile=True, behav="hybrid", classifier="finetuned",
                description="Adds the fine-tuned classifier to behavioural consistency (rules plus classifier)."),
    AtlasConfig("full", backend="local", anomaly="knn", profile=True, behav="hybrid", hist="hybrid", classifier="finetuned",
                description="Adds semantic NLI to historical consistency (rules plus NLI)."),
]
BY_NAME = {c.name: c for c in LADDER}


@contextlib.contextmanager
def applied(cfg: AtlasConfig, profile_embeddings: Optional[np.ndarray] = None,
            classifiers: Optional[dict] = None) -> Iterator[None]:
    """Apply cfg process-wide, then undo it. profile_embeddings are the benign train embeddings."""
    classifiers = classifiers or {}
    saved_settings = (settings.BEHAV_MODE, settings.HIST_MODE, settings.ANOMALY_DETECTOR)
    saved_backend = embedding_service.backend
    saved_detector = copy.copy(anomaly_detector.__dict__)
    try:
        embedding_service.set_backend(cfg.backend)
        settings.BEHAV_MODE, settings.HIST_MODE, settings.ANOMALY_DETECTOR = cfg.behav, cfg.hist, cfg.anomaly
        anomaly_detector.kind = cfg.anomaly
        anomaly_detector._profile = None
        anomaly_detector.set_dimension(embedding_service.dimension)
        if cfg.profile:
            if profile_embeddings is None:
                raise ValueError(f"{cfg.name} needs profile embeddings")
            anomaly_detector.set_profile(profile_embeddings.tolist())
        else:
            anomaly_detector.clear_profile()
        if cfg.classifier:
            set_poison_classifier(classifiers[cfg.classifier])
        else:
            set_poison_classifier(None)
        yield
    finally:
        settings.BEHAV_MODE, settings.HIST_MODE, settings.ANOMALY_DETECTOR = saved_settings
        embedding_service.set_backend(saved_backend)
        anomaly_detector.__dict__.clear()
        anomaly_detector.__dict__.update(saved_detector)
        clear_installed_classifier()
