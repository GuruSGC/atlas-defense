"""
Evaluation Metrics Calculator for ATLAS.
Computes standard academic benchmark security and retrieval metrics based on
actual experimental trial results.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ConfusionMatrix(BaseModel):
    true_positives: int = Field(..., description="Malicious candidate memories correctly QUARANTINED")
    false_positives: int = Field(..., description="Benign candidate memories incorrectly QUARANTINED")
    true_negatives: int = Field(..., description="Benign candidate memories correctly ACCEPTED")
    false_negatives: int = Field(..., description="Malicious candidate memories incorrectly ACCEPTED")


class DatasetInfo(BaseModel):
    name: str = Field(..., description="Benchmark identifier")
    version: str = Field(..., description="Benchmark version tag")
    benign_source: str = Field(..., description="Provenance of the benign class")
    adversarial_source: str = Field(..., description="Provenance of the adversarial class")
    seed: int = Field(..., description="Random seed used for sampling/shuffling, for reproducibility")


class Interval(BaseModel):
    low: float
    high: float


class EvaluationSummary(BaseModel):
    total_samples: int
    attack_samples: int
    benign_samples: int
    attack_success_rate_without_atlas: float
    attack_success_rate_with_atlas: float
    detection_rate: float
    false_positive_rate: float
    precision: float
    recall: float
    f1_score: float
    retrieval_accuracy: float
    confusion_matrix: ConfusionMatrix
    scenario_breakdown: List[Dict[str, Any]]
    dataset_info: DatasetInfo
    execution_timestamp: str
    protocol: str = "pipeline_static"
    precision_ci: Optional[Interval] = None
    recall_ci: Optional[Interval] = None
    f1_ci: Optional[Interval] = None
    false_positive_rate_ci: Optional[Interval] = None


class ProtocolResult(BaseModel):
    protocol: str
    description: str
    confusion_matrix: ConfusionMatrix
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    precision_ci: Interval
    recall_ci: Interval
    f1_ci: Interval
    false_positive_rate_ci: Interval
    auc: Optional[float] = None


class SetResult(BaseModel):
    name: str
    description: str
    kind: str                       # "attack" (rate is recall) or "benign" (rate is false-positive rate)
    n: int
    flagged: int
    rate: float
    rate_ci: Interval
    by_technique: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class BaselineRow(BaseModel):
    defense: str
    attacker: str                   # "unauthenticated" or "authenticated"
    dev_recall: float
    heldout_recall: float
    false_positive_rate_personachat: float
    false_positive_rate_technical: float
    false_positive_rate_updates: float


class SeedSweep(BaseModel):
    seeds: List[int]
    metrics: Dict[str, Dict[str, float]]


class AttackLabRow(BaseModel):
    scenario_id: str
    name: str
    category: str
    attack_succeeded_without_atlas: bool
    attack_succeeded_with_atlas: bool
    persisted_with_atlas: bool
    retrieved_with_atlas: bool
    behavior_compromised_with_atlas: bool
    trust_score_with_atlas: float


class EvaluationReport(BaseModel):
    generated_at: str
    dataset_info: DatasetInfo
    protocols: Dict[str, ProtocolResult]
    sets: Dict[str, SetResult]
    threshold_sweep: List[Dict[str, float]]
    baselines: List[BaselineRow]
    seed_sweep: SeedSweep
    attack_lab: List[AttackLabRow]
    notes: List[str]


def _r(x: float) -> float:
    return round(x, 3)


def compute_metrics(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    asr_without: float,
    asr_with: float,
    retrieval_acc: float,
    scenario_stats: List[Dict[str, Any]],
    dataset_info: DatasetInfo,
    timestamp: str,
    protocol: str = "pipeline_static",
    intervals: Optional[Dict[str, Interval]] = None,
) -> EvaluationSummary:
    """
    Computes Precision, Recall, F1, Detection Rate, and FPR from counts.

    Rates are rounded to three decimals once, here, so that a percentage shown anywhere
    (dashboard, paper, README) comes from the same value. F1 is 2TP / (2TP + FP + FN), which
    is 94/107 = 0.879 for 47/0/60/13. Rounding it to four decimals first and then to one
    decimal of percent would give 87.8 instead.
    """
    total = tp + fp + tn + fn
    attack_samples = tp + fn
    benign_samples = tn + fp

    recall = tp / attack_samples if attack_samples > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    fpr = fp / benign_samples if benign_samples > 0 else 0.0
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else 0.0
    intervals = intervals or {}

    return EvaluationSummary(
        total_samples=total,
        attack_samples=attack_samples,
        benign_samples=benign_samples,
        attack_success_rate_without_atlas=_r(asr_without),
        attack_success_rate_with_atlas=_r(asr_with),
        detection_rate=_r(recall),
        false_positive_rate=_r(fpr),
        precision=_r(precision),
        recall=_r(recall),
        f1_score=_r(f1),
        retrieval_accuracy=_r(retrieval_acc),
        confusion_matrix=ConfusionMatrix(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn
        ),
        scenario_breakdown=scenario_stats,
        dataset_info=dataset_info,
        execution_timestamp=timestamp,
        protocol=protocol,
        precision_ci=intervals.get("precision"),
        recall_ci=intervals.get("recall"),
        f1_ci=intervals.get("f1"),
        false_positive_rate_ci=intervals.get("fpr"),
    )
