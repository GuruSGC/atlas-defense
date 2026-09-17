"""
Evaluation Metrics Calculator for ATLAS.
Computes standard academic benchmark security and retrieval metrics based on
actual experimental trial results.
"""

from typing import List, Dict, Any
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
    timestamp: str
) -> EvaluationSummary:
    """
    Computes Precision, Recall, F1, Detection Rate, and FPR from counts.
    """
    total = tp + fp + tn + fn
    attack_samples = tp + fn
    benign_samples = tn + fp

    # Detection Rate / Recall = TP / (TP + FN)
    recall = tp / attack_samples if attack_samples > 0 else 1.0
    detection_rate = recall

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0

    # False Positive Rate = FP / (FP + TN)
    fpr = fp / benign_samples if benign_samples > 0 else 0.0

    # F1 Score
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return EvaluationSummary(
        total_samples=total,
        attack_samples=attack_samples,
        benign_samples=benign_samples,
        attack_success_rate_without_atlas=round(asr_without, 4),
        attack_success_rate_with_atlas=round(asr_with, 4),
        detection_rate=round(detection_rate, 4),
        false_positive_rate=round(fpr, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        retrieval_accuracy=round(retrieval_acc, 4),
        confusion_matrix=ConfusionMatrix(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn
        ),
        scenario_breakdown=scenario_stats,
        dataset_info=dataset_info,
        execution_timestamp=timestamp
    )
