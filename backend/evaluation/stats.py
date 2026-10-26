"""
Small statistics helpers for the evaluation: confidence intervals, AUC, and a sweep over
the trust threshold. Nothing here depends on the rest of the system, so it can be tested
against known values.
"""

import math
import random
from typing import Dict, List, Sequence, Tuple

Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (two-sided, 95% by default)."""
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def bootstrap_interval(
    labels: Sequence[bool],
    flagged: Sequence[bool],
    statistic: str = "f1",
    iterations: int = 2000,
    seed: int = 42,
    level: float = 0.95,
) -> Tuple[float, float]:
    """
    Percentile bootstrap over items. `labels[i]` is True for an attack, `flagged[i]` is True
    when the defense quarantined it. statistic is one of "f1", "precision", "recall", "fpr".
    """
    n = len(labels)
    rng = random.Random(seed)
    values: List[float] = []
    for _ in range(iterations):
        tp = fp = tn = fn = 0
        for _ in range(n):
            i = rng.randrange(n)
            a, f = labels[i], flagged[i]
            if a and f: tp += 1
            elif a and not f: fn += 1
            elif not a and f: fp += 1
            else: tn += 1
        if statistic == "f1":
            values.append(f1_from_counts(tp, fp, fn))
        elif statistic == "precision":
            values.append(tp / (tp + fp) if tp + fp else 1.0)
        elif statistic == "recall":
            values.append(tp / (tp + fn) if tp + fn else 1.0)
        elif statistic == "fpr":
            values.append(fp / (fp + tn) if fp + tn else 0.0)
        else:
            raise ValueError(f"unknown statistic {statistic}")
    values.sort()
    lo = values[int(((1 - level) / 2) * iterations)]
    hi = values[min(iterations - 1, int((1 - (1 - level) / 2) * iterations))]
    return (lo, hi)


def auc_from_scores(benign_scores: Sequence[float], attack_scores: Sequence[float]) -> float:
    """
    Probability that a random benign item gets a higher trust score than a random attack
    (ties count half). 0.5 means no separation, 1.0 means perfect separation.
    """
    if not benign_scores or not attack_scores:
        return 0.5
    wins = 0.0
    for b in benign_scores:
        for a in attack_scores:
            if b > a:
                wins += 1.0
            elif b == a:
                wins += 0.5
    return wins / (len(benign_scores) * len(attack_scores))


def threshold_sweep(
    scored: Sequence[Tuple[float, bool, bool]],
    thresholds: Sequence[float],
) -> List[Dict[str, float]]:
    """
    scored items are (effective_trust_score, vetoed, is_attack). An item is accepted when its
    score reaches the threshold and no critical veto fired, so only the score part moves.
    """
    out: List[Dict[str, float]] = []
    for tau in thresholds:
        tp = fp = tn = fn = 0
        for score, vetoed, is_attack in scored:
            quarantined = vetoed or score < tau
            if is_attack and quarantined: tp += 1
            elif is_attack: fn += 1
            elif quarantined: fp += 1
            else: tn += 1
        out.append({
            "threshold": round(tau, 2),
            "precision": round(tp / (tp + fp), 3) if tp + fp else 1.0,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 1.0,
            "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0,
        })
    return out


def mean_sd_range(values: Sequence[float]) -> Dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"mean": round(mean, 4), "sd": round(sd, 4), "min": round(min(values), 4), "max": round(max(values), 4)}
