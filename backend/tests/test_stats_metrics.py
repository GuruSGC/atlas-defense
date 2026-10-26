import pytest
from evaluation import stats
from evaluation.metrics import DatasetInfo, compute_metrics

INFO = DatasetInfo(name="t", version="1", benign_source="b", adversarial_source="a", seed=1)


def metrics(tp, fp, tn, fn):
    return compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn, asr_without=1.0, asr_with=fn / (tp + fn) if tp + fn else 0.0,
                           retrieval_acc=1.0, scenario_stats=[], dataset_info=INFO, timestamp="now")


def test_wilson_interval_matches_hand_calculation():
    lo, hi = stats.wilson_interval(47, 60)
    assert lo == pytest.approx(0.6639, abs=0.002)
    assert hi == pytest.approx(0.8689, abs=0.002)


def test_wilson_interval_handles_the_edges():
    lo, hi = stats.wilson_interval(0, 60)
    assert lo == 0.0 and 0.05 < hi < 0.08
    lo, hi = stats.wilson_interval(60, 60)
    assert hi == pytest.approx(1.0) and lo > 0.92
    assert stats.wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_narrows_with_more_trials():
    a = stats.wilson_interval(8, 10)
    b = stats.wilson_interval(80, 100)
    assert (b[1] - b[0]) < (a[1] - a[0])


def test_f1_from_counts():
    assert stats.f1_from_counts(47, 0, 13) == pytest.approx(94 / 107)
    assert stats.f1_from_counts(0, 0, 0) == 0.0


def test_bootstrap_interval_contains_the_point_estimate_and_is_reproducible():
    labels = [True] * 60 + [False] * 60
    flagged = [True] * 47 + [False] * 13 + [False] * 60
    lo, hi = stats.bootstrap_interval(labels, flagged, "f1", iterations=400, seed=1)
    assert lo < stats.f1_from_counts(47, 0, 13) < hi
    assert (lo, hi) == stats.bootstrap_interval(labels, flagged, "f1", iterations=400, seed=1)


def test_bootstrap_rejects_unknown_statistics():
    with pytest.raises(ValueError):
        stats.bootstrap_interval([True], [True], "accuracy", iterations=5)


def test_auc():
    assert stats.auc_from_scores([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert stats.auc_from_scores([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert stats.auc_from_scores([0.5], [0.5]) == 0.5
    assert stats.auc_from_scores([], [0.5]) == 0.5


def test_threshold_sweep_recall_never_decreases_with_a_higher_threshold():
    scored = [(0.9, False, False), (0.8, False, False), (0.7, False, True), (0.4, False, True), (0.2, True, True)]
    rows = stats.threshold_sweep(scored, [0.3, 0.5, 0.7, 0.9])
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls)
    assert rows[-1]["false_positive_rate"] >= rows[0]["false_positive_rate"]


def test_a_veto_quarantines_regardless_of_the_threshold():
    rows = stats.threshold_sweep([(0.99, True, True)], [0.0, 0.5])
    assert all(r["recall"] == 1.0 for r in rows)


def test_mean_sd_range():
    out = stats.mean_sd_range([1.0, 2.0, 3.0])
    assert out["mean"] == 2.0 and out["sd"] == 1.0 and out["min"] == 1.0 and out["max"] == 3.0
    assert stats.mean_sd_range([5.0])["sd"] == 0.0


# ---- compute_metrics ----------------------------------------------------------------------------

def test_f1_for_the_original_confusion_matrix_is_87_9_not_87_8():
    m = metrics(tp=47, fp=0, tn=60, fn=13)
    assert m.f1_score == 0.879
    assert f"{m.f1_score * 100:.1f}" == "87.9"
    assert m.recall == 0.783 and m.precision == 1.0 and m.false_positive_rate == 0.0
    assert m.attack_success_rate_with_atlas == 0.217


def test_f1_equals_the_closed_form_for_other_counts():
    m = metrics(tp=30, fp=5, tn=55, fn=30)
    assert m.f1_score == round(2 * 30 / (2 * 30 + 5 + 30), 3)


def test_metrics_with_no_attacks_do_not_divide_by_zero():
    m = metrics(tp=0, fp=0, tn=10, fn=0)
    assert m.recall == 1.0 and m.f1_score == 0.0


def test_summary_carries_protocol_and_intervals():
    from evaluation.metrics import Interval
    m = compute_metrics(tp=1, fp=0, tn=1, fn=0, asr_without=1.0, asr_with=0.0, retrieval_acc=1.0, scenario_stats=[],
                        dataset_info=INFO, timestamp="t", protocol="p", intervals={"f1": Interval(low=0.1, high=0.9)})
    assert m.protocol == "p" and m.f1_ci.high == 0.9 and m.recall_ci is None
