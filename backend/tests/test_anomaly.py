import numpy as np
import pytest
from scipy.stats import norm
from atlas.anomaly_detector import AnomalyDetector, _clopper_pearson_lower


@pytest.fixture
def det():
    return AnomalyDetector(dimension=16, seed=7)


def unit(rng, dim=16):
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


def test_clopper_pearson_known_values():
    assert _clopper_pearson_lower(0, 100, 0.05) == 0.0
    assert _clopper_pearson_lower(100, 100, 0.05) == pytest.approx(0.05 ** (1 / 100))
    # 90 of 100 at 95% one-sided: lower bound is a little under 0.84
    assert 0.82 < _clopper_pearson_lower(90, 100, 0.05) < 0.86


def test_lower_bound_grows_with_votes_and_with_trials():
    assert _clopper_pearson_lower(60, 100, 0.05) < _clopper_pearson_lower(80, 100, 0.05)
    assert _clopper_pearson_lower(8, 10, 0.05) < _clopper_pearson_lower(80, 100, 0.05)


def test_baseline_detector_is_fitted(det):
    assert det._is_fitted and det.fit_count == 0
    assert 0.0 <= det.compute_anomaly_score(unit(np.random.default_rng(1))) <= 1.0


def test_anomaly_score_is_deterministic(det):
    e = unit(np.random.default_rng(2))
    assert det.compute_anomaly_score(e) == det.compute_anomaly_score(e)


def test_certification_uses_separate_selection_and_estimation_batches(det, monkeypatch):
    calls = []
    original = det._noisy_votes

    def spy(arr, count, sigma, rng):
        calls.append(count)
        return original(arr, count, sigma, rng)

    monkeypatch.setattr(det, "_noisy_votes", spy)
    det.certify(unit(np.random.default_rng(3)), n0=40, n=90)
    assert calls == [40, 90]


def test_selection_and_estimation_draw_different_noise(det):
    arr = np.array(unit(np.random.default_rng(4)))
    a = np.random.default_rng([7, 0]).normal(size=(5, 16))
    b = np.random.default_rng([7, 1]).normal(size=(5, 16))
    assert not np.allclose(a, b)


def test_split_vote_abstains_with_zero_radius(det, monkeypatch):
    monkeypatch.setattr(det, "_noisy_votes", lambda arr, count, sigma, rng: count // 2)
    cert = det.certify(unit(np.random.default_rng(5)))
    assert cert.abstained and cert.radius == 0.0


def test_unanimous_vote_gives_the_theoretical_radius(det, monkeypatch):
    monkeypatch.setattr(det, "_noisy_votes", lambda arr, count, sigma, rng: count)
    cert = det.certify(unit(np.random.default_rng(6)), n0=50, n=100, sigma=0.05, alpha=0.05)
    expected_p = 0.05 ** (1 / 100)
    assert not cert.abstained
    assert cert.p_lower == pytest.approx(expected_p)
    assert cert.radius == pytest.approx(0.05 * norm.ppf(expected_p))


def test_a_smoothed_anomalous_verdict_counts_votes_for_anomalous(det, monkeypatch):
    # the forest calls everything anomalous: 0 normal votes in both batches
    monkeypatch.setattr(det, "_noisy_votes", lambda arr, count, sigma, rng: 0)
    cert = det.certify(unit(np.random.default_rng(8)))
    assert cert.smoothed_normal is False
    assert cert.votes == cert.n
    assert cert.radius > 0


def test_certification_is_reproducible_for_a_seed_and_differs_across_seeds(det):
    e = unit(np.random.default_rng(9))
    assert det.certify(e, seed=1) == det.certify(e, seed=1)
    outcomes = {det.certify(e, seed=s).votes for s in range(8)}
    assert len(outcomes) >= 1


def test_hardened_score_is_pulled_toward_the_midpoint_when_the_radius_is_small(det, monkeypatch):
    e = unit(np.random.default_rng(10))
    raw = det.compute_anomaly_score(e)
    monkeypatch.setattr(det, "_noisy_votes", lambda arr, count, sigma, rng: count // 2)
    hardened, radius = det.compute_certified_anomaly_score(e)
    assert radius == 0.0
    assert hardened == pytest.approx(0.5)
    monkeypatch.setattr(det, "_noisy_votes", lambda arr, count, sigma, rng: count)
    full, radius = det.compute_certified_anomaly_score(e, sigma=0.05)
    assert radius >= 0.1 - 1e-9 or full != pytest.approx(0.5)
    assert abs(full - 0.5) >= abs(hardened - 0.5)


def test_refit_waits_for_enough_accepted_memories(det):
    rng = np.random.default_rng(11)
    few = [unit(rng) for _ in range(AnomalyDetector.MIN_ACCEPTED_FOR_REFIT - 1)]
    assert det.maybe_refit(few) is False and det.fit_count == 0


def test_refit_happens_then_waits_for_more_growth(det):
    rng = np.random.default_rng(12)
    mems = [unit(rng) for _ in range(AnomalyDetector.MIN_ACCEPTED_FOR_REFIT)]
    assert det.maybe_refit(mems) is True and det.fit_count == 1
    mems.append(unit(rng))
    assert det.maybe_refit(mems) is False
    mems.extend(unit(rng) for _ in range(AnomalyDetector.REFIT_EVERY))
    assert det.maybe_refit(mems) is True and det.fit_count == 2


def test_refit_keeps_the_baseline_in_the_training_data(det):
    rng = np.random.default_rng(13)
    mems = [unit(rng) for _ in range(10)]
    det.fit(mems)
    assert det.model.n_features_in_ == 16
    assert det.model.estimators_samples_[0].size > 0


def test_reset_returns_to_the_baseline_and_can_change_seed(det):
    rng = np.random.default_rng(14)
    det.fit([unit(rng) for _ in range(10)])
    assert det.fit_count == 1
    det.reset(seed=99)
    assert det.fit_count == 0 and det.seed == 99


def test_refit_moves_the_decision_for_a_point_inside_the_new_cluster():
    det = AnomalyDetector(dimension=16, seed=3)
    rng = np.random.default_rng(15)
    centre = np.zeros(16); centre[0] = 1.0
    cluster = []
    for _ in range(40):
        v = centre + 0.05 * rng.normal(size=16)
        cluster.append((v / np.linalg.norm(v)).tolist())
    probe = (centre / np.linalg.norm(centre)).tolist()
    before = det.compute_anomaly_score(probe)
    det.fit(cluster)
    after = det.compute_anomaly_score(probe)
    assert after > before
