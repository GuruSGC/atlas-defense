import pytest
from atlas.scoring import ScoringEngine
from memory.memory_models import MemoryStatus, TrustSignalsBreakdown


def sig(rel=0.9, anom=0.9, hist=0.9, behav=0.9, src=0.9, temp=0.9):
    return TrustSignalsBreakdown(semantic_relevance=rel, embedding_anomaly=anom, historical_consistency=hist,
                                 behavioral_consistency=behav, source_reliability=src, temporal_trust=temp)


engine = ScoringEngine()


def test_clean_candidate_is_accepted():
    score, status, _, _ = engine.compute_trust_score(sig())
    assert status == MemoryStatus.ACCEPTED
    assert score == pytest.approx(0.9, abs=1e-3)


def test_weights_are_equal_by_default():
    _, _, contribs, _ = engine.compute_trust_score(sig(rel=0.6))
    assert len(set(round(v, 4) for k, v in contribs.items() if k != "semantic_relevance")) == 1


def test_a_low_average_alone_quarantines():
    _, status, _, _ = engine.compute_trust_score(sig(0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
    assert status == MemoryStatus.QUARANTINED


def test_veto_condition_is_strict_at_the_trigger():
    """Exactly 0.40 does not trigger the historical veto; anything below it does."""
    assert engine.evaluate_veto(sig(hist=0.40))[1] == []
    assert len(engine.evaluate_veto(sig(hist=0.3999))[1]) == 1


def test_contradiction_penalty_of_point_sixty_lands_below_the_trigger():
    """0.95 - 0.55 is exactly 0.40 and misses the veto; the shipped penalty is 0.60."""
    assert round(0.95 - 0.55, 10) == 0.40
    assert 0.95 - 0.60 < 0.40


@pytest.mark.parametrize("field,trigger,floor", [
    ("hist", 0.40, 0.15), ("behav", 0.40, 0.20), ("anom", 0.30, 0.40),
])
def test_each_critical_signal_vetoes_with_its_own_floor(field, trigger, floor):
    s = sig(**{field: 0.01})
    mult, violations = engine.evaluate_veto(s)
    assert len(violations) == 1
    assert mult == pytest.approx(floor)
    assert engine.compute_trust_score(s)[1] == MemoryStatus.QUARANTINED


def test_veto_multiplier_scales_with_the_signal_between_floor_and_trigger():
    mult, _ = engine.evaluate_veto(sig(hist=0.30))
    assert mult == pytest.approx(0.30 / 0.50)


def test_one_bad_critical_signal_cannot_be_averaged_away():
    s = sig(rel=1.0, anom=1.0, hist=0.1, behav=1.0, src=1.0, temp=1.0)
    score, status, _, reason = engine.compute_trust_score(s)
    assert status == MemoryStatus.QUARANTINED
    assert "contradiction" in reason.lower()
    assert score < 0.65


def test_non_critical_signals_do_not_veto():
    s = sig(src=0.05, temp=0.05, rel=0.05)
    _, violations = engine.evaluate_veto(s)
    assert violations == []


def test_threshold_boundary_is_inclusive():
    custom = ScoringEngine(threshold=0.9)
    assert custom.compute_trust_score(sig())[1] == MemoryStatus.ACCEPTED
    assert ScoringEngine(threshold=0.91).compute_trust_score(sig())[1] == MemoryStatus.QUARANTINED
