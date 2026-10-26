from atlas.temporal import TemporalTracker
from atlas.trust_engine import trust_engine
from tests.conftest import make_candidate


def test_tracker_counts_inside_the_window_only():
    t = TemporalTracker(window_seconds=60)
    for ts in (0, 10, 20):
        t.record(ts)
    assert t.recent_count(now=25) == 3
    assert t.recent_count(now=75) == 1       # the events at 0 and 10 have aged out
    assert t.recent_count(now=500) == 0


def test_tracker_reset():
    t = TemporalTracker()
    t.record(1)
    t.reset()
    assert t.recent_count(now=2) == 0


def test_a_burst_lowers_temporal_trust(clean_state):
    scores = []
    for i in range(8):
        c = make_candidate(f"The team uses tool number {i}.")
        _, _, sig, _, _ = trust_engine.evaluate_candidate(c, now=1000 + i, record_event=True)
        scores.append(sig.temporal_trust)
    assert scores[0] == 0.90
    assert scores[-1] < 0.55
    assert all(a >= b for a, b in zip(scores, scores[1:]))


def test_a_conversational_pace_keeps_temporal_trust_at_the_ceiling(clean_state):
    for i in range(6):
        c = make_candidate(f"The team uses tool number {i}.")
        _, _, sig, _, _ = trust_engine.evaluate_candidate(c, now=1000 + 120 * i, record_event=True)
        assert sig.temporal_trust == 0.90


def test_evaluation_without_record_event_leaves_history_unchanged(clean_state):
    c = make_candidate("The team uses a monorepo.")
    trust_engine.evaluate_candidate(c, now=5000)
    assert trust_engine.temporal.recent_count(now=5000) == 0
