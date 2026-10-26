"""
Generate the full evaluation report and write it to evaluation/results/latest.json.

    python -m evaluation.run_report            offline, deterministic, 10 seeds
    python -m evaluation.run_report --seeds 5
    python -m evaluation.run_report --live     use the live embedding model and extractor

The committed latest.json is what the README and the paper quote.
"""

import argparse
import json
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "latest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    from evaluation.experiment_runner import experiment_runner
    report = experiment_runner.run_full_report(seeds=args.seeds, live=args.live)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")

    print(f"wrote {RESULTS_PATH}")
    for name, p in report.protocols.items():
        c = p.confusion_matrix
        print(f"{name:24s} TP={c.true_positives:2d} FN={c.false_negatives:2d} FP={c.false_positives:2d} TN={c.true_negatives:2d} "
              f"recall={p.recall:.3f} precision={p.precision:.3f} f1={p.f1_score:.3f} fpr={p.false_positive_rate:.3f}")
    for name, s in report.sets.items():
        print(f"{name:24s} n={s.n:3d} flagged={s.flagged:3d} rate={s.rate:.3f} ci=[{s.rate_ci.low:.3f},{s.rate_ci.high:.3f}]")
    for row in report.attack_lab:
        print(f"{row.scenario_id:34s} without={row.attack_succeeded_without_atlas} with={row.attack_succeeded_with_atlas}")


if __name__ == "__main__":
    main()
