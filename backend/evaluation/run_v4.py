"""
Benchmark v4 runner.

  python -m evaluation.run_v4 data          build the public part of the benchmark (downloads once)
  python -m evaluation.run_v4 classifier    train the fine-tuned and logistic classifiers on the train split
  python -m evaluation.run_v4 loso          leave-one-source-out for the fine-tuned classifier (about 45 minutes on a CPU)
  python -m evaluation.run_v4 anomaly | history | ladder | baselines | llm | adaptive | prb | sensitivity | latency
  python -m evaluation.run_v4 compose       merge every result file into results/benchmark_v4.json
  python -m evaluation.run_v4 all           everything above in order (hours on a CPU)

Each step writes its own file under evaluation/results/v4/, so a step can be rerun on its own.
"""

import datetime
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

from evaluation.datasets import load_benchmark
from evaluation.public_data import SOURCES

RESULTS = Path(__file__).resolve().parent / "results"
V4 = RESULTS / "v4"
MERGED = RESULTS / "benchmark_v4.json"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def composition() -> List[dict]:
    rows = []
    for name, items in load_benchmark().items():
        meta = SOURCES.get(name, {})
        rows.append({
            "source": name, "kind": "attack" if items[0].is_attack else "benign", "n": len(items),
            "train": sum(i.split == "train" for i in items), "test": sum(i.split == "test" for i in items),
            "origin": "public" if name in SOURCES else "written by the authors",
            "license": meta.get("license", "this repository"), "url": meta.get("url", ""),
        })
    return rows


def frozen_summary():
    d = _read(RESULTS / "frozen_heuristics_v4.json")
    if not d:
        return None
    per: Dict[str, List[int]] = {}
    for r in d["rows"]:
        a = per.setdefault(r["source"], [0, 0])
        a[0] += 1
        a[1] += int(r["quarantined"])
    return {"note": d["note"], "per_source": {k: {"n": n, "flagged": f, "rate": round(f / n, 4)} for k, (n, f) in sorted(per.items())}}


def compose() -> dict:
    rows = composition()
    out = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "composition": rows,
        "totals": {"cases": sum(r["n"] for r in rows), "attacks": sum(r["n"] for r in rows if r["kind"] == "attack"),
                   "benign": sum(r["n"] for r in rows if r["kind"] == "benign"),
                   "public_cases": sum(r["n"] for r in rows if r["origin"] == "public")},
        "frozen_original_on_all_items": frozen_summary(),
        "classifier": {"finetuned": _read(RESULTS / "classifier_finetuned.json"), "logreg": _read(RESULTS / "classifier_logreg.json"),
                       "loso_finetuned": _read(RESULTS / "classifier_loso_finetuned.json"),
                       "loso_logreg": _read(RESULTS / "classifier_loso_logreg.json"),
                       "style_control": _read(RESULTS / "style_control.json")},
    }
    for name in ("anomaly", "history", "ladder", "baselines", "llm_moderator", "adaptive", "prb", "sensitivity", "latency", "attack_lab"):
        out[name] = _read(V4 / f"{name}.json")
    MERGED.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def main(argv: List[str]) -> None:
    steps = argv or ["compose"]
    if steps == ["all"]:
        steps = ["data", "classifier", "loso", "anomaly", "history", "ladder", "baselines", "llm", "adaptive", "prb", "sensitivity", "latency", "compose"]
    for step in steps:
        print(f"== {step}", flush=True)
        if step == "data":
            from evaluation import public_data
            public_data.build()
        elif step == "classifier":
            from evaluation import learned_eval
            learned_eval.main(["train", "--kind", "logreg", "--out", "models/poison_logreg"])
            learned_eval.main(["train", "--kind", "finetuned", "--out", "models/poison_clf"])
        elif step == "loso":
            from evaluation import learned_eval
            learned_eval.main(["loso", "--kind", "logreg"])
            learned_eval.main(["loso", "--kind", "finetuned"])
        elif step == "anomaly":
            from evaluation import anomaly_eval
            anomaly_eval.run()
        elif step == "history":
            from evaluation import history_eval
            history_eval.run()
        elif step == "ladder":
            from evaluation import ladder_eval
            ladder_eval.run()
        elif step == "baselines":
            from evaluation import baselines_v4
            baselines_v4.run()
        elif step == "llm":
            from evaluation import llm_baseline
            llm_baseline.run()
        elif step == "adaptive":
            from evaluation import adaptive_eval
            adaptive_eval.main(["all"])
        elif step == "prb":
            from evaluation import prb
            prb.run()
        elif step == "sensitivity":
            from evaluation import sensitivity
            sensitivity.main()
        elif step == "attacklab":
            from evaluation import attack_lab_eval
            attack_lab_eval.run()
        elif step == "latency":
            from evaluation import latency_eval
            latency_eval.run()
        elif step == "compose":
            compose()
            print("wrote", MERGED)
        else:
            raise SystemExit(f"unknown step {step}")


if __name__ == "__main__":
    main(sys.argv[1:])
