"""
Evaluation API Endpoints for ATLAS.
Provides empirical metrics on Attack Success Rate, Detection Rate, Precision,
Recall, F1-Score, and False Positive Rate, plus the full report with protocol
comparison, confidence intervals, baselines, seed sweep, and Attack Lab results.
"""

import json
from fastapi import APIRouter, HTTPException
from evaluation.metrics import EvaluationReport, EvaluationSummary
from evaluation.experiment_runner import experiment_runner
from evaluation.run_report import RESULTS_PATH
from evaluation.run_v4 import MERGED as V4_PATH

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.get("/summary", response_model=EvaluationSummary)
async def get_evaluation_summary() -> EvaluationSummary:
    """
    Retrieve current empirical evaluation summary metrics (primary benchmark, leakage-free protocol).
    """
    return experiment_runner.get_summary()


@router.post("/run", response_model=EvaluationSummary)
async def run_fresh_evaluation(live: bool = False) -> EvaluationSummary:
    """
    Re-run the primary benchmark. Offline and deterministic by default; live=true uses the
    configured embedding model and extractor.
    """
    return experiment_runner.run_benchmark(live=live)


@router.get("/report", response_model=EvaluationReport)
async def get_full_report() -> EvaluationReport:
    """
    The full report: protocol comparison, intervals, extra datasets, baselines, seed sweep,
    and Attack Lab results. Served from the committed results file.
    """
    if experiment_runner._cached_report is not None:
        return experiment_runner._cached_report
    if RESULTS_PATH.exists():
        return EvaluationReport(**json.loads(RESULTS_PATH.read_text(encoding="utf-8")))
    raise HTTPException(status_code=404, detail="No report yet. POST /api/evaluation/report/run to generate one.")


@router.post("/report/run", response_model=EvaluationReport)
async def run_full_report(seeds: int = 10, live: bool = False) -> EvaluationReport:
    """
    Recompute the full report. This takes a few minutes.
    """
    return experiment_runner.run_full_report(seeds=seeds, live=live)


@router.get("/v4")
async def get_benchmark_v4() -> dict:
    """
    Benchmark v4: composition of the 1,700-case benchmark, the ablation ladder, baselines, anomaly detectors,
    the learned classifier, the adaptive adversary, and sensitivity analyses. Served from
    evaluation/results/benchmark_v4.json, which `python -m evaluation.run_v4 compose` writes.
    """
    if V4_PATH.exists():
        return json.loads(V4_PATH.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="No v4 results yet. Run python -m evaluation.run_v4 compose.")
