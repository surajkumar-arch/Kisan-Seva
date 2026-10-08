"""API Router for Phase 8: Model Evaluation & Performance Metrics."""

from fastapi import APIRouter, HTTPException, Query
from backend.app.evaluation.evaluate_models import run_full_system_evaluation

router = APIRouter(prefix="/api/evaluation", tags=["Model Evaluation & Benchmarks"])

# Cache evaluation report in memory so subsequent queries are instant
_CACHED_EVAL_REPORT = None


@router.get(
    "/metrics",
    summary="Multi-Modal Machine Learning Evaluation Report",
    description="Returns quantitative validation metrics (Accuracy, F1-scores, latencies, confusion matrices) across Soil, Plant, Fusion, and Crop models.",
)
async def get_evaluation_metrics(force_rerun: bool = Query(False, description="Re-run evaluation benchmark")):
    """Retrieve system-wide ML evaluation benchmarks."""
    global _CACHED_EVAL_REPORT
    try:
        if _CACHED_EVAL_REPORT is None or force_rerun:
            _CACHED_EVAL_REPORT = run_full_system_evaluation()
        return _CACHED_EVAL_REPORT
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate evaluation report: {str(exc)}")
