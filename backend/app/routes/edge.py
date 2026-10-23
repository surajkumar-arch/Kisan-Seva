"""FastAPI Router for Zero-Internet Edge ONNX INT8 Offline Engine."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from backend.app.services.edge_onnx_service import get_edge_benchmark_metrics, run_offline_edge_inference

router = APIRouter(prefix="/api/edge", tags=["edge-onnx-offline"])


@router.get("/benchmarks")
async def benchmarks():
    """Retrieve edge ONNX INT8 hardware benchmark metrics comparing Cloud vs Edge."""
    return get_edge_benchmark_metrics()


@router.post("/predict-offline")
async def predict_offline(image: UploadFile = File(...)):
    """Simulate zero-internet local edge ONNX inference.

    Executes on CPU SIMD threads in sub-25ms latency with zero cloud reliance.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image provided.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image data.")

    try:
        result = run_offline_edge_inference(data)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Edge inference error: {str(exc)}")
