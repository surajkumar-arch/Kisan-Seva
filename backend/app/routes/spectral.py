"""FastAPI Router for Synthetic Multispectral NDVI & Vegetation Index Analysis."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from backend.app.services.spectral_service import analyze_spectral_health

router = APIRouter(prefix="/api/vision/spectral", tags=["spectral-ndvi"])


@router.post("/ndvi-analyze")
async def ndvi_analyze(image: UploadFile = File(...)):
    """Analyze crop leaf or canopy image with synthetic Near-Infrared NDVI reconstruction.

    Returns reconstructed NIR reflectance, NDVI/GNDVI/MSAVI matrices, false-color
    heatmaps, and sub-visual stress detection alerts.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image provided.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image data.")

    try:
        report = analyze_spectral_health(data)
        return report
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Spectral analysis error: {str(exc)}")
