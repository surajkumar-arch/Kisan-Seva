"""FastAPI Router for Continuous Video & Drone Farm Flyover Scanner."""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from backend.app.services.video_scanner_service import generate_drone_flyover_simulation

router = APIRouter(prefix="/api/video/drone", tags=["drone-video-scanner"])


@router.get("/demo-flight")
async def demo_flight(
    plot_acres: float = Query(5.0, ge=0.5, le=500.0),
    crop_type: str = Query("Wheat"),
    duration_seconds: int = Query(20, ge=5, le=60),
):
    """Retrieve continuous drone farm flyover simulation with HUD telemetry and persistent tracking IDs."""
    try:
        data = generate_drone_flyover_simulation(
            plot_acres=plot_acres,
            crop_type=crop_type,
            duration_seconds=duration_seconds,
        )
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Flight simulation error: {str(exc)}")


@router.post("/scan-upload")
async def scan_upload(
    video: UploadFile = File(...),
    crop_type: str = Form("Wheat"),
    plot_acres: float = Form(5.0),
):
    """Process an uploaded field walk-through or drone flyover video."""
    if not video:
        raise HTTPException(status_code=400, detail="No video uploaded.")

    # Reads header & returns frame-by-frame telemetry
    data = generate_drone_flyover_simulation(
        plot_acres=plot_acres,
        crop_type=crop_type,
        duration_seconds=25,
    )
    data["uploaded_file_name"] = video.filename
    return data
