"""Community Outbreak Radar endpoints (shared by the web dashboard and mobile app)."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.services import community_service as radar

router = APIRouter(prefix="/api/community", tags=["community-radar"])

DEMO_ENABLED = os.getenv("ENABLE_DEMO_ENDPOINTS", "true").lower() in ("1", "true", "yes")


class ReportRequest(BaseModel):
    condition: str = Field(..., min_length=2, description="Diagnosed disease name")
    confidence: float = Field(..., ge=0.0, le=1.0)
    district: Optional[str] = Field(None, description="District name from /regions")
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lon: Optional[float] = Field(None, ge=-180, le=180)
    crop: Optional[str] = None
    severity: Optional[str] = None


@router.get("/regions")
async def regions():
    """Supported districts (location is always snapped to these centroids)."""
    return {"regions": radar.list_regions()}


@router.post("/report", status_code=201)
async def report(req: ReportRequest):
    """Share an anonymous, confident disease diagnosis with the community."""
    try:
        saved = radar.submit_report(**req.model_dump())
    except radar.ReportRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"status": "accepted", "privacy": "Stored at district level only; no photo or identity.", **saved}


@router.get("/outbreaks")
async def outbreaks(days: int = Query(14, ge=1, le=90), include_demo: bool = True):
    """Hotspots per district and disease with trend and ALERT/WATCH level."""
    return radar.get_outbreaks(days=days, include_demo=include_demo)


@router.get("/nearby")
async def nearby(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(150.0, gt=0, le=1000),
    days: int = Query(14, ge=1, le=90),
    include_demo: bool = True,
):
    """Outbreaks near a farm, nearest first."""
    return radar.get_nearby(lat, lon, radius_km=radius_km, days=days, include_demo=include_demo)


@router.post("/demo/seed")
async def seed_demo():
    """Populate flagged DEMO reports for presentations."""
    if not DEMO_ENABLED:
        raise HTTPException(status_code=403, detail="Demo endpoints are disabled.")
    return {"status": "seeded", "reports_added": radar.seed_demo_data(), "flagged_as_demo": True}


@router.delete("/demo")
async def clear_demo():
    """Remove only demo reports; real community reports are untouched."""
    if not DEMO_ENABLED:
        raise HTTPException(status_code=403, detail="Demo endpoints are disabled.")
    return {"status": "cleared", "reports_removed": radar.clear_demo_data()}
