"""API routes for Live Environmental & Meteorological Sync (WAQI)."""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException, status
from backend.app.services.weather_service import fetch_live_environmental_data

router = APIRouter(prefix="/api/environment", tags=["Environmental & Meteorological Intelligence"])


@router.get("/live")
async def get_live_environment(
    location: str = Query("here", description="Location name, city (e.g. 'delhi', 'mumbai', 'patna'), or 'here'"),
    token: Optional[str] = Query(None, description="Optional custom WAQI API token"),
):
    """Fetch real-time ambient weather (temperature, humidity) and air quality metrics via WAQI API."""
    result = await fetch_live_environmental_data(location=location, token=token)
    if result.get("status") != "success":
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=result.get("message", "Failed to retrieve environmental data from WAQI API")
        )
    return result
