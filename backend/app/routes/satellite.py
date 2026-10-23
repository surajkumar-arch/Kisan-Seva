"""FastAPI Router for Satellite Hydrological & Drought Risk Profiler."""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from backend.app.services.satellite_moisture_service import evaluate_field_hydrology

router = APIRouter(prefix="/api/satellite", tags=["satellite-moisture"])


class HydrologyRequest(BaseModel):
    crop: str = Field("Wheat", description="Target crop name")
    soil_type: str = Field("Alluvial Soil", description="Classified soil taxonomy")
    temperature_c: float = Field(30.0, description="Ambient air temperature in Celsius")
    humidity_pct: float = Field(55.0, description="Relative humidity percentage")
    rainfall_14d_mm: float = Field(12.0, description="Cumulative rainfall over last 14 days in mm")
    growth_stage: str = Field("mid", description="Crop growth stage: initial, mid, late")
    land_area_acres: float = Field(2.5, description="Total cultivated land area in acres")


@router.post("/moisture-profile")
async def get_moisture_profile(payload: HydrologyRequest):
    """Calculates synthetic NDWI, evapotranspiration deficit, and drip irrigation prescription."""
    return evaluate_field_hydrology(
        crop=payload.crop,
        soil_type=payload.soil_type,
        temperature_c=payload.temperature_c,
        humidity_pct=payload.humidity_pct,
        rainfall_14d_mm=payload.rainfall_14d_mm,
        growth_stage=payload.growth_stage,
        land_area_acres=payload.land_area_acres,
    )


@router.get("/demo-profile")
async def get_demo_profile(crop: str = Query("Wheat")):
    """Quick 1-click test endpoint for instant evaluation on web and mobile."""
    return evaluate_field_hydrology(
        crop=crop,
        soil_type="Alluvial Soil",
        temperature_c=32.0,
        humidity_pct=48.0,
        rainfall_14d_mm=5.0,
        growth_stage="mid",
        land_area_acres=2.0,
    )
