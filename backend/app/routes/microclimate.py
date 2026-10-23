"""
FastAPI Routes for Microclimate, Vapor Pressure Deficit, and Spray Windows
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
from backend.app.services.microclimate_service import microclimate_service

router = APIRouter(prefix="/api/microclimate", tags=["Microclimate & Spray Early Warning"])


class MicroclimateRequest(BaseModel):
    farm_name: str = Field("Harit Krishi Kshetra", description="Name of the farm")
    location: str = Field("Karnal, Haryana", description="Geographic location")
    crop_type: str = Field("Wheat", description="Crop variety")
    temp_c: float = Field(26.5, description="Current ambient temperature in Celsius")
    temp_max_c: float = Field(31.0, description="Projected maximum temperature")
    temp_min_c: float = Field(14.0, description="Projected minimum overnight temperature")
    relative_humidity: float = Field(62.0, description="Relative humidity percentage")
    wind_kmh: float = Field(9.5, description="Wind speed in km/h")
    soil_temp_c: Optional[float] = Field(22.0, description="Soil temperature at 5cm depth")
    rain_probability_pct: Optional[float] = Field(15.0, description="Rainfall probability percentage")


@router.post("/analyze")
def analyze_microclimate_endpoint(req: MicroclimateRequest):
    """
    Compute microclimate parameters, Vapor Pressure Deficit, and spray viability window.
    """
    return microclimate_service.analyze_microclimate(
        farm_name=req.farm_name,
        location=req.location,
        crop_type=req.crop_type,
        temp_c=req.temp_c,
        temp_max_c=req.temp_max_c,
        temp_min_c=req.temp_min_c,
        relative_humidity=req.relative_humidity,
        wind_kmh=req.wind_kmh,
        soil_temp_c=req.soil_temp_c if req.soil_temp_c is not None else 22.0,
        rain_probability_pct=req.rain_probability_pct if req.rain_probability_pct is not None else 15.0
    )


@router.get("/demo")
def get_demo_microclimate():
    """
    Return standard demo benchmark microclimate analysis.
    """
    return microclimate_service.analyze_microclimate(
        farm_name="Pragati Farm Cluster #4",
        location="Nashik, Maharashtra",
        crop_type="Grape / Tomato",
        temp_c=28.4,
        temp_max_c=34.2,
        temp_min_c=16.8,
        relative_humidity=58.0,
        wind_kmh=11.2,
        soil_temp_c=24.0,
        rain_probability_pct=5.0
    )
