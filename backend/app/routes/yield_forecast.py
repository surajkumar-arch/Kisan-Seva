"""FastAPI Router for Predictive Crop Yield & Biomass Forecasting Engine."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from backend.app.services.yield_forecast_service import simulate_crop_growth

router = APIRouter(prefix="/api/yield", tags=["yield-forecasting"])


class YieldForecastRequest(BaseModel):
    crop: str = Field("Wheat", description="Target crop for yield prediction")
    soil_type: str = Field("Alluvial Soil", description="Dominant soil taxonomy")
    land_area_acres: float = Field(2.5, description="Cultivated land area in acres")
    soil_nitrogen_kgha: float = Field(120.0, description="Available Nitrogen (N kg/ha)")
    soil_phosphorus_kgha: float = Field(35.0, description="Available Phosphorus (P kg/ha)")
    soil_potassium_kgha: float = Field(110.0, description="Available Potassium (K kg/ha)")
    soil_ph: float = Field(6.8, description="Soil pH reading")
    avg_temperature_c: float = Field(26.0, description="Mean temperature during season")
    cumulative_rainfall_mm: float = Field(280.0, description="Seasonal cumulative rainfall in mm")
    disease_severity_loss_pct: float = Field(0.0, description="Estimated foliar damage loss %")
    num_simulations: int = Field(1000, description="Number of Monte Carlo stochastic runs")


@router.post("/forecast")
async def forecast_crop_yield(payload: YieldForecastRequest):
    """Simulates physiological crop growth and runs 1,000-run Monte Carlo yield bounds."""
    return simulate_crop_growth(
        crop=payload.crop,
        soil_type=payload.soil_type,
        land_area_acres=payload.land_area_acres,
        soil_nitrogen_kgha=payload.soil_nitrogen_kgha,
        soil_phosphorus_kgha=payload.soil_phosphorus_kgha,
        soil_potassium_kgha=payload.soil_potassium_kgha,
        soil_ph=payload.soil_ph,
        avg_temperature_c=payload.avg_temperature_c,
        cumulative_rainfall_mm=payload.cumulative_rainfall_mm,
        disease_severity_loss_pct=payload.disease_severity_loss_pct,
        num_simulations=payload.num_simulations,
    )


@router.get("/demo")
async def demo_yield_forecast(crop: str = Query("Wheat")):
    """Quick 1-tap demo endpoint for instant yield and biomass curve visualization."""
    return simulate_crop_growth(
        crop=crop,
        soil_type="Alluvial Soil",
        land_area_acres=3.0,
        soil_nitrogen_kgha=130.0,
        soil_phosphorus_kgha=40.0,
        soil_potassium_kgha=125.0,
        soil_ph=6.8,
        avg_temperature_c=25.0,
        cumulative_rainfall_mm=320.0,
        disease_severity_loss_pct=5.0,
        num_simulations=1000,
    )
