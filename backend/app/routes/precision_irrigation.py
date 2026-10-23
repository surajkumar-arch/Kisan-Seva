"""
FastAPI Routes for Precision Smart Irrigation & Solar Pump Digital Twin
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
from backend.app.services.precision_irrigation_service import precision_irrigation_service

router = APIRouter(prefix="/api/irrigation", tags=["Precision Smart Irrigation & Solar Twin"])


class IrrigationScheduleRequest(BaseModel):
    farm_id: str = Field("PB-BTI-014", description="Unique Farm / Zone Identifier")
    crop_name: str = Field("wheat", description="Crop variety (wheat, rice, cotton, maize, sugarcane, tomato, potato)")
    growth_stage: str = Field("flowering", description="Crop phenology (initial, vegetative, flowering, grain_filling, maturation)")
    soil_type: str = Field("loam", description="Soil profile classification")
    farm_area_acres: float = Field(4.5, description="Field area in acres")
    current_soil_moisture_pct: float = Field(16.5, description="Volumetric soil water content (%)")
    daily_et0_mm: Optional[float] = Field(5.2, description="Reference evapotranspiration (mm/day)")
    pump_rating_hp: Optional[float] = Field(5.0, description="Solar pump horsepower capacity")
    irrigation_method: Optional[str] = Field("drip", description="Irrigation system (drip, sprinkler, surface)")


@router.get("/soil-types")
def get_soil_hydraulic_types():
    """
    List supported soil hydraulic profiles and water capacities.
    """
    return {
        "success": True,
        "soil_profiles": precision_irrigation_service.SOIL_HYDRAULIC_PROPERTIES,
        "crop_phenology_supported": list(precision_irrigation_service.CROP_KC_TABLE.keys())
    }


@router.post("/schedule")
def calculate_irrigation_endpoint(req: IrrigationScheduleRequest):
    """
    Calculate water deficit, pump duration, solar synchronization window, and MQTT packet.
    """
    return precision_irrigation_service.calculate_irrigation_schedule(
        farm_id=req.farm_id,
        crop_name=req.crop_name,
        growth_stage=req.growth_stage,
        soil_type=req.soil_type,
        farm_area_acres=req.farm_area_acres,
        current_soil_moisture_pct=req.current_soil_moisture_pct,
        daily_et0_mm=req.daily_et0_mm if req.daily_et0_mm is not None else 5.2,
        pump_rating_hp=req.pump_rating_hp if req.pump_rating_hp is not None else 5.0,
        irrigation_method=req.irrigation_method if req.irrigation_method is not None else "drip"
    )


@router.get("/demo")
def get_demo_irrigation():
    """
    Return benchmark precision irrigation and solar digital twin schedule.
    """
    return precision_irrigation_service.calculate_irrigation_schedule(
        farm_id="GJ-AMR-781",
        crop_name="cotton",
        growth_stage="flowering",
        soil_type="black_cotton",
        farm_area_acres=6.0,
        current_soil_moisture_pct=21.0,
        daily_et0_mm=5.8,
        pump_rating_hp=7.5,
        irrigation_method="drip"
    )
