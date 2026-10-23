"""
FastAPI Routes for Bio-Surveillance & Epidemic Swarm Early Warning
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
from backend.app.services.epidemic_swarm_service import epidemic_swarm_service

router = APIRouter(prefix="/api/epidemic", tags=["Bio-Surveillance & Epidemic Swarm"])


class EpidemicSimulationRequest(BaseModel):
    pathogen_key: str = Field("yellow_rust", description="Pathogen identifier (yellow_rust, fall_armyworm, pink_bollworm, late_blight)")
    origin_district: str = Field("Hoshiarpur, Punjab", description="District epicentre of infection foci")
    active_focal_cases: int = Field(8, description="Number of confirmed infected farm plots in cluster")
    ambient_temp_c: float = Field(15.5, description="Regional ambient temperature in Celsius")
    ambient_humidity_pct: float = Field(82.0, description="Regional relative humidity percentage")
    wind_speed_kmh: float = Field(14.0, description="Average wind velocity in km/h")
    wind_azimuth_degrees: Optional[float] = Field(55.0, description="Wind azimuth angle (0=N, 90=E, 180=S, 270=W)")
    surrounding_crop_acreage: Optional[float] = Field(5000.0, description="Total susceptible host crop acreage in 30km radius")


@router.get("/pathogens")
def get_monitored_pathogens():
    """
    List monitored biosecurity pathogens, incubation metrics, and biological vectors.
    """
    return {
        "success": True,
        "monitored_pathogens": epidemic_swarm_service.PATHOGEN_PROPERTIES
    }


@router.post("/simulate")
def simulate_epidemic_endpoint(req: EpidemicSimulationRequest):
    """
    Simulate epidemic dispersal corridor, compute reproduction R0, and generate barrier protocols.
    """
    return epidemic_swarm_service.simulate_epidemic_spread(
        pathogen_key=req.pathogen_key,
        origin_district=req.origin_district,
        active_focal_cases=req.active_focal_cases,
        ambient_temp_c=req.ambient_temp_c,
        ambient_humidity_pct=req.ambient_humidity_pct,
        wind_speed_kmh=req.wind_speed_kmh,
        wind_azimuth_degrees=req.wind_azimuth_degrees if req.wind_azimuth_degrees is not None else 55.0,
        surrounding_crop_acreage=req.surrounding_crop_acreage if req.surrounding_crop_acreage is not None else 5000.0
    )


@router.get("/demo")
def get_demo_epidemic_simulation():
    """
    Return standard benchmark bio-surveillance scenario (Fall Armyworm invasion in Davanagere, Karnataka).
    """
    return epidemic_swarm_service.simulate_epidemic_spread(
        pathogen_key="fall_armyworm",
        origin_district="Davanagere, Karnataka",
        active_focal_cases=12,
        ambient_temp_c=27.5,
        ambient_humidity_pct=68.0,
        wind_speed_kmh=16.5,
        wind_azimuth_degrees=80.0,
        surrounding_crop_acreage=8500.0
    )
