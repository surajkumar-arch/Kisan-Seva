"""FastAPI Router for Precision Smart Tank-Mix & Phytotoxicity Calculator."""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from backend.app.services.tank_mix_service import (
    AGROCHEMICAL_REGISTRY,
    check_tank_mix_compatibility,
)

router = APIRouter(prefix="/api/spray", tags=["tank-mix"])


class TankMixRequest(BaseModel):
    chemical_a: str = Field("Chlorpyrifos 20% EC (Insecticide)", description="First chemical product")
    chemical_b: str = Field("Mancozeb 75% WP (Fungicide)", description="Second chemical product")
    water_ph: float = Field(7.8, description="Spray water pH")
    wind_speed_kmh: float = Field(8.0, description="Wind velocity in km/h")
    temperature_c: float = Field(28.0, description="Ambient air temperature in Celsius")
    humidity_pct: float = Field(60.0, description="Relative humidity percentage")
    tank_capacity_liters: float = Field(15.0, description="Backpack knapsack capacity in liters")


@router.get("/catalog")
async def get_chemical_catalog():
    """Returns list of registered agrochemicals, categories, and optimal pH windows."""
    return {
        "total_chemicals": len(AGROCHEMICAL_REGISTRY),
        "chemicals": list(AGROCHEMICAL_REGISTRY.keys()),
        "categories": list(set(v["category"] for v in AGROCHEMICAL_REGISTRY.values())),
    }


@router.post("/tank-mix-check")
async def analyze_tank_mix(payload: TankMixRequest):
    """Analyzes tank antagonism, alkaline hydrolysis risk, nozzle choice, and PHI safety."""
    return check_tank_mix_compatibility(
        item_a=payload.chemical_a,
        item_b=payload.chemical_b,
        water_ph=payload.water_ph,
        wind_speed_kmh=payload.wind_speed_kmh,
        temperature_c=payload.temperature_c,
        humidity_pct=payload.humidity_pct,
        tank_capacity_liters=payload.tank_capacity_liters,
    )


@router.get("/demo-check")
async def demo_tank_mix(scenario: str = Query("incompatible")):
    """Quick demo endpoint testing either incompatible or compatible mixes."""
    if scenario == "incompatible":
        # Copper + Trichoderma biocide hazard
        return check_tank_mix_compatibility(
            item_a="Copper Oxychloride (Fungicide)",
            item_b="Trichoderma viride (Bio-agent)",
            water_ph=8.2,
            wind_speed_kmh=12.0,
            temperature_c=31.0,
            humidity_pct=45.0,
        )
    else:
        # Mancozeb + Imidacloprid compatible mix
        return check_tank_mix_compatibility(
            item_a="Mancozeb 75% WP (Fungicide)",
            item_b="Imidacloprid 17.8% SL (Insecticide)",
            water_ph=6.8,
            wind_speed_kmh=7.0,
            temperature_c=27.0,
            humidity_pct=65.0,
        )
