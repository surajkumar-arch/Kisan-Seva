"""FastAPI Router for Smart Mandi Arbitrage & Storage Economics."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from backend.app.services.mandi_arbitrage_service import (
    calculate_mandi_arbitrage,
    APMC_MANDI_HUBS,
    GOVT_MSP_TABLE,
)

router = APIRouter(prefix="/api/mandi", tags=["mandi-arbitrage"])


class ArbitrageRequest(BaseModel):
    crop: str = Field("Wheat", description="Commodity crop name")
    quantity_quintals: float = Field(60.0, description="Available harvest volume in Quintals")
    local_mandi_rate_inr: float = Field(2320.0, description="Current spot price at local mandi in ₹/q")
    diesel_price_per_liter: float = Field(90.0, description="Local diesel price in ₹/liter")
    truck_mileage_km_per_liter: float = Field(4.0, description="Truck fuel efficiency (km/L)")
    truck_capacity_quintals: float = Field(100.0, description="Truck load capacity in quintals")
    warehouse_storage_months: int = Field(2, description="Target warehouse holding months")
    storage_rent_per_quintal_month: float = Field(25.0, description="Warehouse storage charge ₹/q/month")


@router.get("/hubs")
async def get_mandi_hubs():
    """Lists major agricultural trading mandis and reference MSP."""
    return {
        "mandi_hubs": APMC_MANDI_HUBS,
        "msp_table": GOVT_MSP_TABLE,
    }


@router.post("/arbitrage-calculator")
async def evaluate_arbitrage(payload: ArbitrageRequest):
    """Calculates transport logistics arbitrage across hubs and warehouse storage holding curves."""
    return calculate_mandi_arbitrage(
        crop=payload.crop,
        quantity_quintals=payload.quantity_quintals,
        local_mandi_rate_inr=payload.local_mandi_rate_inr,
        diesel_price_per_liter=payload.diesel_price_per_liter,
        truck_mileage_km_per_liter=payload.truck_mileage_km_per_liter,
        truck_capacity_quintals=payload.truck_capacity_quintals,
        warehouse_storage_months=payload.warehouse_storage_months,
        storage_rent_per_quintal_month=payload.storage_rent_per_quintal_month,
    )


@router.get("/demo-arbitrage")
async def demo_arbitrage(crop: str = Query("Wheat")):
    """Instant 1-click test endpoint for mandi arbitrage calculation."""
    return calculate_mandi_arbitrage(
        crop=crop,
        quantity_quintals=75.0,
        local_mandi_rate_inr=2280.0,
        diesel_price_per_liter=89.5,
        truck_mileage_km_per_liter=4.2,
        truck_capacity_quintals=100.0,
        warehouse_storage_months=2,
        storage_rent_per_quintal_month=25.0,
    )
