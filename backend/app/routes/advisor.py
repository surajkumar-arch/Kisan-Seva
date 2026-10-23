"""Action Advisor endpoints (fertilizer plan, cost and revenue estimate)."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.models.crop_net import CROP_CATALOG
from backend.app.services import advisor_service as advisor

router = APIRouter(prefix="/api/advisor", tags=["action-advisor"])


class EconomicsRequest(BaseModel):
    crop: str
    area_acres: float = Field(1.0, gt=0, le=10000)
    nitrogen: Optional[float] = Field(None, ge=0)
    phosphorus: Optional[float] = Field(None, ge=0)
    potassium: Optional[float] = Field(None, ge=0)
    state: Optional[str] = None


class CompareRequest(BaseModel):
    crops: List[str] = Field(..., min_length=1)
    area_acres: float = Field(1.0, gt=0, le=10000)
    nitrogen: Optional[float] = Field(None, ge=0)
    phosphorus: Optional[float] = Field(None, ge=0)
    potassium: Optional[float] = Field(None, ge=0)
    state: Optional[str] = None


@router.get("/crops")
async def crops():
    """Crops the advisor can price."""
    return {
        "crops": [
            {"crop": name, "has_msp": advisor.MSP_INR_PER_QUINTAL.get(name) is not None}
            for name in CROP_CATALOG
        ]
    }


@router.post("/economics")
async def economics(req: EconomicsRequest):
    """Fertilizer plan, cost, expected yield and indicative revenue for one crop."""
    try:
        return advisor.build_economics(
            req.crop, req.area_acres, req.nitrogen, req.phosphorus, req.potassium, req.state
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown crop '{req.crop}'.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/compare")
async def compare(req: CompareRequest):
    """Rank several crops by margin after fertilizer cost."""
    rows = advisor.compare_crops(
        req.crops, req.area_acres, req.nitrogen, req.phosphorus, req.potassium, req.state
    )
    if not rows:
        raise HTTPException(status_code=404, detail="None of the requested crops are known.")
    return {"area_acres": req.area_acres, "ranking": rows}


@router.get("/mandi")
async def mandi(crop: str = Query(...), state: Optional[str] = None):
    """Live modal mandi price when DATA_GOV_IN_API_KEY is set, else the MSP reference."""
    if crop not in CROP_CATALOG:
        raise HTTPException(status_code=404, detail=f"Unknown crop '{crop}'.")
    live = advisor.live_mandi_price(crop, state)
    msp = advisor.MSP_INR_PER_QUINTAL.get(crop)
    return {
        "crop": crop,
        "live": live,
        "reference_msp_inr_per_quintal": msp,
        "basis": live["source"] if live else "Reference MSP/FRP 2024-25 (verify current rate)",
    }
