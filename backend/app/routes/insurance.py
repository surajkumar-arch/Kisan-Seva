"""FastAPI Router for PMFBY Smart Crop Insurance Claim Assessor."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from backend.app.services.insurance_claim_service import (
    assess_pmfby_crop_claim,
    PMFBY_ACTUARIAL_BENCHMARKS,
)

router = APIRouter(prefix="/api/insurance", tags=["crop-insurance"])


class InsuranceClaimRequest(BaseModel):
    farmer_name: str = Field("Suraj Kumar", description="Insured farmer full name")
    khasra_survey_no: str = Field("KH-842/19", description="Revenue survey/khasra land number")
    district_state: str = Field("Ludhiana, Punjab", description="Administrative district and state")
    crop: str = Field("Wheat", description="Insured crop name")
    insured_acres: float = Field(3.0, description="Insured land area in acres")
    loss_cause: str = Field("Hailstorm & Post-Harvest Inundation", description="Primary peril or calamity cause")
    visual_foliar_damage_pct: float = Field(45.0, description="Observed foliar lesion/damage %")
    satellite_drought_deficit_pct: float = Field(0.0, description="Remote sensing drought deficit %")
    assessed_actual_yield_q_acre: float = Field(8.5, description="Estimated post-calamity harvest yield in q/acre")
    gps_lat: float = Field(30.9010, description="Field GPS latitude coordinate")
    gps_lon: float = Field(75.8573, description="Field GPS longitude coordinate")


@router.get("/actuarial-scales")
async def get_actuarial_scales():
    """Lists PMFBY scale of finance and 7-year statutory threshold yields."""
    return {
        "scheme": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
        "benchmarks": PMFBY_ACTUARIAL_BENCHMARKS,
    }


@router.post("/assess-claim")
async def process_insurance_claim(payload: InsuranceClaimRequest):
    """Calculates shortfall yield, statutory indemnity payout, and signs claim dossier."""
    return assess_pmfby_crop_claim(
        farmer_name=payload.farmer_name,
        khasra_survey_no=payload.khasra_survey_no,
        district_state=payload.district_state,
        crop=payload.crop,
        insured_acres=payload.insured_acres,
        loss_cause=payload.loss_cause,
        visual_foliar_damage_pct=payload.visual_foliar_damage_pct,
        satellite_drought_deficit_pct=payload.satellite_drought_deficit_pct,
        assessed_actual_yield_q_acre=payload.assessed_actual_yield_q_acre,
        gps_lat=payload.gps_lat,
        gps_lon=payload.gps_lon,
    )


@router.get("/demo-claim")
async def demo_insurance_claim(crop: str = Query("Wheat")):
    """Quick 1-click test endpoint generating a verified PMFBY insurance claim dossier."""
    return assess_pmfby_crop_claim(
        farmer_name="Suraj Kumar Jaiswal",
        khasra_survey_no="KH-104/7A",
        district_state="Bathinda, Punjab",
        crop=crop,
        insured_acres=4.0,
        loss_cause="Unseasonal Hailstorm & Severe Foliar Lodging",
        visual_foliar_damage_pct=52.0,
        satellite_drought_deficit_pct=0.0,
        assessed_actual_yield_q_acre=7.2,
        gps_lat=30.2110,
        gps_lon=74.9455,
    )
