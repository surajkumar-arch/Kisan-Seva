"""
FastAPI Routes for Farm Ledger, Cost Economics, and Carbon Credits
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from backend.app.services.carbon_ledger_service import carbon_ledger_service

router = APIRouter(prefix="/api/carbon", tags=["Farm Ledger & Carbon Credit Offset"])


class CarbonAuditRequest(BaseModel):
    farmer_name: str = Field("Sardar Gurpreet Singh", description="Farmer Name")
    farm_id: str = Field("PB-LDH-098", description="Unique Farm Identifier")
    farm_size_hectares: float = Field(3.5, description="Farm size in hectares")
    adopted_practices: List[str] = Field(
        ["residue_management", "conservation_tillage", "solar_pump"],
        description="Codes of adopted regenerative practices"
    )


class FarmFinancialRequest(BaseModel):
    farm_size_acres: float = Field(8.65, description="Farm area in acres")
    crop_type: str = Field("Wheat (HD-2967)", description="Crop variety")
    total_production_quintals: float = Field(190.0, description="Total yield in quintals")
    crop_sales_revenue: float = Field(437000.0, description="Gross mandi sales revenue in INR")
    expenses: Dict[str, float] = Field(
        {
            "seeds": 16500.0,
            "fertilizers": 28000.0,
            "crop_protection": 14500.0,
            "labor": 42000.0,
            "machinery_and_diesel": 36000.0,
            "irrigation": 12000.0,
            "harvesting_and_transport": 22000.0
        },
        description="Detailed itemized farm expenditures"
    )
    subsidies_received: Optional[float] = Field(18000.0, description="Government subsidies received in INR")


@router.get("/practices")
def get_carbon_practices():
    """
    List all supported carbon sequestration practices with coefficients and rates.
    """
    return {
        "success": True,
        "vcm_rate_inr_per_ton": carbon_ledger_service.INR_PER_TON_CO2E,
        "practices": carbon_ledger_service.CARBON_COEFFICIENTS
    }


@router.post("/audit")
def audit_carbon_credits_endpoint(req: CarbonAuditRequest):
    """
    Audit regenerative practices and issue cryptographically hashed carbon certificate.
    """
    return carbon_ledger_service.audit_carbon_credits(
        farmer_name=req.farmer_name,
        farm_id=req.farm_id,
        farm_size_hectares=req.farm_size_hectares,
        adopted_practices=req.adopted_practices
    )


@router.post("/farm-financials")
def calculate_financials_endpoint(req: FarmFinancialRequest):
    """
    Compute double-entry farm economics, cost per quintal, and ROI.
    """
    return carbon_ledger_service.calculate_farm_financials(
        farm_size_acres=req.farm_size_acres,
        crop_type=req.crop_type,
        total_production_quintals=req.total_production_quintals,
        crop_sales_revenue=req.crop_sales_revenue,
        expenses=req.expenses,
        subsidies_received=req.subsidies_received if req.subsidies_received is not None else 0.0
    )


@router.get("/demo")
def get_demo_carbon_audit():
    """
    Return comprehensive demo carbon credit and farm ledger audit.
    """
    audit = carbon_ledger_service.audit_carbon_credits(
        farmer_name="Shri Ramesh Patil",
        farm_id="MH-NSK-442",
        farm_size_hectares=4.2,
        adopted_practices=["residue_management", "biochar_application", "micro_irrigation", "solar_pump"]
    )
    financials = carbon_ledger_service.calculate_farm_financials(
        farm_size_acres=10.37,
        crop_type="Sugarcane / Maize",
        total_production_quintals=650.0,
        crop_sales_revenue=1950000.0,
        expenses={
            "seed_cane": 85000.0,
            "organic_biofertilizer": 62000.0,
            "drip_fertigation_inputs": 38000.0,
            "machinery_and_tractor": 75000.0,
            "harvesting_labor": 140000.0,
            "post_harvest_transport": 45000.0
        },
        subsidies_received=35000.0
    )
    return {
        "success": True,
        "carbon_audit": audit,
        "financial_ledger": financials
    }
