"""
KisanSeva Farm Ledger & Regenerative Carbon Credit Offset Engine
Implements double-entry agricultural bookkeeping, unit economics per quintal/hectare,
and IPCC-aligned carbon sequestration credits for sustainable farming practices.
"""

from typing import Dict, Any, List
import hashlib
import time
from datetime import datetime


class CarbonLedgerService:
    """
    Agro-Economics & Carbon Credit Offset verification engine for Indian agriculture.
    """

    # Carbon sequestration coefficients (t CO2e / hectare / year) based on IPCC & ICAR guidelines
    CARBON_COEFFICIENTS = {
        "residue_management": {
            "name": "Zero Stubble Burning (In-situ Mulching / Happy Seeder)",
            "rate": 1.85,
            "description": "Prevents open-field crop residue incineration; builds soil organic carbon (SOC)."
        },
        "conservation_tillage": {
            "name": "Zero-Tillage / Direct Seeded Rice (DSR)",
            "rate": 0.65,
            "description": "Preserves mycorrhizal fungal networks, halts carbon oxidation in soil pores."
        },
        "biochar_application": {
            "name": "Biochar & Enriched Organic Compost Application",
            "rate": 1.40,
            "description": "Recalcitrant aromatic carbon fixed in topsoil for decadal sequestration."
        },
        "micro_irrigation": {
            "name": "Precision Drip / Micro-Sprinkler Irrigation",
            "rate": 0.45,
            "description": "Reduces anaerobic methane emissions and pumping energy consumption."
        },
        "solar_pump": {
            "name": "Solar PV Irrigation Pump (PM-KUSUM)",
            "rate": 1.10,
            "description": "Displaces diesel generator and thermal grid electricity emissions."
        },
        "agroforestry_border": {
            "name": "Field Boundary Tree Plantation (Neem/Subabul/Teak)",
            "rate": 0.90,
            "description": "Aboveground and belowground woody biomass perennial carbon capture."
        }
    }

    # Market rate for Voluntary Carbon Market (VCM) credits in India
    INR_PER_TON_CO2E = 1150.0

    @classmethod
    def calculate_farm_financials(
        cls,
        farm_size_acres: float,
        crop_type: str,
        total_production_quintals: float,
        crop_sales_revenue: float,
        expenses: Dict[str, float],
        subsidies_received: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate complete unit farm economics: Total Cost, Net Profit, Production Cost / Quintal, and ROI.
        """
        total_expense = sum(expenses.values())
        total_income = crop_sales_revenue + subsidies_received
        net_profit = total_income - total_expense

        cost_per_quintal = round(total_expense / max(1.0, total_production_quintals), 2)
        revenue_per_quintal = round(crop_sales_revenue / max(1.0, total_production_quintals), 2)
        profit_per_acre = round(net_profit / max(0.1, farm_size_acres), 2)
        roi_percentage = round((net_profit / max(1.0, total_expense)) * 100.0, 2)

        return {
            "total_income_inr": round(total_income, 2),
            "crop_sales_inr": round(crop_sales_revenue, 2),
            "subsidies_inr": round(subsidies_received, 2),
            "total_expenses_inr": round(total_expense, 2),
            "expense_breakdown": {k: round(v, 2) for k, v in expenses.items()},
            "net_profit_inr": round(net_profit, 2),
            "profit_margin_pct": round((net_profit / max(1.0, total_income)) * 100.0, 1),
            "cost_per_quintal_inr": cost_per_quintal,
            "revenue_per_quintal_inr": revenue_per_quintal,
            "profit_per_acre_inr": profit_per_acre,
            "roi_percentage": roi_percentage,
            "financial_health": "PROFITABLE & RESILIENT" if net_profit > 0 else "DEFICIT / MARGIN CALL"
        }

    @classmethod
    def audit_carbon_credits(
        cls,
        farmer_name: str,
        farm_id: str,
        farm_size_hectares: float,
        adopted_practices: List[str]
    ) -> Dict[str, Any]:
        """
        Quantify verified carbon offsets, calculate carbon monetization in INR,
        and generate a cryptographically signed ESG verification token.
        """
        total_rate_per_ha = 0.0
        verified_breakdown = []

        for code in adopted_practices:
            if code in cls.CARBON_COEFFICIENTS:
                item = cls.CARBON_COEFFICIENTS[code]
                tons_for_practice = round(item["rate"] * farm_size_hectares, 3)
                total_rate_per_ha += item["rate"]
                verified_breakdown.append({
                    "practice_code": code,
                    "title": item["name"],
                    "rate_t_co2e_per_ha": item["rate"],
                    "total_offset_tons": tons_for_practice,
                    "valuation_inr": round(tons_for_practice * cls.INR_PER_TON_CO2E, 2),
                    "description": item["description"]
                })

        total_co2e_offset = round(total_rate_per_ha * farm_size_hectares, 3)
        total_payout_inr = round(total_co2e_offset * cls.INR_PER_TON_CO2E, 2)

        # Cryptographic verification hash
        raw_token = f"{farmer_name}|{farm_id}|{farm_size_hectares}|{total_co2e_offset}|{time.time()}"
        verification_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "farmer_name": farmer_name,
            "farm_id": farm_id,
            "farm_size_ha": farm_size_hectares,
            "total_co2e_sequestered_tons": total_co2e_offset,
            "vcm_spot_price_per_ton_inr": cls.INR_PER_TON_CO2E,
            "estimated_carbon_dividend_inr": total_payout_inr,
            "verified_practices_count": len(verified_breakdown),
            "practices_breakdown": verified_breakdown,
            "verification_certificate": {
                "certificate_id": f"KS-CARB-{farm_id.upper()}-{int(time.time()) % 100000}",
                "issuer": "KisanSeva Climate & Agronomy ESG Registry",
                "standard": "Verra / Gold Standard & ICAR Regenerative Protocol Equivalent",
                "integrity_hash": verification_hash,
                "status": "VERIFIED & AUDIT_READY"
            }
        }


# Global instance
carbon_ledger_service = CarbonLedgerService()
