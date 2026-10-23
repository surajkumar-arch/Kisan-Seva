"""Action Advisor: turns a crop choice + soil values into money terms.

Outputs a fertilizer plan (Urea / DAP / MOP quantities and cost), an expected
yield, and an indicative revenue figure.

Important honesty notes
-----------------------
* Fertilizer and MSP prices below are *reference defaults* and are overridable
  with environment variables. They are not live quotes; the API returns
  ``price_basis`` so clients can show that to the user.
* Live market price is fetched only when ``DATA_GOV_IN_API_KEY`` is configured
  (Agmarknet via data.gov.in). Any failure falls back to the MSP reference.
* Yield uses ``base_yield_tons_ha`` from the crop catalog, which is an average
  and is not adjusted for the individual farm.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

from backend.app.models.crop_net import CROP_CATALOG

ACRE_TO_HECTARE = 0.404686

# Nutrient content (fraction by weight).
UREA_N = 0.46
DAP_N = 0.18
DAP_P2O5 = 0.46
MOP_K2O = 0.60


def _price(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def fertilizer_prices() -> Dict[str, float]:
    """INR per kg (reference defaults, overridable)."""
    return {
        "urea": _price("PRICE_UREA_INR_PER_KG", 5.4),
        "dap": _price("PRICE_DAP_INR_PER_KG", 27.0),
        "mop": _price("PRICE_MOP_INR_PER_KG", 34.0),
    }


# Reference MSP in INR per quintal (2024-25 marketing season). ``None`` means
# the crop has no central MSP; sugarcane uses the FRP. Verify before relying.
MSP_INR_PER_QUINTAL: Dict[str, Optional[float]] = {
    "Rice": 2300.0,
    "Wheat": 2275.0,
    "Cotton": 7121.0,
    "Maize": 2225.0,
    "Chickpea": 5440.0,
    "Kidney Beans": None,
    "Pigeon Peas": 7550.0,
    "Moth Beans": 8682.0,
    "Groundnut": 6783.0,
    "Sugarcane": 340.0,
    "Coffee": None,
    "Tea": None,
}

AGMARKNET_COMMODITY = {
    "Rice": "Paddy(Dhan)(Common)", "Wheat": "Wheat", "Cotton": "Cotton", "Maize": "Maize",
    "Chickpea": "Bengal Gram(Gram)(Whole)", "Pigeon Peas": "Arhar (Tur/Red Gram)(Whole)",
    "Groundnut": "Groundnut", "Sugarcane": "Sugarcane",
}


def _mid(rng: Any, fallback: float) -> float:
    try:
        return (float(rng[0]) + float(rng[1])) / 2.0
    except Exception:
        return fallback


def live_mandi_price(crop: str, state: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Latest modal price (INR/quintal) from data.gov.in Agmarknet, if configured."""
    api_key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
    commodity = AGMARKNET_COMMODITY.get(crop)
    if not api_key or not commodity:
        return None
    params = {
        "api-key": api_key, "format": "json", "limit": 20,
        "filters[commodity]": commodity,
    }
    if state:
        params["filters[state]"] = state
    try:
        resp = requests.get(
            "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070",
            params=params, timeout=6,
        )
        if resp.status_code != 200:
            return None
        records = resp.json().get("records", [])
        prices = [float(r["modal_price"]) for r in records if r.get("modal_price")]
        if not prices:
            return None
        return {
            "modal_price_inr_per_quintal": round(sum(prices) / len(prices), 0),
            "markets_sampled": len(prices),
            "source": "Agmarknet via data.gov.in (live)",
        }
    except Exception:
        return None


def build_fertilizer_plan(
    crop: str,
    n: Optional[float],
    p: Optional[float],
    k: Optional[float],
    area_acres: float,
) -> Dict[str, Any]:
    """Compute fertilizer needed to lift soil N-P-K to the crop's mid-optimal level."""
    profile = CROP_CATALOG[crop]
    target = {
        "N": _mid(profile.get("optimal_n"), 120.0),
        "P": _mid(profile.get("optimal_p"), 40.0),
        "K": _mid(profile.get("optimal_k"), 100.0),
    }
    current = {"N": n, "P": p, "K": k}
    # Missing readings are treated conservatively as "unknown" -> no deficit is assumed.
    deficit = {
        key: (max(0.0, target[key] - float(current[key])) if current[key] is not None else 0.0)
        for key in target
    }
    unknown = [key for key, val in current.items() if val is None]

    area_ha = area_acres * ACRE_TO_HECTARE
    # Phosphorus from DAP (also supplies some N), potash from MOP, remaining N from urea.
    dap_per_ha = deficit["P"] / DAP_P2O5 if deficit["P"] else 0.0
    n_from_dap = dap_per_ha * DAP_N
    urea_per_ha = max(0.0, deficit["N"] - n_from_dap) / UREA_N if deficit["N"] else 0.0
    mop_per_ha = deficit["K"] / MOP_K2O if deficit["K"] else 0.0

    prices = fertilizer_prices()
    kg = {
        "urea": round(urea_per_ha * area_ha, 1),
        "dap": round(dap_per_ha * area_ha, 1),
        "mop": round(mop_per_ha * area_ha, 1),
    }
    cost = {name: round(qty * prices[name], 0) for name, qty in kg.items()}
    return {
        "targets_kg_per_ha": {k_: round(v, 1) for k_, v in target.items()},
        "deficit_kg_per_ha": {k_: round(v, 1) for k_, v in deficit.items()},
        "unknown_nutrients": unknown,
        "fertilizer_kg": kg,
        "fertilizer_cost_inr": cost,
        "total_fertilizer_cost_inr": round(sum(cost.values()), 0),
        "prices_inr_per_kg": prices,
    }


def build_economics(
    crop: str,
    area_acres: float,
    n: Optional[float] = None,
    p: Optional[float] = None,
    k: Optional[float] = None,
    state: Optional[str] = None,
) -> Dict[str, Any]:
    """Full advisor payload for one crop."""
    if crop not in CROP_CATALOG:
        raise KeyError(crop)
    if area_acres <= 0:
        raise ValueError("area_acres must be positive")

    profile = CROP_CATALOG[crop]
    plan = build_fertilizer_plan(crop, n, p, k, area_acres)

    area_ha = area_acres * ACRE_TO_HECTARE
    yield_t_ha = float(profile.get("base_yield_tons_ha", 0.0))
    expected_tons = round(yield_t_ha * area_ha, 2)
    quintals = expected_tons * 10.0

    live = live_mandi_price(crop, state)
    msp = MSP_INR_PER_QUINTAL.get(crop)
    if live:
        price, basis = live["modal_price_inr_per_quintal"], live["source"]
    elif msp is not None:
        price, basis = msp, "Reference MSP/FRP 2024-25 (verify current rate)"
    else:
        price, basis = None, "No MSP for this crop; enter a local market price"

    revenue = round(quintals * price, 0) if price else None
    margin = round(revenue - plan["total_fertilizer_cost_inr"], 0) if revenue is not None else None

    return {
        "crop": crop,
        "area_acres": area_acres,
        "area_hectares": round(area_ha, 3),
        "duration": profile.get("duration_days"),
        "water_requirement": profile.get("water_requirement"),
        "expected_yield_tons": expected_tons,
        "yield_basis": "Catalog average yield; actual yield varies with farm conditions",
        "price_inr_per_quintal": price,
        "price_basis": basis,
        "live_market": live,
        "expected_gross_revenue_inr": revenue,
        "fertilizer": plan,
        "margin_after_fertilizer_inr": margin,
        "margin_note": "Excludes seed, labour, irrigation, pesticide and other costs.",
        "cultivation_tips": list(profile.get("cultivation_tips", [])),
    }


def compare_crops(
    crops: List[str], area_acres: float, n: Optional[float], p: Optional[float], k: Optional[float],
    state: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows = []
    for c in crops:
        if c in CROP_CATALOG:
            e = build_economics(c, area_acres, n, p, k, state)
            rows.append(
                {
                    "crop": c,
                    "expected_gross_revenue_inr": e["expected_gross_revenue_inr"],
                    "fertilizer_cost_inr": e["fertilizer"]["total_fertilizer_cost_inr"],
                    "margin_after_fertilizer_inr": e["margin_after_fertilizer_inr"],
                    "price_basis": e["price_basis"],
                }
            )
    rows.sort(key=lambda r: (r["margin_after_fertilizer_inr"] is None, -(r["margin_after_fertilizer_inr"] or 0)))
    return rows
