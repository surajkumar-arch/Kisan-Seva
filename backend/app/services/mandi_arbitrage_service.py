"""Smart Mandi Real-Time Price Arbitrage & Optimal Selling Window Predictor.

Evaluates inter-mandi price spreads, freight logistics costs (diesel & tolls),
warehouse storage economics, and future price seasonal curves to advise farmers
on the highest net-margin sales strategy.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# Benchmark Government MSP (Minimum Support Price) for Reference (₹ / Quintal)
GOVT_MSP_TABLE = {
    "Wheat": 2275.0,
    "Rice": 2203.0,
    "Cotton": 7020.0,
    "Maize": 2090.0,
    "Mustard": 5650.0,
    "Soybean": 4600.0,
    "Groundnut": 6377.0,
    "Sugarcane": 315.0,
    "Potato": 1250.0,
    "Tomato": 1400.0,
}

# Major Agricultural Mandis Catalog with Coordinates and Market Premiums
APMC_MANDI_HUBS = [
    {"name": "Local District Mandi", "distance_km": 15, "buyer_density": "Moderate", "handling_charge_per_q": 45.0},
    {"name": "Khanna Grain Market (Punjab)", "distance_km": 95, "buyer_density": "Very High", "handling_charge_per_q": 35.0},
    {"name": "Neemuch Mandi (Madhya Pradesh)", "distance_km": 210, "buyer_density": "Very High", "handling_charge_per_q": 30.0},
    {"name": "Unjha APMC (Gujarat)", "distance_km": 340, "buyer_density": "Very High (Export Hub)", "handling_charge_per_q": 40.0},
    {"name": "Indore Agri Terminal (MP)", "distance_km": 180, "buyer_density": "High", "handling_charge_per_q": 35.0},
    {"name": "Guntur Mirchi & Grain Yard (AP)", "distance_km": 420, "buyer_density": "Export Mega Hub", "handling_charge_per_q": 50.0},
    {"name": "Hapur Mandi (Uttar Pradesh)", "distance_km": 120, "buyer_density": "High", "handling_charge_per_q": 38.0},
]


def calculate_mandi_arbitrage(
    crop: str = "Wheat",
    quantity_quintals: float = 60.0,
    local_mandi_rate_inr: float = 2320.0,
    diesel_price_per_liter: float = 90.0,
    truck_mileage_km_per_liter: float = 4.0,
    truck_capacity_quintals: float = 100.0,
    warehouse_storage_months: int = 2,
    storage_rent_per_quintal_month: float = 25.0,
) -> Dict[str, Any]:
    """Calculates inter-mandi transport logistics arbitrage and warehouse holding return."""
    msp = GOVT_MSP_TABLE.get(crop, 2200.0)

    # Simulate price dynamics across regional hubs
    arbitrage_opportunities: List[Dict[str, Any]] = []

    # Freight cost formula: (Distance * 2 roundtrip / mileage) * diesel_price / min(truck_capacity, quantity)
    for m in APMC_MANDI_HUBS:
        dist = m["distance_km"]
        # Regional supply-demand premium simulation
        if dist <= 20:
            spot_rate = local_mandi_rate_inr
        else:
            # Regional hubs command a 4% to 15% liquidity premium
            premium_pct = 0.04 + (math.log(dist + 10) * 0.015)
            spot_rate = round(local_mandi_rate_inr * (1.0 + premium_pct), 1)

        # Diesel + driver wage freight allocation per quintal
        trips = math.ceil(quantity_quintals / truck_capacity_quintals)
        total_diesel_liters = (dist * 2.0 / truck_mileage_km_per_liter) * trips
        fuel_cost = total_diesel_liters * diesel_price_per_liter
        toll_and_driver = (dist * 4.0) * trips  # approx ₹4/km toll + driver
        total_transport_cost = fuel_cost + toll_and_driver
        transport_cost_per_q = round(total_transport_cost / max(1.0, quantity_quintals), 1)

        handling_per_q = m["handling_charge_per_q"]
        net_effective_price_per_q = round(spot_rate - transport_cost_per_q - handling_per_q, 1)
        gross_revenue = round(spot_rate * quantity_quintals, 0)
        net_payout = round(net_effective_price_per_q * quantity_quintals, 0)

        price_spread_vs_local = round(net_effective_price_per_q - local_mandi_rate_inr, 1)
        net_profit_gain = round(price_spread_vs_local * quantity_quintals, 0)

        arbitrage_opportunities.append({
            "mandi_name": m["name"],
            "distance_km": dist,
            "spot_rate_per_quintal": spot_rate,
            "transport_cost_per_quintal": transport_cost_per_q,
            "handling_fee_per_quintal": handling_per_q,
            "net_effective_price_per_quintal": net_effective_price_per_q,
            "gross_revenue_inr": gross_revenue,
            "net_farmer_payout_inr": net_payout,
            "net_profit_gain_vs_local_inr": net_profit_gain,
            "arbitrage_viable": net_profit_gain > 1500.0,
        })

    # Sort opportunities by highest net farmer payout
    arbitrage_opportunities.sort(key=lambda x: x["net_farmer_payout_inr"], reverse=True)
    best_hub = arbitrage_opportunities[0]

    # -------------------------------------------------------------
    # 2. Warehouse Storage Economics & Holding Window Analysis
    # -------------------------------------------------------------
    # Post-harvest glut typically depresses prices; off-season (60-90 days) sees 8-18% rise
    historical_appreciation_pct_30d = 0.05
    historical_appreciation_pct_60d = 0.12
    historical_appreciation_pct_90d = 0.19

    # Holding costs
    monthly_storage_cost_per_q = storage_rent_per_quintal_month
    spoilage_shrinkage_loss_pct_month = 0.012  # 1.2% per month
    cost_of_capital_interest_pct_month = 0.006  # ~7.2% annual crop loan interest

    holding_projections: List[Dict[str, Any]] = []
    horizons = [
        {"months": 1, "appreciation": historical_appreciation_pct_30d, "label": "30 Days (Early Buffer)"},
        {"months": 2, "appreciation": historical_appreciation_pct_60d, "label": "60 Days (Post-Glut Rebound)"},
        {"months": 3, "appreciation": historical_appreciation_pct_90d, "label": "90 Days (Off-Season Peak)"},
    ]

    base_local_net = local_mandi_rate_inr * quantity_quintals

    for h in horizons:
        m_count = h["months"]
        projected_rate = round(local_mandi_rate_inr * (1.0 + h["appreciation"]), 1)
        total_rent = monthly_storage_cost_per_q * m_count * quantity_quintals
        spoilage_loss_inr = (spoilage_shrinkage_loss_pct_month * m_count) * (projected_rate * quantity_quintals)
        interest_cost_inr = (cost_of_capital_interest_pct_month * m_count) * base_local_net

        total_carrying_costs = round(total_rent + spoilage_loss_inr + interest_cost_inr, 0)
        projected_gross = round(projected_rate * quantity_quintals * (1.0 - (spoilage_shrinkage_loss_pct_month * m_count)), 0)
        net_after_storage = round(projected_gross - total_rent - interest_cost_inr, 0)
        net_gain_vs_today = round(net_after_storage - base_local_net, 0)

        holding_projections.append({
            "horizon_label": h["label"],
            "holding_period_months": m_count,
            "projected_mandi_rate_per_q": projected_rate,
            "total_storage_and_carrying_cost_inr": total_carrying_costs,
            "net_revenue_after_carrying_inr": net_after_storage,
            "net_profit_vs_selling_today_inr": net_gain_vs_today,
            "recommendation": "PROFITABLE TO HOLD" if net_gain_vs_today > 2500.0 else "SELL TODAY",
        })

    # Final Strategic Ruling
    if best_hub["net_profit_gain_vs_local_inr"] > 4000.0:
        verdict = f"DISPATCH TO {best_hub['mandi_name'].upper()}"
        verdict_badge = "🚚 Regional Arbitrage Dispatch"
        strategy_desc = (
            f"Net gain of ₹{best_hub['net_profit_gain_vs_local_inr']:,.0f} by transporting {quantity_quintals} quintals "
            f"to {best_hub['mandi_name']} ({best_hub['distance_km']} km) even after covering all diesel and handling expenses."
        )
    elif holding_projections[1]["net_profit_vs_selling_today_inr"] > 3500.0:
        verdict = "STORE IN WAREHOUSE (60 DAYS)"
        verdict_badge = "⏳ Hold in Certified Warehouse"
        strategy_desc = (
            f"Prices are currently depressed by harvest glut. Storing for 60 days in a WDRA warehouse yields "
            f"an estimated net profit of ₹{holding_projections[1]['net_profit_vs_selling_today_inr']:,.0f} after carrying costs."
        )
    else:
        verdict = "SELL LOCALLY TODAY"
        verdict_badge = "💰 Immediate Local Liquidation"
        strategy_desc = "Transport freight or storage carry costs negate distant price gains. Best net margin is realized at your local district mandi today."

    return {
        "status": "SUCCESS",
        "crop": crop,
        "quantity_quintals": quantity_quintals,
        "msp_benchmark_inr": msp,
        "current_local_rate_inr": local_mandi_rate_inr,
        "arbitrage_verdict": {
            "ruling": verdict,
            "badge": verdict_badge,
            "action_strategy": strategy_desc,
            "best_destination_mandi": best_hub["mandi_name"],
            "max_arbitrage_gain_inr": best_hub["net_profit_gain_vs_local_inr"],
        },
        "mandi_arbitrage_matrix": arbitrage_opportunities,
        "storage_economics_projections": holding_projections,
    }
