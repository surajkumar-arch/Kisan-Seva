"""Automated Verification Suite for Night Innovations:
1. Crop Yield & Biomass Forecasting (WOFOST/DSSAT Monte Carlo)
2. Smart Mandi Price Arbitrage & Storage Economics
3. Kisan Vaani Vernacular Agro-Voice Engine
4. PMFBY Smart Crop Insurance Claim Assessor
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.yield_forecast_service import simulate_crop_growth
from backend.app.services.mandi_arbitrage_service import calculate_mandi_arbitrage
from backend.app.services.vernacular_voice_service import process_vernacular_query
from backend.app.services.insurance_claim_service import assess_pmfby_crop_claim

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. CROP YIELD & BIOMASS FORECASTING TESTS
# ---------------------------------------------------------------------------

def test_simulate_crop_growth_service():
    """Verify biophysical growth simulation and Monte Carlo confidence bounds."""
    res = simulate_crop_growth(
        crop="Wheat",
        soil_type="Alluvial Soil",
        land_area_acres=3.0,
        soil_nitrogen_kgha=120.0,
        soil_phosphorus_kgha=35.0,
        soil_potassium_kgha=110.0,
        soil_ph=6.8,
        avg_temperature_c=25.0,
        cumulative_rainfall_mm=300.0,
        disease_severity_loss_pct=10.0,
        num_simulations=500,
    )
    assert res["status"] == "SUCCESS"
    assert "yield_forecast" in res
    assert res["yield_forecast"]["expected_yield_quintals_acre"] > 5.0
    assert res["yield_forecast"]["total_harvest_expected_quintals"] > 0
    bounds = res["monte_carlo_risk_bounds"]
    assert bounds["p10_pessimistic_quintals_acre"] <= bounds["p50_median_quintals_acre"] <= bounds["p90_optimistic_quintals_acre"]
    assert len(res["biomass_progression_curve"]) > 5


def test_yield_forecast_endpoints():
    """Verify HTTP endpoints for crop yield forecasting."""
    resp_demo = client.get("/api/yield/demo?crop=Wheat")
    assert resp_demo.status_code == 200
    assert resp_demo.json()["status"] == "SUCCESS"

    payload = {
        "crop": "Cotton",
        "soil_type": "Black Soil",
        "land_area_acres": 2.0,
        "soil_nitrogen_kgha": 110.0,
        "soil_phosphorus_kgha": 30.0,
        "soil_potassium_kgha": 95.0,
        "soil_ph": 7.4,
        "avg_temperature_c": 30.0,
        "cumulative_rainfall_mm": 450.0,
        "disease_severity_loss_pct": 0.0,
        "num_simulations": 300,
    }
    resp_post = client.post("/api/yield/forecast", json=payload)
    assert resp_post.status_code == 200
    assert resp_post.json()["crop"] == "Cotton"


# ---------------------------------------------------------------------------
# 2. MANDI PRICE ARBITRAGE & STORAGE ECONOMICS TESTS
# ---------------------------------------------------------------------------

def test_mandi_arbitrage_service():
    """Verify multi-mandi freight economics and warehouse holding curve."""
    res = calculate_mandi_arbitrage(
        crop="Wheat",
        quantity_quintals=80.0,
        local_mandi_rate_inr=2300.0,
        diesel_price_per_liter=90.0,
    )
    assert res["status"] == "SUCCESS"
    assert len(res["mandi_arbitrage_matrix"]) >= 5
    assert len(res["storage_economics_projections"]) == 3
    assert "arbitrage_verdict" in res
    assert res["arbitrage_verdict"]["ruling"] is not None


def test_mandi_arbitrage_endpoints():
    """Verify HTTP endpoints for mandi arbitrage."""
    resp_hubs = client.get("/api/mandi/hubs")
    assert resp_hubs.status_code == 200
    assert len(resp_hubs.json()["mandi_hubs"]) >= 5

    resp_demo = client.get("/api/mandi/demo-arbitrage?crop=Wheat")
    assert resp_demo.status_code == 200
    assert resp_demo.json()["status"] == "SUCCESS"

    payload = {
        "crop": "Mustard",
        "quantity_quintals": 50.0,
        "local_mandi_rate_inr": 5400.0,
        "diesel_price_per_liter": 91.0,
        "truck_mileage_km_per_liter": 4.0,
        "truck_capacity_quintals": 100.0,
        "warehouse_storage_months": 2,
        "storage_rent_per_quintal_month": 25.0,
    }
    resp_post = client.post("/api/mandi/arbitrage-calculator", json=payload)
    assert resp_post.status_code == 200
    assert resp_post.json()["crop"] == "Mustard"


# ---------------------------------------------------------------------------
# 3. KISAN VAANI VERNACULAR AGRO-VOICE TESTS
# ---------------------------------------------------------------------------

def test_vernacular_voice_service():
    """Verify Hindi intent parsing and synthetic audio envelope generation."""
    res = process_vernacular_query("पत्तियों में पीलापन आ रहा है क्या खाद डालें?")
    assert res["status"] == "SUCCESS"
    assert res["matched_intent"] == "FOLIAR_CHLOROSIS_DEFICIENCY"
    assert len(res["vernacular_card"]["prescribed_steps_hi"]) == 3
    assert len(res["audio_chime_wav_base64"]) > 100


def test_vernacular_voice_endpoints():
    """Verify HTTP routes for Kisan Vaani."""
    resp_prompts = client.get("/api/vaani/sample-prompts")
    assert resp_prompts.status_code == 200
    assert len(resp_prompts.json()["prompts"]) >= 4

    resp_demo = client.get("/api/vaani/demo-vaani?topic=pest")
    assert resp_demo.status_code == 200
    assert resp_demo.json()["matched_intent"] == "INSECT_PEST_INFESTATION"

    payload = {
        "query_text": "यूरिया और डीएपी खाद कब डालना चाहिए?",
        "preferred_dialect": "Hindi",
    }
    resp_post = client.post("/api/vaani/vernacular-query", json=payload)
    assert resp_post.status_code == 200
    assert resp_post.json()["matched_intent"] == "BALANCED_FERTILIZER_TIMING"


# ---------------------------------------------------------------------------
# 4. PMFBY CROP INSURANCE CLAIM ASSESSOR TESTS
# ---------------------------------------------------------------------------

def test_insurance_claim_service():
    """Verify PMFBY shortfall yield deduction, indemnity calculation, and SHA-256 fingerprint."""
    res = assess_pmfby_crop_claim(
        farmer_name="Suraj Kumar",
        khasra_survey_no="KH-108/3B",
        district_state="Ludhiana, Punjab",
        crop="Wheat",
        insured_acres=3.0,
        loss_cause="Unseasonal Hailstorm",
        visual_foliar_damage_pct=50.0,
        assessed_actual_yield_q_acre=8.0,
        gps_lat=30.9010,
        gps_lon=75.8573,
    )
    assert res["status"] == "SUCCESS"
    assert "PMFBY-" in res["claim_registration_id"]
    loss = res["actuarial_loss_assessment"]
    assert loss["crop_shortfall_percentage"] > 40.0
    fin = res["settlement_financials"]
    assert fin["approved_indemnity_claim_payout_inr"] > 0
    assert len(res["statutory_verification"]["evidence_cryptographic_sha256"]) == 64


def test_insurance_endpoints():
    """Verify HTTP endpoints for PMFBY insurance assessor."""
    resp_scales = client.get("/api/insurance/actuarial-scales")
    assert resp_scales.status_code == 200
    assert "benchmarks" in resp_scales.json()

    resp_demo = client.get("/api/insurance/demo-claim?crop=Wheat")
    assert resp_demo.status_code == 200
    assert resp_demo.json()["status"] == "SUCCESS"

    payload = {
        "farmer_name": "Ramesh Singh",
        "khasra_survey_no=":"KH-501/2",
        "district_state": "Bathinda, Punjab",
        "crop": "Cotton",
        "insured_acres": 2.5,
        "loss_cause": "Pink Bollworm Infestation",
        "visual_foliar_damage_pct": 60.0,
        "satellite_drought_deficit_pct": 0.0,
        "assessed_actual_yield_q_acre": 4.0,
        "gps_lat": 30.2110,
        "gps_lon": 74.9455,
    }
    # Note: fix typo in key if needed before request
    payload_clean = {
        "farmer_name": "Ramesh Singh",
        "khasra_survey_no": "KH-501/2",
        "district_state": "Bathinda, Punjab",
        "crop": "Cotton",
        "insured_acres": 2.5,
        "loss_cause": "Pink Bollworm Infestation",
        "visual_foliar_damage_pct": 60.0,
        "satellite_drought_deficit_pct": 0.0,
        "assessed_actual_yield_q_acre": 4.0,
        "gps_lat": 30.2110,
        "gps_lon": 74.9455,
    }
    resp_post = client.post("/api/insurance/assess-claim", json=payload_clean)
    assert resp_post.status_code == 200
    assert resp_post.json()["crop_sown" if "crop_sown" in resp_post.json() else "beneficiary_details"]["crop_sown"] == "Cotton"
