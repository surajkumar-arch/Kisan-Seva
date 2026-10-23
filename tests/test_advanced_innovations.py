"""Automated Verification Suite for Satellite Hydrology and Smart Tank-Mix Modules."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.satellite_moisture_service import (
    evaluate_field_hydrology,
    calculate_penman_monteith_et0,
)
from backend.app.services.tank_mix_service import (
    check_tank_mix_compatibility,
    AGROCHEMICAL_REGISTRY,
)

client = TestClient(app)


def test_penman_monteith_calculation():
    """Verify FAO-56 Penman-Monteith Evapotranspiration formula output."""
    et0 = calculate_penman_monteith_et0(temperature_c=28.0, humidity_pct=60.0, wind_kmh=10.0)
    assert 2.0 <= et0 <= 10.0, f"Unexpected ET0 value: {et0}"


def test_satellite_hydrology_evaluation():
    """Verify drought risk and irrigation prescription calculations."""
    res = evaluate_field_hydrology(
        crop="Cotton",
        soil_type="Black Soil",
        temperature_c=34.0,
        humidity_pct=40.0,
        rainfall_14d_mm=4.0,
        growth_stage="mid",
        land_area_acres=3.0,
    )
    assert res["status"] == "SUCCESS"
    assert "hydrology_metrics" in res
    assert 10.0 <= res["hydrology_metrics"]["drought_risk_index_score"] <= 95.0
    assert "irrigation_prescription" in res
    assert res["irrigation_prescription"]["total_field_volume_liters"] > 0
    assert len(res["heatmap_png_base64"]) > 500

    # Test acute drought condition on Sandy Soil (low AWC)
    acute_res = evaluate_field_hydrology(
        crop="Wheat",
        soil_type="Sandy Soil",
        temperature_c=38.0,
        humidity_pct=25.0,
        rainfall_14d_mm=0.0,
        growth_stage="mid",
        land_area_acres=2.0,
    )
    assert acute_res["hydrology_metrics"]["drought_risk_index_score"] >= 60.0
    assert acute_res["hydrology_metrics"]["drought_status"] == "CRITICAL_DEFICIT"


def test_satellite_api_endpoints():
    """Verify HTTP endpoints for satellite hydrology profiler."""
    # Demo profile
    resp_demo = client.get("/api/satellite/demo-profile?crop=Wheat")
    assert resp_demo.status_code == 200
    data = resp_demo.json()
    assert data["crop"] == "Wheat"
    assert "hydrology_metrics" in data

    # POST calculation
    payload = {
        "crop": "Tomato",
        "soil_type": "Alluvial Soil",
        "temperature_c": 29.0,
        "humidity_pct": 50.0,
        "rainfall_14d_mm": 10.0,
        "growth_stage": "mid",
        "land_area_acres": 1.5,
    }
    resp_post = client.post("/api/satellite/moisture-profile", json=payload)
    assert resp_post.status_code == 200
    assert resp_post.json()["status"] == "SUCCESS"


def test_tank_mix_incompatible_biocide():
    """Verify biological vs chemical biocide hazard detection."""
    # Mixing chemical fungicide with living biological agent
    res = check_tank_mix_compatibility(
        item_a="Copper Oxychloride (Fungicide)",
        item_b="Trichoderma viride (Bio-agent)",
        water_ph=7.2,
    )
    assert res["status"] == "SUCCESS"
    assert res["tank_compatibility"]["is_compatible"] is False
    assert any("BIOCIDE HAZARD" in c for c in res["tank_compatibility"]["conflicts"])


def test_tank_mix_compatible():
    """Verify safe mixing of compatible agrochemicals."""
    res = check_tank_mix_compatibility(
        item_a="Mancozeb 75% WP (Fungicide)",
        item_b="Imidacloprid 17.8% SL (Insecticide)",
        water_ph=6.5,
        wind_speed_kmh=8.0,
    )
    assert res["status"] == "SUCCESS"
    assert res["tank_compatibility"]["is_compatible"] is True
    assert res["tank_compatibility"]["overall_safety_rating"] == "SAFE TO MIX"
    assert res["compliance_and_safety"]["mandatory_phi_days"] == 14


def test_tank_mix_ph_buffering_and_drift():
    """Verify alkaline hydrolysis detection and spray drift warning."""
    res = check_tank_mix_compatibility(
        item_a="Chlorpyrifos 20% EC (Insecticide)",
        item_b="Mancozeb 75% WP (Fungicide)",
        water_ph=8.6,  # strongly alkaline
        wind_speed_kmh=18.0,  # strong wind
    )
    assert res["water_quality_buffering"]["buffering_needed"] is True
    assert "ALKALINE_HYDROLYSIS_RISK" in res["water_quality_buffering"]["ph_status"]
    assert res["spray_physics_and_drift"]["drift_category"] == "HIGH_DRIFT_DANGER"


def test_tank_mix_api_endpoints():
    """Verify FastAPI routes for tank-mix calculator."""
    resp_cat = client.get("/api/spray/catalog")
    assert resp_cat.status_code == 200
    assert resp_cat.json()["total_chemicals"] >= 8

    resp_demo = client.get("/api/spray/demo-check?scenario=incompatible")
    assert resp_demo.status_code == 200
    assert resp_demo.json()["tank_compatibility"]["is_compatible"] is False

    payload = {
        "chemical_a": "19:19:19 NPK Foliar (Micronutrient)",
        "chemical_b": "Mancozeb 75% WP (Fungicide)",
        "water_ph": 6.8,
        "wind_speed_kmh": 6.0,
        "temperature_c": 26.0,
        "humidity_pct": 70.0,
        "tank_capacity_liters": 15.0,
    }
    resp_post = client.post("/api/spray/tank-mix-check", json=payload)
    assert resp_post.status_code == 200
    assert resp_post.json()["status"] == "SUCCESS"
