"""
Automated Verification Suite for Deep-Tech Agronomic Innovations:
1. Microclimate & Spray Early Warning (Penman-Monteith, VPD, Frost/Heatwave Radar)
2. Farm Ledger & Regenerative Carbon Credit Offset Engine (IPCC Protocols & SHA-256 Certificates)
3. Precision Smart Irrigation & Solar Pump Digital Twin (Crop ETc, Soil Deficit, IoT MQTT)
4. Bio-Surveillance & Epidemic Swarm Early Warning (R0 Spread Dynamics, Wind Corridor Trajectory)
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.microclimate_service import microclimate_service
from backend.app.services.carbon_ledger_service import carbon_ledger_service
from backend.app.services.precision_irrigation_service import precision_irrigation_service
from backend.app.services.epidemic_swarm_service import epidemic_swarm_service

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. MICROCLIMATE & SPRAY EARLY WARNING TESTS
# ---------------------------------------------------------------------------

def test_microclimate_service_direct():
    """Verify VPD, ET0, spray suitability, and extreme weather alerts."""
    res = microclimate_service.analyze_microclimate(
        farm_name="Test Punjab Farm",
        location="Amritsar, Punjab",
        crop_type="Wheat",
        temp_c=25.0,
        temp_max_c=32.0,
        temp_min_c=3.5,  # Should trigger frost warning
        relative_humidity=65.0,
        wind_kmh=12.0,
        soil_temp_c=21.0,
        rain_probability_pct=10.0
    )
    assert res["success"] is True
    assert "microclimate_metrics" in res
    assert res["microclimate_metrics"]["vapor_pressure_deficit_kpa"] > 0
    assert res["microclimate_metrics"]["reference_et0_mm_day"] > 0
    assert "spray_window" in res
    assert len(res["hourly_spray_timeline"]) == 12
    # Verify frost warning was triggered due to temp_min_c <= 4.0
    alert_types = [a["type"] for a in res["early_warning_alerts"]]
    assert "Radiation Frost Early Warning" in alert_types


def test_microclimate_endpoints():
    """Verify HTTP GET /demo and POST /analyze endpoints for microclimate."""
    resp_demo = client.get("/api/microclimate/demo")
    assert resp_demo.status_code == 200
    assert resp_demo.json()["success"] is True

    payload = {
        "farm_name": "Godavari Orchards",
        "location": "Nashik",
        "crop_type": "Grapes",
        "temp_c": 39.5,
        "temp_max_c": 42.0,  # Heatwave
        "temp_min_c": 22.0,
        "relative_humidity": 45.0,
        "wind_kmh": 22.0,  # Excessive wind
        "soil_temp_c": 28.0,
        "rain_probability_pct": 5.0
    }
    resp_post = client.post("/api/microclimate/analyze", json=payload)
    assert resp_post.status_code == 200
    data = resp_post.json()
    assert data["success"] is True
    assert data["spray_window"]["status"] == "HAZARDOUS"
    assert data["spray_window"]["can_spray"] is False


# ---------------------------------------------------------------------------
# 2. FARM LEDGER & CARBON CREDIT OFFSET TESTS
# ---------------------------------------------------------------------------

def test_carbon_ledger_service_direct():
    """Verify double-entry economics and verified carbon credit computation."""
    fin = carbon_ledger_service.calculate_farm_financials(
        farm_size_acres=5.0,
        crop_type="Wheat",
        total_production_quintals=100.0,
        crop_sales_revenue=230000.0,
        expenses={"seeds": 10000.0, "fertilizers": 15000.0, "labor": 25000.0},
        subsidies_received=10000.0
    )
    assert fin["net_profit_inr"] == 190000.0
    assert fin["cost_per_quintal_inr"] == 500.0
    assert fin["financial_health"] == "PROFITABLE & RESILIENT"

    audit = carbon_ledger_service.audit_carbon_credits(
        farmer_name="Shri Rajesh Sharma",
        farm_id="PB-LUD-101",
        farm_size_hectares=2.5,
        adopted_practices=["residue_management", "conservation_tillage", "solar_pump"]
    )
    assert audit["success"] is True
    assert audit["total_co2e_sequestered_tons"] > 0
    assert audit["estimated_carbon_dividend_inr"] > 0
    assert "verification_certificate" in audit
    assert len(audit["verification_certificate"]["integrity_hash"]) == 64  # SHA-256


def test_carbon_ledger_endpoints():
    """Verify HTTP GET /practices, GET /demo, POST /audit, and POST /farm-financials."""
    resp_practices = client.get("/api/carbon/practices")
    assert resp_practices.status_code == 200
    assert "practices" in resp_practices.json()

    resp_demo = client.get("/api/carbon/demo")
    assert resp_demo.status_code == 200
    assert resp_demo.json()["success"] is True

    audit_payload = {
        "farmer_name": "Kavita Devi",
        "farm_id": "UP-VAR-502",
        "farm_size_hectares": 1.8,
        "adopted_practices": ["biochar_application", "micro_irrigation"]
    }
    resp_audit = client.post("/api/carbon/audit", json=audit_payload)
    assert resp_audit.status_code == 200
    assert resp_audit.json()["verified_practices_count"] == 2


# ---------------------------------------------------------------------------
# 3. PRECISION SMART IRRIGATION & SOLAR TWIN TESTS
# ---------------------------------------------------------------------------

def test_precision_irrigation_service_direct():
    """Verify crop water deficit, irrigation scheduling, and IoT MQTT packet generation."""
    res = precision_irrigation_service.calculate_irrigation_schedule(
        farm_id="TEST-IRR-01",
        crop_name="cotton",
        growth_stage="flowering",
        soil_type="black_cotton",
        farm_area_acres=3.0,
        current_soil_moisture_pct=15.0,
        daily_et0_mm=5.5,
        pump_rating_hp=5.0,
        irrigation_method="drip"
    )
    assert res["success"] is True
    assert res["water_budget"]["gross_water_needed_liters"] > 0
    assert res["solar_pump_dispatch"]["runtime_minutes"] > 0
    assert "iot_mqtt_control_packet" in res
    assert "kisanseva/pumps/" in res["iot_mqtt_control_packet"]["topic"]


def test_precision_irrigation_endpoints():
    """Verify HTTP GET /soil-types, GET /demo, and POST /schedule."""
    resp_types = client.get("/api/irrigation/soil-types")
    assert resp_types.status_code == 200
    assert "soil_profiles" in resp_types.json()

    resp_demo = client.get("/api/irrigation/demo")
    assert resp_demo.status_code == 200
    assert resp_demo.json()["success"] is True

    payload = {
        "farm_id": "HR-KRL-331",
        "crop_name": "wheat",
        "growth_stage": "vegetative",
        "soil_type": "loam",
        "farm_area_acres": 2.5,
        "current_soil_moisture_pct": 14.0,
        "daily_et0_mm": 4.5,
        "pump_rating_hp": 3.0,
        "irrigation_method": "sprinkler"
    }
    resp_sched = client.post("/api/irrigation/schedule", json=payload)
    assert resp_sched.status_code == 200
    assert resp_sched.json()["irrigation_urgency"] in ["URGENT", "SCHEDULED", "ADEQUATE"]


# ---------------------------------------------------------------------------
# 4. BIO-SURVEILLANCE & EPIDEMIC SWARM TESTS
# ---------------------------------------------------------------------------

def test_epidemic_swarm_service_direct():
    """Verify pathogen R0 dynamics, downwind corridor trajectory, and barrier protocols."""
    sim = epidemic_swarm_service.simulate_epidemic_spread(
        pathogen_key="yellow_rust",
        origin_district="Gurdaspur, Punjab",
        active_focal_cases=10,
        ambient_temp_c=14.0,  # Optimal for yellow rust
        ambient_humidity_pct=85.0,
        wind_speed_kmh=18.0,
        wind_azimuth_degrees=45.0,
        surrounding_crop_acreage=4000.0
    )
    assert sim["success"] is True
    assert sim["epidemiological_metrics"]["effective_reproduction_rt"] >= 1.0
    assert len(sim["downwind_trajectory_forecast"]) == 3
    assert sim["prophylactic_barrier_protocol"]["quarantine_buffer_radius_km"] > 0
    assert "Tebuconazole" in sim["prophylactic_barrier_protocol"]["advisory"] or "Propiconazole" in sim["prophylactic_barrier_protocol"]["advisory"]


def test_epidemic_swarm_endpoints():
    """Verify HTTP GET /pathogens, GET /demo, and POST /simulate."""
    resp_path = client.get("/api/epidemic/pathogens")
    assert resp_path.status_code == 200
    assert "monitored_pathogens" in resp_path.json()

    resp_demo = client.get("/api/epidemic/demo")
    assert resp_demo.status_code == 200
    assert resp_demo.json()["success"] is True

    payload = {
        "pathogen_key": "late_blight",
        "origin_district": "Agra, Uttar Pradesh",
        "active_focal_cases": 15,
        "ambient_temp_c": 16.0,
        "ambient_humidity_pct": 90.0,
        "wind_speed_kmh": 12.0,
        "wind_azimuth_degrees": 180.0,
        "surrounding_crop_acreage": 6000.0
    }
    resp_sim = client.post("/api/epidemic/simulate", json=payload)
    assert resp_sim.status_code == 200
    data = resp_sim.json()
    assert data["target_crop"] == "Potato / Tomato"
    assert "RED ALERT" in data["epidemiological_metrics"]["threat_level"] or "AMBER WARNING" in data["epidemiological_metrics"]["threat_level"]


# ---------------------------------------------------------------------------
# 5. REAL-TIME WEATHER-DRIVEN CROP PREDICTION TESTS
# ---------------------------------------------------------------------------

def test_realtime_weather_crop_service_direct():
    """Verify live meteorological fetch and weather-adjusted crop rankings."""
    from backend.app.services.realtime_weather_crop_service import realtime_weather_crop_service

    weather = realtime_weather_crop_service.fetch_live_weather(location_name="Ludhiana")
    assert weather["success"] is True
    assert "temperature_c" in weather
    assert "humidity_pct" in weather
    assert "cumulative_rainfall_7d_mm" in weather

    # Test live prediction with soil inputs
    rec = realtime_weather_crop_service.predict_crops_with_live_weather(
        ph=6.8,
        nitrogen=120.0,
        phosphorus=35.0,
        potassium=110.0,
        soil_type="Alluvial Soil",
        location_name="Ludhiana, Punjab"
    )
    assert rec["status"] == "SUCCESS"
    assert len(rec["realtime_crop_rankings"]) > 0
    top = rec["realtime_crop_rankings"][0]
    assert "realtime_adjusted_score" in top
    assert "live_weather_rationale" in top


def test_realtime_weather_crop_endpoints():
    """Verify HTTP GET /live-weather-telemetry, POST /predict-with-live-weather, and GET /demo-live-weather."""
    resp_tele = client.get("/api/crop/live-weather-telemetry?location=Karnal")
    assert resp_tele.status_code == 200
    assert resp_tele.json()["success"] is True

    resp_demo = client.get("/api/crop/demo-live-weather?district=karnal")
    assert resp_demo.status_code == 200
    assert resp_demo.json()["status"] == "SUCCESS"
    assert "primary_recommended_crop" in resp_demo.json()

    post_payload = {
        "ph": 7.5,
        "nitrogen": 115.0,
        "phosphorus": 32.0,
        "potassium": 105.0,
        "soil_type": "Black Soil",
        "location_name": "Nagpur, Maharashtra",
        "override_temperature": 32.5,
        "override_rainfall": 95.0
    }
    resp_post = client.post("/api/crop/predict-with-live-weather", json=post_payload)
    assert resp_post.status_code == 200
    post_data = resp_post.json()
    assert post_data["status"] == "SUCCESS"
    assert post_data["soil_context"]["soil_type"] == "Black Soil"
    assert len(post_data["realtime_crop_rankings"]) > 0

