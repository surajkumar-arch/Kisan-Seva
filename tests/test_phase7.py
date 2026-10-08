"""Test suite for Phase 7: Interactive Frontend, Visualizations & Agronomic Rx."""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from frontend.utils.charts import (
    create_health_gauge,
    create_crop_radar_chart,
    create_modality_weights_donut,
    create_stress_scores_bar,
    create_spatial_farm_heatmap,
)
from frontend.utils.presets import FARM_PRESETS, generate_synthetic_field_image

client = TestClient(app)


def test_agronomic_prescription_endpoint():
    """Verify medical-grade prescription endpoint returns structured clinical farming advice."""
    # 1. Run quick scan to get request_id
    buf1 = io.BytesIO()
    Image.new("RGB", (250, 250), color=(140, 80, 50)).save(buf1, format="JPEG")
    buf1.seek(0)

    buf2 = io.BytesIO()
    Image.new("RGB", (250, 250), color=(40, 160, 40)).save(buf2, format="JPEG")
    buf2.seek(0)

    p_resp = client.post(
        "/api/pipeline/analyze-all",
        files={"soil_image": ("s.jpg", buf1, "image/jpeg"), "plant_image": ("p.jpg", buf2, "image/jpeg")},
        data={"ph": 6.5, "nitrogen": 140.0}
    )
    assert p_resp.status_code == 200
    req_id = p_resp.json()["request_id"]

    # 2. Query prescription endpoint
    rx_resp = client.get(f"/api/pipeline/prescription/{req_id}")
    assert rx_resp.status_code == 200
    rx_data = rx_resp.json()

    assert rx_data["prescription_id"] == f"RX-{req_id}"
    assert "vital_signs" in rx_data
    assert "pathology_diagnosis" in rx_data
    assert "soil_diagnosis" in rx_data
    assert "prescribed_treatments" in rx_data
    assert "calendar_schedule" in rx_data
    assert len(rx_data["calendar_schedule"]) >= 3
    assert "extension_officer_notes" in rx_data


def test_agronomic_prescription_nonexistent():
    """Verify 404 for non-existent request ID in prescription route."""
    resp = client.get("/api/pipeline/prescription/REQ-NONEXISTENT")
    assert resp.status_code == 404


def test_synthetic_preset_images():
    """Verify preset image generator creates valid 300x300 JPEG images."""
    for name, p in FARM_PRESETS.items():
        soil_buf = generate_synthetic_field_image("soil", p["soil_type"])
        soil_img = Image.open(soil_buf)
        assert soil_img.size == (300, 300)
        assert soil_img.format == "JPEG"

        plant_buf = generate_synthetic_field_image("plant", p["plant_condition"])
        plant_img = Image.open(plant_buf)
        assert plant_img.size == (300, 300)
        assert plant_img.format == "JPEG"


def test_plotly_chart_builders():
    """Verify all Phase 7 interactive Plotly visualization helper functions."""
    # 1. Gauge
    fig_gauge = create_health_gauge(85, "Farm Health", "Optimal")
    assert fig_gauge is not None
    assert fig_gauge.data[0].type == "indicator"
    assert fig_gauge.data[0].value == 85

    # 2. Radar
    farm_p = {"ph": 6.5, "nitrogen": 140, "phosphorus": 35, "potassium": 150, "temperature": 27, "rainfall": 120}
    crop_p = {
        "crop_name": "Rice",
        "optimal_ph": [6.0, 7.5],
        "optimal_n": [120, 180],
        "optimal_p": [30, 60],
        "optimal_k": [120, 200],
        "optimal_temp": [22, 32],
        "optimal_rainfall": [150, 250],
    }
    fig_radar = create_crop_radar_chart(farm_p, crop_p)
    assert fig_radar is not None
    assert len(fig_radar.data) == 2  # Optimal polygon + Current polygon

    # 3. Donut
    mod_w = {"soil_weight": 0.4, "plant_weight": 0.35, "climate_weight": 0.25}
    fig_donut = create_modality_weights_donut(mod_w)
    assert fig_donut is not None
    assert fig_donut.data[0].type == "pie"

    # 4. Stress bar
    stress = {"soil_stress": 0.3, "plant_disease_risk": 0.1, "environmental_stress": 0.2}
    fig_bar = create_stress_scores_bar(stress)
    assert fig_bar is not None
    assert fig_bar.data[0].type == "bar"

    # 5. Spatial Heatmap
    fig_heat = create_spatial_farm_heatmap(82, "Healthy Foliage")
    assert fig_heat is not None
    assert fig_heat.data[0].type == "heatmap"
    assert len(fig_heat.data[0].z) == 5
