"""Test suite for Phase 6: Unified Pipeline Orchestration & Farm Diagnostics."""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def create_test_image_bytes(color: tuple = (120, 75, 45), size: tuple = (250, 250)) -> io.BytesIO:
    """Helper to generate in-memory JPEG image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_pipeline_analyze_all_multipart_upload():
    """Verify 1-click end-to-end multi-modal analysis with fresh image uploads."""
    soil_buf = create_test_image_bytes(color=(140, 80, 50))
    plant_buf = create_test_image_bytes(color=(40, 160, 40))

    files = {
        "soil_image": ("soil_sample.jpg", soil_buf, "image/jpeg"),
        "plant_image": ("leaf_sample.jpg", plant_buf, "image/jpeg"),
    }
    data = {
        "ph": 6.8,
        "nitrogen": 180.0,
        "phosphorus": 45.0,
        "potassium": 190.0,
        "temperature": 26.5,
        "humidity": 70.0,
        "rainfall": 120.0,
    }

    response = client.post("/api/pipeline/analyze-all", files=files, data=data)
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["status"] == "success"
    assert "request_id" in payload
    assert payload["request_id"].startswith("REQ-")

    # Verify Summary Section
    summary = payload["summary"]
    assert "farm_health_score" in summary
    assert 0 <= summary["farm_health_score"] <= 100
    assert "farm_status" in summary
    assert "soil_health_score" in summary
    assert "plant_health_score" in summary
    assert "farm_vigor_index" in summary
    assert "soil_type" in summary
    assert "plant_condition" in summary
    assert "primary_crop" in summary
    assert summary["primary_suitability"] >= 0.0
    assert "projected_yield" in summary
    assert "critical_alerts" in summary
    assert "action_plan" in summary
    assert len(summary["action_plan"]) >= 1
    assert "execution_time_ms" in summary
    assert "total_pipeline_ms" in summary["execution_time_ms"]

    # Verify Sub-Phase Results
    assert "phase1_input" in payload
    assert payload["phase2_soil"]["status"] == "success"
    assert payload["phase3_plant"]["status"] == "success"
    assert payload["phase4_fusion"]["status"] == "success"
    assert len(payload["phase4_fusion"]["latent_embedding"]) == 128
    assert payload["phase5_crop"]["status"] == "success"
    assert len(payload["phase5_crop"]["recommendations"]) >= 1


def test_pipeline_analyze_by_request_id():
    """Verify executing unified pipeline using pre-acquired Phase 1 request ID."""
    # 1. First upload via Phase 1 endpoint
    soil_buf = create_test_image_bytes(color=(100, 60, 40))
    plant_buf = create_test_image_bytes(color=(30, 140, 30))

    files = {
        "soil_image": ("soil.jpg", soil_buf, "image/jpeg"),
        "plant_image": ("plant.jpg", plant_buf, "image/jpeg"),
    }
    data = {
        "ph": 5.2,
        "nitrogen": 90.0,
        "phosphorus": 18.0,
        "potassium": 110.0,
        "temperature": 32.0,
        "humidity": 60.0,
        "rainfall": 80.0,
    }

    p1_resp = client.post("/api/analyze/input", files=files, data=data)
    assert p1_resp.status_code == 200
    req_id = p1_resp.json()["request_id"]

    # 2. Run pipeline by ID
    p6_resp = client.post(f"/api/pipeline/analyze-by-id/{req_id}")
    assert p6_resp.status_code == 200
    payload = p6_resp.json()

    assert payload["status"] == "success"
    assert payload["request_id"] == req_id
    assert payload["summary"]["soil_type"] is not None
    assert payload["summary"]["primary_crop"] is not None
    # Acidic alert check
    alert_categories = [a["category"] for a in payload["summary"]["critical_alerts"]]
    assert "SOIL" in alert_categories or "NUTRIENT" in alert_categories


def test_pipeline_caching():
    """Verify in-memory caching returns immediate response for identical request_id."""
    soil_buf = create_test_image_bytes()
    plant_buf = create_test_image_bytes()

    files = {
        "soil_image": ("soil.jpg", soil_buf, "image/jpeg"),
        "plant_image": ("plant.jpg", plant_buf, "image/jpeg"),
    }
    resp1 = client.post("/api/pipeline/analyze-all", files=files, data={"ph": 6.5})
    assert resp1.status_code == 200
    req_id = resp1.json()["request_id"]

    # Second call by ID (hits cache)
    resp2 = client.post(f"/api/pipeline/analyze-by-id/{req_id}?force_refresh=false")
    assert resp2.status_code == 200
    assert resp2.json()["request_id"] == req_id
    assert resp2.json()["summary"]["farm_health_score"] == resp1.json()["summary"]["farm_health_score"]


def test_pipeline_report_formats():
    """Verify diagnostic report generation in JSON, Markdown, and HTML formats."""
    soil_buf = create_test_image_bytes()
    plant_buf = create_test_image_bytes()

    files = {
        "soil_image": ("soil.jpg", soil_buf, "image/jpeg"),
        "plant_image": ("plant.jpg", plant_buf, "image/jpeg"),
    }
    resp = client.post("/api/pipeline/analyze-all", files=files, data={"ph": 7.0})
    req_id = resp.json()["request_id"]

    # 1. JSON
    r_json = client.get(f"/api/pipeline/report/{req_id}?format=json")
    assert r_json.status_code == 200
    assert r_json.json()["request_id"] == req_id

    # 2. Markdown
    r_md = client.get(f"/api/pipeline/report/{req_id}?format=markdown")
    assert r_md.status_code == 200
    assert "KisanSeva: Farm Diagnostic & Agronomic Prescription Report" in r_md.text
    assert req_id in r_md.text

    # 3. HTML
    r_html = client.get(f"/api/pipeline/report/{req_id}?format=html")
    assert r_html.status_code == 200
    assert "<!DOCTYPE html>" in r_html.text
    assert "window.print()" in r_html.text
    assert req_id in r_html.text


def test_pipeline_history_and_filters():
    """Verify querying historical pipeline runs with pagination and filters."""
    response = client.get("/api/pipeline/history?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["total_records"] >= 1
    assert len(data["records"]) >= 1
    rec = data["records"][0]
    assert "request_id" in rec
    assert "farm_health_score" in rec
    assert "farm_vigor_index" in rec
    assert "primary_crop" in rec


def test_pipeline_statistics():
    """Verify system-wide analytics endpoint."""
    response = client.get("/api/pipeline/stats")
    assert response.status_code == 200
    stats = response.json()

    assert stats["status"] == "success"
    assert stats["total_farms_analyzed"] >= 1
    assert stats["avg_farm_health_score"] >= 0.0
    assert stats["avg_farm_vigor_index"] >= 0.0
    assert isinstance(stats["soil_distribution"], dict)
    assert isinstance(stats["disease_distribution"], dict)
    assert isinstance(stats["crop_recommendation_distribution"], dict)


def test_pipeline_batch_analysis():
    """Verify multi-farm batch execution and regional intelligence aggregation."""
    # Create two records
    soil1 = create_test_image_bytes(color=(120, 80, 50))
    plant1 = create_test_image_bytes(color=(50, 150, 50))
    r1 = client.post("/api/pipeline/analyze-all", files={"soil_image": ("s1.jpg", soil1, "image/jpeg"), "plant_image": ("p1.jpg", plant1, "image/jpeg")}).json()["request_id"]

    soil2 = create_test_image_bytes(color=(90, 60, 40))
    plant2 = create_test_image_bytes(color=(60, 160, 60))
    r2 = client.post("/api/pipeline/analyze-all", files={"soil_image": ("s2.jpg", soil2, "image/jpeg"), "plant_image": ("p2.jpg", plant2, "image/jpeg")}).json()["request_id"]

    batch_resp = client.post("/api/pipeline/batch-analyze", json=[r1, r2])
    assert batch_resp.status_code == 200
    data = batch_resp.json()

    assert data["status"] == "success"
    assert data["total_submitted"] == 2
    assert data["total_processed"] == 2
    assert data["failed_count"] == 0
    assert len(data["results"]) == 2

    reg = data["regional_summary"]
    assert reg["total_scans"] == 2
    assert reg["avg_farm_health"] >= 0.0
    assert len(reg["top_recommended_crops"]) >= 1


def test_pipeline_missing_inputs_validation():
    """Verify 400 error when neither images nor request_id are provided."""
    response = client.post("/api/pipeline/analyze-all", data={"ph": 6.5})
    assert response.status_code == 400
