"""Pytest Test Suite for Phase 5: Precision Crop Recommendation."""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.crop_net import CROPS_LIST

client = TestClient(app)


def create_test_image(
    format: str = "JPEG",
    size: tuple = (256, 256),
    color: tuple = (60, 110, 40)
) -> bytes:
    """Helper to generate a valid test image."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


# 1. GET /api/crop/catalog
def test_crop_catalog_endpoint():
    response = client.get("/api/crop/catalog")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_crops"] == 12
    assert "Rice" in data["crops"]
    assert "Wheat" in data["crops"]
    assert "Cotton" in data["crops"]
    assert "Sugarcane" in data["crops"]
    assert len(data["catalog"]) == 12


# 2. Complete end-to-end 5-phase chained recommendation pipeline
def test_crop_recommendation_pipeline():
    # Step 1: Phase 1
    soil_b = create_test_image(color=(80, 50, 30))
    plant_b = create_test_image(color=(40, 160, 50))
    resp1 = client.post(
        "/api/analyze/input",
        files={
            "soil_image": ("soil_p5.jpg", soil_b, "image/jpeg"),
            "plant_image": ("leaf_p5.jpg", plant_b, "image/jpeg"),
        },
        data={
            "ph": "6.8",
            "nitrogen": "90",
            "phosphorus": "45",
            "potassium": "50",
            "temperature": "28",
            "humidity": "70",
            "rainfall": "120",
        }
    )
    assert resp1.status_code == 200
    req_id = resp1.json()["request_id"]

    # Step 2: Phase 2 Soil
    resp2 = client.post("/api/soil/predict", data={"request_id": req_id})
    assert resp2.status_code == 200

    # Step 3: Phase 3 Plant
    resp3 = client.post("/api/plant/predict", data={"request_id": req_id})
    assert resp3.status_code == 200

    # Step 4: Phase 4 Fusion
    resp4 = client.post("/api/fusion/fuse", data={"request_id": req_id})
    assert resp4.status_code == 200

    # Step 5: Phase 5 Crop Recommendation
    resp5 = client.post("/api/crop/recommend", data={"request_id": req_id})
    assert resp5.status_code == 200
    data = resp5.json()

    assert data["status"] == "success"
    assert data["request_id"] == req_id
    assert data["primary_crop"] in CROPS_LIST
    assert 0.0 <= data["primary_suitability"] <= 100.0
    assert len(data["recommendations"]) == 5
    assert data["ready_for_phase_6"] is True
    assert data["next_step"] == "Phase 6: Backend/API Integration & Orchestration"

    # Verify rank sequence and descending suitability score
    scores = [item["suitability_score"] for item in data["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    for idx, item in enumerate(data["recommendations"], start=1):
        assert item["rank"] == idx
        assert item["compatibility_grade"] in ["Highly Recommended", "Moderately Suitable", "Marginally Suitable"]
        assert len(item["cultivation_tips"]) > 0


# 3. Missing request_id returns 400
def test_crop_missing_request_id():
    response = client.post("/api/crop/recommend", data={"request_id": ""})
    assert response.status_code == 400


# 4. Non-existent request_id returns 404
def test_crop_nonexistent_request_id():
    response = client.post("/api/crop/recommend", data={"request_id": "REQ-NONEXISTENT"})
    assert response.status_code == 404


# 5. GET /api/crop/history/{request_id}
def test_crop_history_retrieval():
    soil_b = create_test_image()
    plant_b = create_test_image()
    resp1 = client.post(
        "/api/analyze/input",
        files={
            "soil_image": ("soil_h.jpg", soil_b, "image/jpeg"),
            "plant_image": ("leaf_h.jpg", plant_b, "image/jpeg"),
        },
        data={"ph": "6.5"}
    )
    req_id = resp1.json()["request_id"]
    client.post("/api/soil/predict", data={"request_id": req_id})
    client.post("/api/plant/predict", data={"request_id": req_id})
    client.post("/api/fusion/fuse", data={"request_id": req_id})
    client.post("/api/crop/recommend", data={"request_id": req_id})

    resp_hist = client.get(f"/api/crop/history/{req_id}")
    assert resp_hist.status_code == 200
    hist_data = resp_hist.json()
    assert hist_data["status"] == "success"
    assert len(hist_data["data"]) >= 1
    assert hist_data["data"][0]["request_id"] == req_id
    assert len(hist_data["data"][0]["top_crops"]) == 5
