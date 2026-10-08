"""Pytest Test Suite for Phase 2: Soil Health Detection."""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.soil_net import SOIL_CLASSES

client = TestClient(app)


def create_test_image(
    format: str = "JPEG",
    size: tuple = (256, 256),
    color: tuple = (120, 80, 50)
) -> bytes:
    """Helper to generate a valid test image."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


# 1. GET /api/soil/classes
def test_soil_classes_endpoint():
    response = client.get("/api/soil/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["classes"]) == 6
    assert "Black Soil" in data["classes"]
    assert "Alluvial Soil" in data["classes"]
    assert "metadata" in data


# 2. POST /api/soil/predict with direct soil_image upload
def test_soil_predict_direct_upload():
    soil_img = create_test_image(color=(50, 40, 35))  # Dark soil tone
    response = client.post(
        "/api/soil/predict",
        files={"soil_image": ("soil_direct.jpg", soil_img, "image/jpeg")},
        data={"ph": "6.8", "nitrogen": "75", "phosphorus": "35", "potassium": "50"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["soil_type"] in SOIL_CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0 <= data["soil_health_score"] <= 100
    assert data["fertility_grade"] in ["High", "Moderate", "Low"]
    assert "physical_properties" in data
    assert "chemical_assessment" in data
    assert data["ready_for_phase_3"] is True
    assert data["next_step"] == "Phase 3: Plant Health Detection"

    # Probabilities sum ~ 1.0
    probs = data["probabilities"]
    assert len(probs) == 6
    total_prob = sum(probs.values())
    assert 0.99 <= total_prob <= 1.01


# 3. POST /api/soil/predict via Phase 1 request_id
def test_soil_predict_from_request_id():
    # Step 1: Create Phase 1 record
    soil_bytes = create_test_image(color=(160, 60, 40))  # Reddish tone
    plant_bytes = create_test_image(color=(30, 160, 50))
    resp_init = client.post(
        "/api/analyze/input",
        files={
            "soil_image": ("soil_p1.jpg", soil_bytes, "image/jpeg"),
            "plant_image": ("leaf_p1.jpg", plant_bytes, "image/jpeg"),
        },
        data={"ph": "6.2", "nitrogen": "40"}
    )
    assert resp_init.status_code == 200
    p1_data = resp_init.json()
    req_id = p1_data["request_id"]

    # Step 2: Call Phase 2 predict referencing request_id
    resp_pred = client.post(
        "/api/soil/predict",
        data={"request_id": req_id}
    )
    assert resp_pred.status_code == 200
    pred_data = resp_pred.json()
    assert pred_data["request_id"] == req_id
    assert pred_data["soil_type"] in SOIL_CLASSES
    assert pred_data["chemical_assessment"]["nitrogen_status"] == "Low"


# 4. Predict without request_id or soil_image returns 400
def test_soil_predict_missing_inputs():
    response = client.post("/api/soil/predict", data={})
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "Missing request_id or soil_image" in data["errors"]["input"]


# 5. Predict with non-existent request_id returns 404
def test_soil_predict_nonexistent_request_id():
    response = client.post(
        "/api/soil/predict",
        data={"request_id": "REQ-NONEXISTENT"}
    )
    assert response.status_code == 404


# 6. Acidic pH and Nitrogen deficiency trigger amendments
def test_soil_health_acidic_and_deficiencies():
    soil_img = create_test_image()
    response = client.post(
        "/api/soil/predict",
        files={"soil_image": ("soil.jpg", soil_img, "image/jpeg")},
        data={"ph": "4.8", "nitrogen": "20", "phosphorus": "10", "potassium": "15"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["chemical_assessment"]["ph_status"] == "Strongly Acidic"
    assert data["chemical_assessment"]["nitrogen_status"] == "Low"
    assert data["chemical_assessment"]["phosphorus_status"] == "Low"
    assert data["chemical_assessment"]["potassium_status"] == "Low"

    recs = " ".join(data["recommendations"])
    assert "lime" in recs.lower()
    assert "nitrogen" in recs.lower() or "manure" in recs.lower()


# 7. GET /api/soil/history/{request_id}
def test_soil_history_retrieval():
    soil_img = create_test_image()
    resp_pred = client.post(
        "/api/soil/predict",
        files={"soil_image": ("soil_hist.jpg", soil_img, "image/jpeg")},
        data={"ph": "7.0"}
    )
    assert resp_pred.status_code == 200
    req_id = resp_pred.json()["request_id"]

    resp_hist = client.get(f"/api/soil/history/{req_id}")
    assert resp_hist.status_code == 200
    hist_data = resp_hist.json()
    assert hist_data["status"] == "success"
    assert len(hist_data["data"]) >= 1
    assert hist_data["data"][0]["request_id"] == req_id
