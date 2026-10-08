"""Pytest Test Suite for Phase 3: Plant Health & Foliage Disease Detection."""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.plant_net import PLANT_DISEASE_CLASSES

client = TestClient(app)


def create_test_image(
    format: str = "JPEG",
    size: tuple = (256, 256),
    color: tuple = (40, 160, 60)
) -> bytes:
    """Helper to generate a test plant foliage image in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


# 1. GET /api/plant/classes
def test_plant_classes_endpoint():
    response = client.get("/api/plant/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["classes"]) == 7
    assert "Healthy Foliage" in data["classes"]
    assert "Early Blight" in data["classes"]
    assert "Late Blight" in data["classes"]
    assert "Powdery Mildew" in data["classes"]
    assert "metadata" in data


# 2. POST /api/plant/predict with direct plant_image upload
def test_plant_predict_direct_upload():
    leaf_bytes = create_test_image(color=(35, 175, 45))  # Lush green tone
    response = client.post(
        "/api/plant/predict",
        files={"plant_image": ("leaf_test.jpg", leaf_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["condition"] in PLANT_DISEASE_CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0 <= data["plant_health_score"] <= 100
    assert "pathogen_type" in data
    assert "severity_level" in data
    assert "symptoms" in data
    assert isinstance(data["organic_treatment"], list)
    assert isinstance(data["chemical_treatment"], list)
    assert isinstance(data["preventive_measures"], list)
    assert data["ready_for_phase_4"] is True
    assert data["next_step"] == "Phase 4: Feature Fusion"

    # Probabilities sum ~ 1.0
    probs = data["probabilities"]
    assert len(probs) == 7
    total_prob = sum(probs.values())
    assert 0.99 <= total_prob <= 1.01


# 3. POST /api/plant/predict chained from Phase 1 request_id
def test_plant_predict_from_request_id():
    # Step 1: Create Phase 1 record
    soil_bytes = create_test_image(color=(120, 80, 50))
    plant_bytes = create_test_image(color=(30, 160, 50))
    resp_init = client.post(
        "/api/analyze/input",
        files={
            "soil_image": ("soil_p3.jpg", soil_bytes, "image/jpeg"),
            "plant_image": ("leaf_p3.jpg", plant_bytes, "image/jpeg"),
        },
        data={"ph": "6.5"}
    )
    assert resp_init.status_code == 200
    req_id = resp_init.json()["request_id"]

    # Step 2: Call Phase 3 predict with request_id
    resp_pred = client.post(
        "/api/plant/predict",
        data={"request_id": req_id}
    )
    assert resp_pred.status_code == 200
    pred_data = resp_pred.json()
    assert pred_data["request_id"] == req_id
    assert pred_data["condition"] in PLANT_DISEASE_CLASSES


# 4. Predict without request_id or plant_image returns 400
def test_plant_predict_missing_inputs():
    response = client.post("/api/plant/predict", data={})
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "Missing request_id or plant_image" in data["errors"]["input"]


# 5. Predict with non-existent request_id returns 404
def test_plant_predict_nonexistent_request_id():
    response = client.post(
        "/api/plant/predict",
        data={"request_id": "REQ-NONEXISTENT"}
    )
    assert response.status_code == 404


# 6. Treatment prescription validation
def test_plant_treatment_prescription():
    leaf_bytes = create_test_image(color=(180, 80, 40))  # Discolored leaf
    response = client.post(
        "/api/plant/predict",
        files={"plant_image": ("diseased_leaf.jpg", leaf_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    # Treatments and preventions must not be empty
    assert len(data["organic_treatment"]) > 0
    assert len(data["preventive_measures"]) > 0


# 7. GET /api/plant/history/{request_id}
def test_plant_history_retrieval():
    leaf_bytes = create_test_image()
    resp_pred = client.post(
        "/api/plant/predict",
        files={"plant_image": ("leaf_hist.jpg", leaf_bytes, "image/jpeg")}
    )
    assert resp_pred.status_code == 200
    req_id = resp_pred.json()["request_id"]

    resp_hist = client.get(f"/api/plant/history/{req_id}")
    assert resp_hist.status_code == 200
    hist_data = resp_hist.json()
    assert hist_data["status"] == "success"
    assert len(hist_data["data"]) >= 1
    assert hist_data["data"][0]["request_id"] == req_id
