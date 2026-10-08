"""Pytest Test Suite for Phase 4: Multi-Modal Feature Fusion."""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def create_test_image(
    format: str = "JPEG",
    size: tuple = (256, 256),
    color: tuple = (50, 120, 50)
) -> bytes:
    """Helper to generate a valid test image."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


# 1. Complete end-to-end chained fusion test
def test_feature_fusion_pipeline():
    # Step 1: Phase 1 Input Acquisition
    soil_bytes = create_test_image(color=(90, 60, 40))
    plant_bytes = create_test_image(color=(35, 160, 45))
    resp_init = client.post(
        "/api/analyze/input",
        files={
            "soil_image": ("soil_f4.jpg", soil_bytes, "image/jpeg"),
            "plant_image": ("leaf_f4.jpg", plant_bytes, "image/jpeg"),
        },
        data={
            "ph": "6.8",
            "nitrogen": "85",
            "phosphorus": "40",
            "potassium": "45",
            "temperature": "27",
            "humidity": "68",
            "rainfall": "110",
        }
    )
    assert resp_init.status_code == 200
    req_id = resp_init.json()["request_id"]

    # Step 2: Phase 2 Soil Prediction
    resp_soil = client.post("/api/soil/predict", data={"request_id": req_id})
    assert resp_soil.status_code == 200

    # Step 3: Phase 3 Plant Prediction
    resp_plant = client.post("/api/plant/predict", data={"request_id": req_id})
    assert resp_plant.status_code == 200

    # Step 4: Phase 4 Feature Fusion
    resp_fusion = client.post("/api/fusion/fuse", data={"request_id": req_id})
    assert resp_fusion.status_code == 200
    data = resp_fusion.json()

    assert data["status"] == "success"
    assert data["request_id"] == req_id
    assert 0 <= data["farm_vigor_index"] <= 100
    assert data["vigor_status"] in ["Optimal", "Moderate", "Critical"]
    assert data["latent_dim"] == 128
    assert len(data["latent_embedding"]) == 128
    assert data["ready_for_phase_5"] is True
    assert data["next_step"] == "Phase 5: Crop Recommendation"

    # Modality weights sum to ~1.0
    weights = data["modality_weights"]
    assert len(weights) == 3
    assert 0.99 <= sum(weights.values()) <= 1.01

    # Stress scores are bounded
    stress = data["stress_scores"]
    assert 0.0 <= stress["soil_stress"] <= 1.0
    assert 0.0 <= stress["plant_disease_risk"] <= 1.0
    assert 0.0 <= stress["environmental_stress"] <= 1.0


# 2. Missing request_id returns 400
def test_fusion_missing_request_id():
    response = client.post("/api/fusion/fuse", data={"request_id": ""})
    assert response.status_code == 400


# 3. Non-existent request_id returns 404
def test_fusion_nonexistent_request_id():
    response = client.post("/api/fusion/fuse", data={"request_id": "REQ-NONEXISTENT"})
    assert response.status_code == 404


# 4. Latent embedding normalization check
def test_latent_embedding_range():
    soil_bytes = create_test_image()
    plant_bytes = create_test_image()
    resp_init = client.post(
        "/api/analyze/input",
        files={
            "soil_image": ("soil_range.jpg", soil_bytes, "image/jpeg"),
            "plant_image": ("leaf_range.jpg", plant_bytes, "image/jpeg"),
        },
        data={"ph": "7.0"}
    )
    req_id = resp_init.json()["request_id"]
    client.post("/api/soil/predict", data={"request_id": req_id})
    client.post("/api/plant/predict", data={"request_id": req_id})

    resp_fusion = client.post("/api/fusion/fuse", data={"request_id": req_id})
    assert resp_fusion.status_code == 200
    embedding = resp_fusion.json()["latent_embedding"]

    # All embedding elements must be bounded between -1.0 and 1.0 (Tanh activation)
    for val in embedding:
        assert -1.0 <= val <= 1.0


# 5. GET /api/fusion/history/{request_id}
def test_fusion_history_retrieval():
    soil_bytes = create_test_image()
    plant_bytes = create_test_image()
    resp_init = client.post(
        "/api/analyze/input",
        files={
            "soil_image": ("soil_hist.jpg", soil_bytes, "image/jpeg"),
            "plant_image": ("leaf_hist.jpg", plant_bytes, "image/jpeg"),
        },
        data={"ph": "6.5"}
    )
    req_id = resp_init.json()["request_id"]
    client.post("/api/soil/predict", data={"request_id": req_id})
    client.post("/api/plant/predict", data={"request_id": req_id})
    client.post("/api/fusion/fuse", data={"request_id": req_id})

    resp_hist = client.get(f"/api/fusion/history/{req_id}")
    assert resp_hist.status_code == 200
    hist_data = resp_hist.json()
    assert hist_data["status"] == "success"
    assert len(hist_data["data"]) >= 1
    assert hist_data["data"][0]["request_id"] == req_id
    assert len(hist_data["data"][0]["latent_embedding"]) == 128
