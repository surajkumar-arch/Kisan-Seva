"""Integration tests for EDAPHNET FastAPI Endpoints."""

import io
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from apps.api.main import app

client = TestClient(app)


import numpy as np


def create_test_image_bytes(color=(120, 80, 50), size=(250, 250)):
    rng = np.random.RandomState(42)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for c in range(3):
        ch = color[c] + rng.randint(-25, 26, size=(size[1], size[0]))
        arr[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()



def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_version_endpoint():
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"


def test_list_models_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 4
    model_names = [m["model_name"] for m in models]
    assert "soil_classifier" in model_names
    assert "plant_disease_classifier" in model_names


def test_recommendations_endpoint():
    response = client.post(
        "/api/v1/recommendations",
        data={
            "soil_type": "alluvial",
            "n": 80.0,
            "p": 40.0,
            "k": 40.0,
            "ph": 6.5,
            "temperature": 25.0,
            "humidity": 70.0,
            "rainfall": 150.0,
        },
    )
    assert response.status_code == 200
    recs = response.json()
    assert len(recs) == 5
    assert recs[0]["rank"] == 1


def test_analyze_multimodal_endpoint():
    soil_bytes = create_test_image_bytes(color=(110, 75, 45))  # Brown soil
    leaf_bytes = create_test_image_bytes(color=(40, 160, 50))  # Green leaf

    response = client.post(
        "/api/v1/analyze",
        files={
            "soil_image": ("soil.jpg", soil_bytes, "image/jpeg"),
            "leaf_image": ("leaf.jpg", leaf_bytes, "image/jpeg"),
        },
        data={
            "n": "90.0",
            "p": "45.0",
            "k": "40.0",
            "ph": "6.5",
            "temperature": "26.0",
            "humidity": "75.0",
            "rainfall": "180.0",
            "include_explainability": "false",
            "include_segmentation": "true",
            "include_localization": "true",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "request_id" in body
    assert "reliability" in body
    assert len(body["recommendations"]) > 0
