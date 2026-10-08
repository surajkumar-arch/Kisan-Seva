"""Pytest Test Suite for Phase 1: Input Acquisition, Validation, and Storage."""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def create_test_image(
    format: str = "JPEG",
    size: tuple = (256, 256),
    color: tuple = (100, 150, 200)
) -> bytes:
    """Helper to generate a valid test image in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


# 1. GET /health
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "smart-agriculture-api"


# 2. Valid soil image upload
def test_valid_soil_image_upload():
    img_bytes = create_test_image(format="JPEG", size=(300, 300))
    response = client.post(
        "/api/soil/upload",
        files={"file": ("soil_sample.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "image" in data
    assert data["image"]["dimensions"] == [300, 300]


# 3. Invalid soil image (unsupported file type / corrupt)
def test_invalid_soil_image_format():
    response = client.post(
        "/api/soil/upload",
        files={"file": ("payload.exe", b"MZDummyBinaryExecutable", "application/octet-stream")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "soil_image" in data["errors"]


def test_corrupt_soil_image():
    response = client.post(
        "/api/soil/upload",
        files={"file": ("soil.jpg", b"corruptednotanimagecontent", "image/jpeg")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"


# 4. Valid plant image upload
def test_valid_plant_image_upload():
    img_bytes = create_test_image(format="PNG", size=(250, 250))
    response = client.post(
        "/api/plant/upload",
        files={"file": ("leaf_sample.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "image" in data


# 5. Invalid plant image (resolution below 224x224)
def test_invalid_plant_image_low_resolution():
    small_img = create_test_image(format="JPEG", size=(100, 100))
    response = client.post(
        "/api/plant/upload",
        files={"file": ("leaf_tiny.jpg", small_img, "image/jpeg")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "plant_image" in data["errors"]
    assert "minimum requirement of 224x224" in data["errors"]["plant_image"]


# 6. Invalid pH validation in /api/analyze/input
def test_invalid_ph_value():
    soil_img = create_test_image(size=(224, 224))
    plant_img = create_test_image(size=(224, 224))

    response = client.post(
        "/api/analyze/input",
        files={
            "soil_image": ("soil.jpg", soil_img, "image/jpeg"),
            "plant_image": ("plant.jpg", plant_img, "image/jpeg"),
        },
        data={"ph": "100"}  # Invalid: pH cannot exceed 14
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "ph" in data["errors"]
    assert "between 0 and 14" in data["errors"]["ph"]


# 7. Invalid humidity validation
def test_invalid_humidity_value():
    soil_img = create_test_image(size=(224, 224))
    plant_img = create_test_image(size=(224, 224))

    response = client.post(
        "/api/analyze/input",
        files={
            "soil_image": ("soil.jpg", soil_img, "image/jpeg"),
            "plant_image": ("plant.jpg", plant_img, "image/jpeg"),
        },
        data={"humidity": "-50"}  # Invalid: Humidity < 0
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "humidity" in data["errors"]


# 8. Invalid N/P/K validation
def test_invalid_npk_negative():
    soil_img = create_test_image(size=(224, 224))
    plant_img = create_test_image(size=(224, 224))

    response = client.post(
        "/api/analyze/input",
        files={
            "soil_image": ("soil.jpg", soil_img, "image/jpeg"),
            "plant_image": ("plant.jpg", plant_img, "image/jpeg"),
        },
        data={
            "nitrogen": "-10",
            "phosphorus": "20",
            "potassium": "-5",
        }
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "nitrogen" in data["errors"]
    assert "potassium" in data["errors"]


# 9. Invalid rainfall validation
def test_invalid_rainfall_negative():
    soil_img = create_test_image(size=(224, 224))
    plant_img = create_test_image(size=(224, 224))

    response = client.post(
        "/api/analyze/input",
        files={
            "soil_image": ("soil.jpg", soil_img, "image/jpeg"),
            "plant_image": ("plant.jpg", plant_img, "image/jpeg"),
        },
        data={"rainfall": "-100"}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "rainfall" in data["errors"]


# 10. Valid complete input
def test_valid_complete_input():
    soil_img = create_test_image(size=(300, 300), color=(120, 80, 50))
    plant_img = create_test_image(size=(300, 300), color=(30, 150, 40))

    response = client.post(
        "/api/analyze/input",
        files={
            "soil_image": ("soil_001.jpg", soil_img, "image/jpeg"),
            "plant_image": ("leaf_001.jpg", plant_img, "image/jpeg"),
        },
        data={
            "ph": "6.5",
            "nitrogen": "80",
            "phosphorus": "40",
            "potassium": "45",
            "temperature": "28",
            "humidity": "70",
            "rainfall": "120",
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Input validation successful"
    assert data["request_id"].startswith("REQ-")
    assert data["ready_for_phase_2"] is True
    assert data["next_step"] == "Phase 2: Soil Health Detection"

    rec = data["data"]
    assert rec["soil_parameters"]["ph"] == 6.5
    assert rec["soil_parameters"]["nitrogen"] == 80.0
    assert rec["environment"]["temperature"] == 28.0
    assert rec["environment"]["humidity"] == 70.0
    assert rec["environment"]["rainfall"] == 120.0


# 11. Missing required image
def test_missing_required_image():
    soil_img = create_test_image(size=(224, 224))
    # Omitting plant_image
    response = client.post(
        "/api/analyze/input",
        files={
            "soil_image": ("soil.jpg", soil_img, "image/jpeg"),
        },
        data={"ph": "6.5"}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"


# 12. Image larger than 5 MB
def test_image_exceeds_5mb():
    # 5 MB = 5 * 1024 * 1024 = 5242880 bytes
    oversized_bytes = b"0" * (5 * 1024 * 1024 + 1024)
    response = client.post(
        "/api/soil/upload",
        files={"file": ("huge_image.jpg", oversized_bytes, "image/jpeg")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "exceeds maximum allowed limit of 5 MB" in data["errors"]["soil_image"]
