"""End-to-end tests for EDAPHNET pipeline including happy path and negative tests."""

import io
import numpy as np
from PIL import Image
import pytest
from src.edaphnet.application.pipeline_service import EdaphNetPipelineService
from src.edaphnet.domain.enums import ReliabilityLevel


@pytest.fixture
def service():
    return EdaphNetPipelineService()


def create_sample_textured_image(base_color=(120, 80, 50), size=(250, 250)):
    """Generate a textured image with sufficient gradient variance to pass sharpness gate."""
    rng = np.random.RandomState(42)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for c in range(3):
        channel = base_color[c] + rng.randint(-25, 26, size=(size[1], size[0]))
        arr[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_e2e_pipeline_happy_path(service):
    """Full workflow: Valid soil image + Valid leaf image + Agronomic features -> Analysis."""
    soil_bytes = create_sample_textured_image(base_color=(130, 85, 50))  # Earthy soil tone
    leaf_bytes = create_sample_textured_image(base_color=(45, 150, 55))  # Foliage green tone

    agronomic = {
        "N": 85.0,
        "P": 45.0,
        "K": 40.0,
        "ph": 6.8,
        "temperature": 27.0,
        "humidity": 78.0,
        "rainfall": 190.0,
    }

    response = service.analyze(
        soil_image_bytes=soil_bytes,
        leaf_image_bytes=leaf_bytes,
        agronomic_features=agronomic,
        include_explainability=True,
        include_segmentation=True,
        include_localization=True,
    )

    # Assertions
    assert response.request_id is not None
    assert response.soil is not None
    assert response.disease is not None
    assert response.segmentation is not None
    assert len(response.recommendations) >= 5
    assert response.reliability.level in [ReliabilityLevel.HIGH, ReliabilityLevel.MODERATE]
    assert response.inference_latency_ms > 0.0


def test_e2e_negative_corrupted_image(service):
    """Corrupted image rejection."""
    corrupted_bytes = b"CORRUPTED_RAW_STREAM_DATA_NOT_VALID"
    response = service.analyze(soil_image_bytes=corrupted_bytes)
    assert response.soil is not None
    assert response.reliability.level == ReliabilityLevel.ABSTAIN


def test_e2e_negative_missing_parameters(service):
    """Missing laboratory parameters handled by Missing Data Strategy."""
    response = service.analyze(agronomic_features={})
    assert len(response.recommendations) > 0
    assert len(response.missing_measurements) >= 5
    assert response.reliability.level in [ReliabilityLevel.MODERATE, ReliabilityLevel.LOW]


def test_e2e_negative_extreme_ood_image(service):
    """Out-of-distribution blue image should trigger rejection or abstain."""
    blue_img = create_sample_textured_image(base_color=(0, 20, 245))  # Bright blue
    response = service.analyze(soil_image_bytes=blue_img)
    assert response.reliability.level in [ReliabilityLevel.ABSTAIN, ReliabilityLevel.LOW]
    assert response.reliability.ood_detected or response.soil.confidence < 0.85

