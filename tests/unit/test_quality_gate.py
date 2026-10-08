"""Unit tests for EDAPHNET Image Quality Gate."""

import io
import pytest
from PIL import Image
import numpy as np
from src.edaphnet.data.quality_gate import ImageQualityGate


@pytest.fixture
def quality_gate():
    return ImageQualityGate()


def create_solid_image_bytes(color: tuple, size=(300, 300), fmt="JPEG") -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def create_gradient_image_bytes(size=(300, 300)) -> bytes:
    arr = np.linspace(0, 255, size[0] * size[1], dtype=np.uint8).reshape((size[0], size[1]))
    img = Image.fromarray(np.stack([arr, arr, arr], axis=-1), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_corrupted_image(quality_gate):
    garbage_bytes = b"NOT_AN_IMAGE_RANDOM_BYTES_123456"
    report = quality_gate.validate_soil_image(garbage_bytes)
    assert not report.is_acceptable
    assert any("Corrupted" in r or "integrity" in r for r in report.reasons)


def test_empty_image(quality_gate):
    empty_bytes = b""
    report = quality_gate.validate_soil_image(empty_bytes)
    assert not report.is_acceptable
    assert any("Corrupted" in r or "integrity" in r for r in report.reasons)


def test_tiny_resolution(quality_gate):
    tiny_bytes = create_solid_image_bytes((120, 80, 50), size=(64, 64))
    report = quality_gate.validate_soil_image(tiny_bytes)
    assert not report.is_acceptable
    assert any("Resolution" in r or "resolution" in r for r in report.reasons)


def test_dark_underexposed_image(quality_gate):
    dark_bytes = create_solid_image_bytes((5, 5, 5), size=(300, 300))
    report = quality_gate.validate_soil_image(dark_bytes)
    assert not report.is_acceptable
    assert any("dark" in r.lower() or "underexpos" in r.lower() for r in report.reasons)


def test_overexposed_image(quality_gate):
    white_bytes = create_solid_image_bytes((252, 252, 252), size=(300, 300))
    report = quality_gate.validate_soil_image(white_bytes)
    assert not report.is_acceptable
    assert any("overexposed" in r.lower() for r in report.reasons)


def test_non_soil_validation(quality_gate):
    # Pure vibrant blue is flagged by soil domain check
    blue_img = Image.new("RGB", (300, 300), color=(10, 20, 240))
    buf = io.BytesIO()
    blue_img.save(buf, format="JPEG")
    report = quality_gate.validate_soil_image(buf.getvalue())
    assert not report.is_acceptable or any("Soil domain" in r or "domain_mismatch" in r for r in report.reasons)


def test_non_leaf_validation(quality_gate):
    # Pure bright cyan/blue is flagged by leaf domain check
    blue_img = Image.new("RGB", (300, 300), color=(0, 10, 230))
    buf = io.BytesIO()
    blue_img.save(buf, format="JPEG")
    report = quality_gate.validate_leaf_image(buf.getvalue())
    assert not report.is_acceptable or any("Foliage domain" in r or "domain_mismatch" in r for r in report.reasons)
