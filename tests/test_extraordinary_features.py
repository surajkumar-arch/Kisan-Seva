"""Automated Test Suite for Extraordinary Features:
1. Synthetic Multispectral NDVI & Vegetation Radiometry Engine
2. Drone Flyover Scanner with Continuous HUD & Minimap
3. Autonomous Multi-Agent Agronomic Debate Council
4. Zero-Internet Edge ONNX INT8 Offline Engine
"""

import io
import pytest
import numpy as np
from PIL import Image
from starlette.testclient import TestClient

from backend.app.main import app
from backend.app.services.spectral_service import reconstruct_spectral_bands, analyze_spectral_health
from backend.app.services.video_scanner_service import generate_drone_flyover_simulation
from backend.app.services.multi_agent_council import DebateRequest, run_council_debate
from backend.app.services.edge_onnx_service import get_edge_benchmark_metrics, run_offline_edge_inference


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_leaf_bytes():
    img = Image.new("RGB", (160, 160), color=(40, 160, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# =========================================================================
# 1. SYNTHETIC MULTISPECTRAL NDVI TESTS
# =========================================================================

def test_spectral_reconstruction_bands():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[:, :, 1] = 200  # High green (healthy leaf)
    rgb[:, :, 0] = 50   # Low red
    rgb[:, :, 2] = 30   # Low blue

    bands = reconstruct_spectral_bands(rgb)
    assert "NIR_est" in bands
    assert "NDVI" in bands
    assert "GNDVI" in bands
    assert "MSAVI" in bands
    assert bands["NDVI"].shape == (100, 100)
    assert np.mean(bands["NDVI"]) > 0.40  # Healthy green canopy should have high NDVI


def test_analyze_spectral_health(sample_leaf_bytes):
    report = analyze_spectral_health(sample_leaf_bytes)
    assert report["status"] == "success"
    assert "metrics" in report
    assert "mean_ndvi" in report["metrics"]
    assert "visualizations" in report
    assert report["visualizations"]["ndvi_heatmap"].startswith("data:image/png;base64,")


def test_api_spectral_ndvi(client, sample_leaf_bytes):
    files = {"image": ("leaf.jpg", sample_leaf_bytes, "image/jpeg")}
    res = client.post("/api/vision/spectral/ndvi-analyze", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "risk_level" in data


# =========================================================================
# 2. CONTINUOUS DRONE FLYOVER SCANNER TESTS
# =========================================================================

def test_drone_flyover_simulation():
    flight = generate_drone_flyover_simulation(plot_acres=4.0, crop_type="Wheat", duration_seconds=10, fps=2)
    assert flight["status"] == "success"
    assert flight["plot_acres"] == 4.0
    assert len(flight["frames"]) == 20
    frame0 = flight["frames"][0]
    assert "telemetry" in frame0
    assert "latitude" in frame0["telemetry"]
    assert "rolling_health_score" in frame0
    assert "unique_clusters_tracked" in flight["field_summary"]


def test_api_drone_demo_flight(client):
    res = client.get("/api/video/drone/demo-flight?plot_acres=5.0&crop_type=Cotton&duration_seconds=10")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert len(data["frames"]) > 0


# =========================================================================
# 3. MULTI-AGENT AGRONOMIC DEBATE COUNCIL TESTS
# =========================================================================

def test_multi_agent_council_debate():
    req = DebateRequest(
        condition="Yellow Rust",
        crop="Wheat",
        severity="High",
        farm_size_acres=3.0,
        soil_ph=7.2,
        upcoming_rain_mm=40.0,
    )
    verdict = run_council_debate(req)
    assert verdict["status"] == "consensus_achieved"
    assert len(verdict["council_agents"]) == 3
    assert verdict["consensus_verdict"]["safety_integrity_score"] == "98.4%"
    assert verdict["consensus_verdict"]["cryptographic_verification_token"].startswith("KS-AGR-")


def test_api_council_debate(client):
    payload = {
        "condition": "Late Blight",
        "crop": "Potato",
        "severity": "Moderate",
        "farm_size_acres": 2.0,
        "soil_ph": 6.5,
        "upcoming_rain_mm": 10.0,
    }
    res = client.post("/api/council/debate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "consensus_achieved"
    assert "consensus_verdict" in data


# =========================================================================
# 4. ZERO-INTERNET EDGE ONNX INT8 OFFLINE TESTS
# =========================================================================

def test_edge_benchmark_metrics():
    bench = get_edge_benchmark_metrics()
    assert "comparison" in bench
    assert len(bench["comparison"]) >= 4


def test_offline_edge_inference(sample_leaf_bytes):
    res = run_offline_edge_inference(sample_leaf_bytes)
    assert res["status"] == "success"
    assert res["offline_edge_execution"] is True
    assert res["network_required"] is False
    assert "telemetry" in res
    assert res["telemetry"]["edge_latency_ms"] > 0
    assert "diagnosis" in res


def test_api_edge_benchmarks(client):
    res = client.get("/api/edge/benchmarks")
    assert res.status_code == 200
    assert "comparison" in res.json()


def test_api_edge_predict_offline(client, sample_leaf_bytes):
    files = {"image": ("leaf.jpg", sample_leaf_bytes, "image/jpeg")}
    res = client.post("/api/edge/predict-offline", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["offline_edge_execution"] is True
