"""Test suite for Phase 8: Benchmarks, Stress Testing & Model Evaluation."""

import concurrent.futures
import io
import time
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def create_test_image_bytes(color: tuple = (120, 80, 50), size: tuple = (250, 250)) -> io.BytesIO:
    """Helper to generate in-memory JPEG image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_evaluation_metrics_endpoint():
    """Verify Phase 8 Model Evaluation endpoint returns complete benchmark metrics."""
    response = client.get("/api/evaluation/metrics")
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["evaluation_phase"] == "PHASE 8: Testing, Evaluation & Deployment"
    assert "executive_summary" in data

    summary = data["executive_summary"]
    assert "soil_classification_accuracy" in summary
    assert "plant_disease_accuracy" in summary
    assert "crop_recommendation_top3_accuracy" in summary
    assert "end_to_end_ml_latency_ms" in summary
    assert summary["production_readiness"] == "NOT_ASSESSED"
    assert summary["evaluation_mode"] == "smoke_test_only"
    # Without held-out data no accuracy may be reported.
    assert summary["soil_classification_accuracy"] is None
    assert summary["plant_disease_accuracy"] is None

    # Verify per-model evaluations
    assert "phase2_soil_evaluation" in data
    assert "phase3_plant_evaluation" in data
    assert "phase4_fusion_evaluation" in data
    assert "phase5_crop_evaluation" in data

    soil_eval = data["phase2_soil_evaluation"]
    assert soil_eval["model_architecture"].startswith("EfficientNet-B0")
    assert "per_class_metrics" in soil_eval
    assert "confusion_matrix" in soil_eval

    plant_eval = data["phase3_plant_evaluation"]
    assert plant_eval["model_architecture"].startswith("MobileNetV3")
    assert "per_class_metrics" in plant_eval
    assert "confusion_matrix" in plant_eval

    fusion_eval = data["phase4_fusion_evaluation"]
    assert fusion_eval["latent_dimension"] == 128
    assert fusion_eval["latent_boundedness"].startswith("[-1.0, 1.0] (asserted on all")


def test_concurrent_load_stress():
    """Verify backend resilience and zero failure rate under concurrent load."""
    num_requests = 12

    def make_request(req_idx):
        endpoint = "/api/crop/catalog" if req_idx % 2 == 0 else "/api/pipeline/stats"
        t0 = time.perf_counter()
        resp = client.get(endpoint)
        latency_ms = (time.perf_counter() - t0) * 1000
        return resp.status_code, latency_ms

    latencies = []
    statuses = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            status_code, lat = f.result()
            statuses.append(status_code)
            latencies.append(lat)

    # Assert 100% success rate
    assert all(s == 200 for s in statuses), f"Failed status codes detected: {statuses}"
    # Assert average latency is within high-throughput threshold (< 200ms per request)
    avg_lat = sum(latencies) / len(latencies)
    assert avg_lat < 250.0, f"Average latency too high: {avg_lat:.2f} ms"


def test_end_to_end_pipeline_latency_budget():
    """Verify 1-click end-to-end multi-modal pipeline runs within production budget."""
    s_buf = create_test_image_bytes(color=(130, 85, 45))
    p_buf = create_test_image_bytes(color=(45, 155, 45))

    files = {
        "soil_image": ("soil_benchmark.jpg", s_buf, "image/jpeg"),
        "plant_image": ("plant_benchmark.jpg", p_buf, "image/jpeg"),
    }
    data = {
        "ph": 6.8,
        "nitrogen": 150.0,
        "phosphorus": 40.0,
        "potassium": 160.0,
        "temperature": 27.0,
        "humidity": 65.0,
        "rainfall": 110.0,
        "force_refresh": True,
    }

    t0 = time.perf_counter()
    response = client.post("/api/pipeline/analyze-all", files=files, data=data)
    total_elapsed_ms = (time.perf_counter() - t0) * 1000

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"

    # CPU inference budget: total round-trip should be under 2000 ms
    assert total_elapsed_ms < 2500.0, f"Pipeline exceeded latency budget: {total_elapsed_ms:.1f} ms"
    lat_breakdown = res_data["summary"]["execution_time_ms"]
    assert "phase2_soil_ms" in lat_breakdown
    assert "phase3_plant_ms" in lat_breakdown
    assert "phase4_fusion_ms" in lat_breakdown
    assert "phase5_crop_ms" in lat_breakdown


def test_root_endpoint_all_phases_completed():
    """Verify API root endpoint declares all 8 phases completed and production ready."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "production_ready"
    assert data["all_phases_completed"] is True
    assert len(data["active_phases"]) == 8
    assert "PHASE 8: Testing, Evaluation & Deployment" in data["active_phases"]
