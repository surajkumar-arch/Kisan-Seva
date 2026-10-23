"""Tests for Outbreak Radar, Action Advisor and Voice endpoints."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import advisor_service as advisor
from backend.app.services import community_service as radar

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Every test gets its own SQLite file so real data is never touched."""
    monkeypatch.setattr(radar, "DB_PATH", tmp_path / "community_test.db")
    yield


def _report(**kw):
    base = dict(condition="Tomato Late Blight", confidence=0.9, district="Nashik", crop="Tomato")
    base.update(kw)
    return radar.submit_report(**base)


# ---------------------------------------------------------------- Radar logic
def test_rejects_healthy_result():
    with pytest.raises(radar.ReportRejected):
        _report(condition="Healthy Foliage")


def test_rejects_low_confidence():
    with pytest.raises(radar.ReportRejected):
        _report(confidence=0.30)


def test_rejects_unknown_district_and_missing_location():
    with pytest.raises(radar.ReportRejected):
        _report(district="Atlantis")
    with pytest.raises(radar.ReportRejected):
        radar.submit_report(condition="Rice Blast", confidence=0.9)


def test_gps_is_snapped_to_district_centroid():
    saved = radar.submit_report(condition="Rice Blast", confidence=0.9, lat=19.99, lon=73.80)
    assert saved["district"] == "Nashik"
    h = radar.get_outbreaks()["hotspots"][0]
    assert h["lat"] == radar.DISTRICTS["Nashik"]["lat"]  # no exact GPS is stored


def test_watch_and_alert_levels():
    for _ in range(3):
        _report(district="Pune")
    assert radar.get_outbreaks()["hotspots"][0]["level"] == "WATCH"
    for _ in range(3):
        _report(district="Pune")
    h = radar.get_outbreaks()["hotspots"][0]
    assert h["cases"] == 6 and h["level"] == "ALERT" and h["trend"] == "new"


def test_trend_compares_with_prior_window():
    old = datetime.now(timezone.utc) - timedelta(days=20)
    for _ in range(4):
        _report(created_at=old)
    for _ in range(2):
        _report()
    h = radar.get_outbreaks(days=14)["hotspots"][0]
    assert h["prior_cases"] == 4 and h["cases"] == 2 and h["trend"] == "falling"


def test_nearby_filters_by_radius():
    for _ in range(3):
        _report(district="Nashik")
    for _ in range(3):
        _report(district="Kochi")
    near = radar.get_nearby(19.99, 73.79, radius_km=100)
    assert {h["district"] for h in near["hotspots"]} == {"Nashik"}
    assert near["highest_level"] == "WATCH"


def test_demo_data_is_flagged_and_removable():
    _report()  # one real report
    added = radar.seed_demo_data()
    assert added > 0
    data = radar.get_outbreaks(include_demo=True)
    assert any(h["includes_demo_data"] for h in data["hotspots"])
    assert radar.clear_demo_data() == added
    after = radar.get_outbreaks()
    assert after["total_cases"] == 1  # the real report survives


# ------------------------------------------------------------------ Radar API
def test_report_api_accepts_and_rejects():
    ok = client.post("/api/community/report", json={
        "condition": "Wheat Yellow Rust", "confidence": 0.8, "district": "Ludhiana", "crop": "Wheat"})
    assert ok.status_code == 201 and ok.json()["district"] == "Ludhiana"
    bad = client.post("/api/community/report", json={
        "condition": "Wheat Yellow Rust", "confidence": 0.2, "district": "Ludhiana"})
    assert bad.status_code == 422


def test_outbreaks_and_regions_api():
    client.post("/api/community/demo/seed")
    r = client.get("/api/community/outbreaks?days=14")
    assert r.status_code == 200 and r.json()["alerts"] >= 1
    assert len(client.get("/api/community/regions").json()["regions"]) > 20
    nearby = client.get("/api/community/nearby", params={"lat": 30.9, "lon": 75.85, "radius_km": 100})
    assert nearby.status_code == 200 and nearby.json()["highest_level"] in ("ALERT", "WATCH")
    cleared = client.delete("/api/community/demo")
    assert cleared.status_code == 200 and cleared.json()["reports_removed"] > 0


# -------------------------------------------------------------------- Advisor
def test_fertilizer_math_is_consistent():
    plan = advisor.build_fertilizer_plan("Rice", n=100, p=20, k=100, area_acres=1.0)
    ha = 0.404686
    # Rice mid targets: N 160, P 45, K 160 -> deficits 60 / 25 / 60 kg per ha
    assert plan["deficit_kg_per_ha"] == {"N": 60.0, "P": 25.0, "K": 60.0}
    expected_dap = round(25 / 0.46 * ha, 1)
    expected_mop = round(60 / 0.60 * ha, 1)
    assert plan["fertilizer_kg"]["dap"] == expected_dap
    assert plan["fertilizer_kg"]["mop"] == expected_mop
    # urea covers only the N that DAP did not already supply
    n_left = 60 - (25 / 0.46) * 0.18
    assert plan["fertilizer_kg"]["urea"] == round(n_left / 0.46 * ha, 1)
    total = sum(plan["fertilizer_cost_inr"].values())
    assert plan["total_fertilizer_cost_inr"] == total


def test_no_deficit_when_soil_rich_and_unknown_not_assumed():
    plan = advisor.build_fertilizer_plan("Wheat", n=500, p=500, k=None, area_acres=2.0)
    assert plan["fertilizer_kg"] == {"urea": 0.0, "dap": 0.0, "mop": 0.0}
    assert plan["unknown_nutrients"] == ["K"]


def test_economics_labels_price_basis(monkeypatch):
    monkeypatch.delenv("DATA_GOV_IN_API_KEY", raising=False)
    e = advisor.build_economics("Wheat", 2.0, 80, 30, 20)
    assert "MSP" in e["price_basis"]
    assert e["expected_gross_revenue_inr"] == round(e["expected_yield_tons"] * 10 * 2275.0, 0)
    assert e["margin_after_fertilizer_inr"] == e["expected_gross_revenue_inr"] - e["fertilizer"]["total_fertilizer_cost_inr"]


def test_crop_without_msp_has_no_revenue(monkeypatch):
    monkeypatch.delenv("DATA_GOV_IN_API_KEY", raising=False)
    e = advisor.build_economics("Tea", 1.0)
    assert e["price_inr_per_quintal"] is None and e["expected_gross_revenue_inr"] is None


def test_advisor_api_validation():
    ok = client.post("/api/advisor/economics", json={"crop": "Rice", "area_acres": 3, "nitrogen": 90})
    assert ok.status_code == 200 and ok.json()["crop"] == "Rice"
    assert client.post("/api/advisor/economics", json={"crop": "Moonrock"}).status_code == 404
    # The app's global handler maps schema-validation errors to HTTP 400.
    assert client.post("/api/advisor/economics", json={"crop": "Rice", "area_acres": 0}).status_code == 400
    cmp_ = client.post("/api/advisor/compare", json={"crops": ["Rice", "Wheat", "Cotton"], "area_acres": 1})
    assert cmp_.status_code == 200 and len(cmp_.json()["ranking"]) == 3


# ----------------------------------------------------------------------- Voice
def test_voice_degrades_gracefully_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert client.get("/api/ai/voice/status").json()["voice_question_enabled"] is False
    r = client.post("/api/ai/voice-chat", files={"audio": ("q.wav", b"RIFF0000", "audio/wav")})
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["detail"]


def test_voice_validates_input_when_enabled(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "backend.app.routes.voice.transcribe_and_answer",
        lambda audio, mime, language, ctx: {"transcript": "namaste", "reply": "ok"},
    )
    good = client.post("/api/ai/voice-chat", files={"audio": ("q.wav", b"RIFF0000", "audio/wav")},
                       data={"language": "Hindi"})
    assert good.status_code == 200 and good.json()["transcript"] == "namaste"
    bad_type = client.post("/api/ai/voice-chat", files={"audio": ("q.txt", b"hello", "text/plain")})
    assert bad_type.status_code == 415
    empty = client.post("/api/ai/voice-chat", files={"audio": ("q.wav", b"", "audio/wav")})
    assert empty.status_code == 400
