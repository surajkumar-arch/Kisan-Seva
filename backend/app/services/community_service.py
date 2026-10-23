"""Community Outbreak Radar.

Farmers can anonymously share a confident, non-healthy diagnosis. Reports are
stored at *district* granularity only (no names, no exact GPS, no images) and
aggregated into hotspots with a trend and an alert level, so that neighbouring
farmers can act before a disease spreads.

Design notes
------------
* Quality gate: only reports with ``confidence >= MIN_REPORT_CONFIDENCE`` and a
  non-healthy condition are accepted. This reuses the model's calibrated
  confidence so noisy predictions do not pollute the map.
* Privacy: location is snapped to a district centroid from ``DISTRICTS``.
* Demo data is stored with ``is_demo = 1`` so it can be shown, hidden or
  deleted and is always labelled in API responses.
"""

from __future__ import annotations

import math
import os
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = BASE_DIR / "data" / "smart_agriculture.db"

MIN_REPORT_CONFIDENCE = float(os.getenv("RADAR_MIN_CONFIDENCE", "0.55"))
WATCH_MIN_CASES = int(os.getenv("RADAR_WATCH_MIN_CASES", "3"))
ALERT_MIN_CASES = int(os.getenv("RADAR_ALERT_MIN_CASES", "5"))
ALERT_GROWTH_FACTOR = float(os.getenv("RADAR_ALERT_GROWTH", "1.5"))

# Approximate district / city centroids (lat, lon) for major farming regions.
DISTRICTS: Dict[str, Dict[str, Any]] = {
    "Ludhiana": {"state": "Punjab", "lat": 30.90, "lon": 75.85},
    "Amritsar": {"state": "Punjab", "lat": 31.63, "lon": 74.87},
    "Bathinda": {"state": "Punjab", "lat": 30.21, "lon": 74.95},
    "Karnal": {"state": "Haryana", "lat": 29.69, "lon": 76.99},
    "Hisar": {"state": "Haryana", "lat": 29.15, "lon": 75.72},
    "Meerut": {"state": "Uttar Pradesh", "lat": 28.98, "lon": 77.71},
    "Lucknow": {"state": "Uttar Pradesh", "lat": 26.85, "lon": 80.95},
    "Varanasi": {"state": "Uttar Pradesh", "lat": 25.32, "lon": 82.97},
    "Gorakhpur": {"state": "Uttar Pradesh", "lat": 26.76, "lon": 83.37},
    "Patna": {"state": "Bihar", "lat": 25.59, "lon": 85.14},
    "Muzaffarpur": {"state": "Bihar", "lat": 26.12, "lon": 85.39},
    "Ranchi": {"state": "Jharkhand", "lat": 23.34, "lon": 85.31},
    "Burdwan": {"state": "West Bengal", "lat": 23.23, "lon": 87.86},
    "Bhubaneswar": {"state": "Odisha", "lat": 20.30, "lon": 85.82},
    "Guwahati": {"state": "Assam", "lat": 26.14, "lon": 91.74},
    "Dehradun": {"state": "Uttarakhand", "lat": 30.32, "lon": 78.03},
    "Jaipur": {"state": "Rajasthan", "lat": 26.91, "lon": 75.79},
    "Ahmedabad": {"state": "Gujarat", "lat": 23.02, "lon": 72.57},
    "Rajkot": {"state": "Gujarat", "lat": 22.30, "lon": 70.80},
    "Indore": {"state": "Madhya Pradesh", "lat": 22.72, "lon": 75.86},
    "Bhopal": {"state": "Madhya Pradesh", "lat": 23.26, "lon": 77.41},
    "Raipur": {"state": "Chhattisgarh", "lat": 21.25, "lon": 81.63},
    "Nagpur": {"state": "Maharashtra", "lat": 21.15, "lon": 79.09},
    "Nashik": {"state": "Maharashtra", "lat": 19.99, "lon": 73.79},
    "Pune": {"state": "Maharashtra", "lat": 18.52, "lon": 73.86},
    "Jalgaon": {"state": "Maharashtra", "lat": 21.00, "lon": 75.57},
    "Akola": {"state": "Maharashtra", "lat": 20.71, "lon": 77.00},
    "Sangli": {"state": "Maharashtra", "lat": 16.85, "lon": 74.57},
    "Kolhapur": {"state": "Maharashtra", "lat": 16.70, "lon": 74.24},
    "Hyderabad": {"state": "Telangana", "lat": 17.38, "lon": 78.49},
    "Warangal": {"state": "Telangana", "lat": 17.97, "lon": 79.59},
    "Guntur": {"state": "Andhra Pradesh", "lat": 16.31, "lon": 80.44},
    "Kurnool": {"state": "Andhra Pradesh", "lat": 15.83, "lon": 78.04},
    "Belagavi": {"state": "Karnataka", "lat": 15.85, "lon": 74.50},
    "Mysuru": {"state": "Karnataka", "lat": 12.30, "lon": 76.64},
    "Coimbatore": {"state": "Tamil Nadu", "lat": 11.00, "lon": 76.96},
    "Thanjavur": {"state": "Tamil Nadu", "lat": 10.79, "lon": 79.14},
    "Kochi": {"state": "Kerala", "lat": 9.93, "lon": 76.27},
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS community_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    district TEXT NOT NULL,
    state TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    crop TEXT,
    condition TEXT NOT NULL,
    confidence REAL NOT NULL,
    severity TEXT,
    is_demo INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reports_time ON community_reports(created_at);
CREATE INDEX IF NOT EXISTS idx_reports_district ON community_reports(district, condition);
"""


class ReportRejected(ValueError):
    """Raised when a submitted report fails the quality gate."""


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    target = Path(db_path or DB_PATH)
    os.makedirs(target.parent, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def init_community_db(db_path: Optional[Path] = None) -> None:
    """Create the community tables (idempotent)."""
    with _connect(db_path):
        pass


def list_regions() -> List[Dict[str, Any]]:
    return [
        {"district": name, "state": meta["state"], "lat": meta["lat"], "lon": meta["lon"]}
        for name, meta in sorted(DISTRICTS.items())
    ]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_district(lat: float, lon: float) -> str:
    return min(
        DISTRICTS,
        key=lambda d: haversine_km(lat, lon, DISTRICTS[d]["lat"], DISTRICTS[d]["lon"]),
    )


def _is_healthy(condition: str) -> bool:
    c = (condition or "").strip().lower()
    return c in ("", "healthy", "healthy foliage", "none") or c.startswith("healthy")


def submit_report(
    *,
    condition: str,
    confidence: float,
    district: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    crop: Optional[str] = None,
    severity: Optional[str] = None,
    is_demo: bool = False,
    created_at: Optional[datetime] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate and store one anonymous report. Raises ``ReportRejected``."""
    if _is_healthy(condition):
        raise ReportRejected("Healthy results are not shared; only diseases are reported.")
    if confidence is None or not (0.0 <= float(confidence) <= 1.0):
        raise ReportRejected("confidence must be between 0 and 1.")
    if float(confidence) < MIN_REPORT_CONFIDENCE:
        raise ReportRejected(
            f"Confidence {float(confidence):.0%} is below the {MIN_REPORT_CONFIDENCE:.0%} "
            "threshold for community reports. Retake the photo in better light."
        )

    if district:
        match = next((d for d in DISTRICTS if d.lower() == district.strip().lower()), None)
        if not match:
            raise ReportRejected(f"Unknown district '{district}'. Use /api/community/regions.")
        district = match
    elif lat is not None and lon is not None:
        district = nearest_district(float(lat), float(lon))
    else:
        raise ReportRejected("Provide a district, or lat and lon.")

    meta = DISTRICTS[district]
    ts = created_at or _now()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO community_reports (created_at, district, state, lat, lon, crop, "
            "condition, confidence, severity, is_demo) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                _iso(ts), district, meta["state"], meta["lat"], meta["lon"],
                (crop or "").strip() or None, condition.strip(), float(confidence),
                severity, 1 if is_demo else 0,
            ),
        )
        conn.commit()
        return {"id": cur.lastrowid, "district": district, "state": meta["state"], "is_demo": is_demo}


def _alert_level(cases: int, prior: int) -> str:
    growth_ok = cases >= ALERT_GROWTH_FACTOR * max(prior, 1)
    if cases >= ALERT_MIN_CASES and growth_ok:
        return "ALERT"
    if cases >= WATCH_MIN_CASES:
        return "WATCH"
    return "NORMAL"


def get_outbreaks(
    days: int = 14, include_demo: bool = True, db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Aggregate reports per district and disease; compare with the prior window."""
    days = max(1, min(int(days), 90))
    now = _now()
    cur_start = _iso(now - timedelta(days=days))
    prev_start = _iso(now - timedelta(days=2 * days))

    demo_clause = "" if include_demo else " AND is_demo = 0"
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT district, state, lat, lon, condition, crop, created_at, is_demo "
            f"FROM community_reports WHERE created_at >= ?{demo_clause}",
            (prev_start,),
        ).fetchall()

    buckets: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        key = (r["district"], r["condition"])
        b = buckets.setdefault(
            key,
            {
                "district": r["district"], "state": r["state"], "lat": r["lat"], "lon": r["lon"],
                "condition": r["condition"], "cases": 0, "prior_cases": 0,
                "crops": {}, "demo_cases": 0,
            },
        )
        if r["created_at"] >= cur_start:
            b["cases"] += 1
            if r["crop"]:
                b["crops"][r["crop"]] = b["crops"].get(r["crop"], 0) + 1
            if r["is_demo"]:
                b["demo_cases"] += 1
        else:
            b["prior_cases"] += 1

    hotspots: List[Dict[str, Any]] = []
    for b in buckets.values():
        if b["cases"] == 0:
            continue
        prior = b["prior_cases"]
        trend = "new" if prior == 0 else ("rising" if b["cases"] > prior else ("falling" if b["cases"] < prior else "steady"))
        top_crop = max(b["crops"], key=b["crops"].get) if b["crops"] else None
        hotspots.append(
            {
                "district": b["district"], "state": b["state"], "lat": b["lat"], "lon": b["lon"],
                "condition": b["condition"], "cases": b["cases"], "prior_cases": prior,
                "trend": trend, "level": _alert_level(b["cases"], prior),
                "top_crop": top_crop, "includes_demo_data": b["demo_cases"] > 0,
            }
        )
    order = {"ALERT": 0, "WATCH": 1, "NORMAL": 2}
    hotspots.sort(key=lambda h: (order[h["level"]], -h["cases"]))

    return {
        "window_days": days,
        "generated_at": _iso(now),
        "total_cases": sum(h["cases"] for h in hotspots),
        "alerts": sum(1 for h in hotspots if h["level"] == "ALERT"),
        "watch": sum(1 for h in hotspots if h["level"] == "WATCH"),
        "hotspots": hotspots,
        "thresholds": {
            "watch_min_cases": WATCH_MIN_CASES,
            "alert_min_cases": ALERT_MIN_CASES,
            "alert_growth_factor": ALERT_GROWTH_FACTOR,
            "min_report_confidence": MIN_REPORT_CONFIDENCE,
        },
    }


def get_nearby(
    lat: float, lon: float, radius_km: float = 150.0, days: int = 14,
    include_demo: bool = True, db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Hotspots within ``radius_km`` of a point, nearest first."""
    data = get_outbreaks(days=days, include_demo=include_demo, db_path=db_path)
    near = []
    for h in data["hotspots"]:
        d = haversine_km(lat, lon, h["lat"], h["lon"])
        if d <= radius_km:
            near.append({**h, "distance_km": round(d, 1)})
    near.sort(key=lambda h: (h["distance_km"], h["level"] != "ALERT"))
    return {
        "center": {"lat": lat, "lon": lon, "nearest_district": nearest_district(lat, lon)},
        "radius_km": radius_km,
        "window_days": data["window_days"],
        "hotspots": near,
        "highest_level": (
            "ALERT" if any(h["level"] == "ALERT" for h in near)
            else "WATCH" if any(h["level"] == "WATCH" for h in near)
            else "NORMAL"
        ),
    }


DEMO_SCENARIOS = [
    ("Ludhiana", "Wheat", "Wheat Yellow Rust", 9, 3),
    ("Karnal", "Wheat", "Wheat Yellow Rust", 6, 4),
    ("Nashik", "Tomato", "Tomato Late Blight", 7, 2),
    ("Guntur", "Cotton", "Cotton Bacterial Blight", 4, 4),
    ("Thanjavur", "Rice", "Rice Blast", 5, 1),
    ("Pune", "Tomato", "Tomato Early Blight", 3, 3),
]


def seed_demo_data(seed: int = 7, db_path: Optional[Path] = None) -> int:
    """Insert clearly flagged demo reports so the map is populated for demos."""
    rng = random.Random(seed)
    now = _now()
    n = 0
    for district, crop, condition, current, prior in DEMO_SCENARIOS:
        for _ in range(current):
            ts = now - timedelta(days=rng.uniform(0, 13), hours=rng.uniform(0, 12))
            submit_report(
                condition=condition, confidence=round(rng.uniform(0.62, 0.96), 2),
                district=district, crop=crop, severity=rng.choice(["Moderate", "Severe"]),
                is_demo=True, created_at=ts, db_path=db_path,
            )
            n += 1
        for _ in range(prior):
            ts = now - timedelta(days=rng.uniform(15, 27))
            submit_report(
                condition=condition, confidence=round(rng.uniform(0.62, 0.96), 2),
                district=district, crop=crop, severity="Moderate",
                is_demo=True, created_at=ts, db_path=db_path,
            )
            n += 1
    return n


def clear_demo_data(db_path: Optional[Path] = None) -> int:
    """Delete only rows flagged as demo data."""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM community_reports WHERE is_demo = 1")
        conn.commit()
        return cur.rowcount
