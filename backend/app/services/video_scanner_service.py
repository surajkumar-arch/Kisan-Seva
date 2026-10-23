"""Continuous Video & Drone Flyover Crop Scanner Service with Real-Time HUD & Minimap.

Processes continuous farm flyover videos (or generates physics-grounded flight simulations)
to track crop rows, assign persistent Track IDs to diseased clusters, monitor rolling
health vigor, and plot the real-time flight path minimap.
"""

from __future__ import annotations

import io
import math
import random
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image


def generate_drone_flyover_simulation(
    plot_acres: float = 5.0,
    crop_type: str = "Wheat",
    duration_seconds: int = 20,
    fps: int = 2,
    base_lat: float = 30.9010,
    base_lon: float = 75.8573,
) -> Dict[str, Any]:
    """Generate a realistic drone crop flyover telemetry stream with persistent tracking IDs.

    Simulates a drone flying an 'S-pattern' (lawnmower pattern) over a farm plot,
    detecting crop stress anomalies with persistent Track IDs and generating HUD data.
    """
    total_frames = int(duration_seconds * fps)
    frames_telemetry: List[Dict[str, Any]] = []

    # Persistent clusters in the field with ground coordinates
    anomaly_clusters = [
        {"id": "TRK-01", "name": "Yellow Rust (Stripe Rust)", "severity": "High", "x_norm": 0.35, "y_norm": 0.45, "conf": 0.94},
        {"id": "TRK-02", "name": "Leaf Blight Lesions", "severity": "Moderate", "x_norm": 0.65, "y_norm": 0.28, "conf": 0.88},
        {"id": "TRK-03", "name": "Nitrogen Chlorosis Patch", "severity": "Low", "x_norm": 0.50, "y_norm": 0.72, "conf": 0.91},
        {"id": "TRK-04", "name": "Powdery Mildew Cluster", "severity": "Moderate", "x_norm": 0.22, "y_norm": 0.80, "conf": 0.86},
    ]

    total_detections_count = 0
    cumulative_infected_area = 0.0

    for i in range(total_frames):
        t = i / total_frames
        # Lawnmower pattern trajectory
        flight_x = t
        flight_y = 0.5 + 0.35 * math.sin(t * math.pi * 4.0)

        # GPS offsets in meters / degrees
        d_lat = (flight_y - 0.5) * 0.002
        d_lon = flight_x * 0.003
        cur_lat = round(base_lat + d_lat, 6)
        cur_lon = round(base_lon + d_lon, 6)
        cur_altitude = round(12.5 + 0.5 * math.sin(t * 10), 1)
        cur_speed = round(4.8 + 0.4 * math.cos(t * 8), 1)
        heading_deg = round((math.atan2(0.35 * math.cos(t * math.pi * 4.0), 0.25) * 180 / math.pi) % 360, 1)

        # Check proximity to known clusters to simulate camera field-of-view (FOV)
        detections_in_frame = []
        for cluster in anomaly_clusters:
            dist = math.hypot(flight_x - cluster["x_norm"], flight_y - cluster["y_norm"])
            if dist < 0.18:  # Cluster is visible inside drone camera frame
                # Camera projection coordinates [0, 100]%
                cam_x = int(50 + (cluster["x_norm"] - flight_x) * 350 + random.uniform(-2, 2))
                cam_y = int(50 + (cluster["y_norm"] - flight_y) * 350 + random.uniform(-2, 2))
                cam_x = max(10, min(85, cam_x))
                cam_y = max(10, min(85, cam_y))

                detections_in_frame.append({
                    "track_id": cluster["id"],
                    "condition": cluster["name"],
                    "severity": cluster["severity"],
                    "confidence": round(cluster["conf"] + random.uniform(-0.03, 0.03), 2),
                    "bbox": [cam_x, cam_y, 18, 16],  # x, y, w, h in %
                })
                total_detections_count += 1

        # Rolling crop health index
        instant_stress = len(detections_in_frame) * 18.0
        rolling_health = max(35.0, min(96.0, 92.0 - instant_stress - (t * 8.0) + random.uniform(-1.5, 1.5)))

        frames_telemetry.append({
            "frame_idx": i + 1,
            "timestamp_sec": round(i / fps, 1),
            "telemetry": {
                "latitude": cur_lat,
                "longitude": cur_lon,
                "altitude_m": cur_altitude,
                "speed_mps": cur_speed,
                "heading_deg": heading_deg,
                "flight_pos_norm": [round(flight_x, 3), round(flight_y, 3)],
            },
            "rolling_health_score": round(rolling_health, 1),
            "active_detections": detections_in_frame,
            "visible_anomalies_count": len(detections_in_frame),
        })

    avg_health = round(float(np.mean([f["rolling_health_score"] for f in frames_telemetry])), 1)
    health_rating = "Excellent" if avg_health >= 80 else ("Fair - Action Required" if avg_health >= 60 else "Critical Infestation")

    return {
        "status": "success",
        "plot_acres": plot_acres,
        "crop_type": crop_type,
        "duration_sec": duration_seconds,
        "total_frames_analyzed": total_frames,
        "fps": fps,
        "field_summary": {
            "average_health_score": avg_health,
            "field_rating": health_rating,
            "unique_clusters_tracked": len(anomaly_clusters),
            "total_detections_logged": total_detections_count,
            "affected_area_est_acres": round(plot_acres * 0.18, 2),
        },
        "clusters_catalog": anomaly_clusters,
        "frames": frames_telemetry,
    }
