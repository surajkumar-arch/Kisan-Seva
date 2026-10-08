"""Multi-Modal Feature Fusion Service for Phase 4."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch

from backend.app.models.fusion_net import get_fusion_model
from backend.app.models.soil_net import SOIL_CLASSES
from backend.app.models.plant_net import PLANT_DISEASE_CLASSES
from backend.app.schemas.fusion_schema import (
    FusionResponse,
    DomainStressScores,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = BASE_DIR / "data" / "smart_agriculture.db"


def init_phase4_db(db_path: Optional[Path] = None) -> None:
    """Ensure feature_fusion_records table exists in SQLite database."""
    target_path = db_path or DB_PATH
    os.makedirs(target_path.parent, exist_ok=True)

    with sqlite3.connect(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_fusion_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                fused_at TEXT NOT NULL,
                farm_vigor_index INTEGER NOT NULL,
                vigor_status TEXT NOT NULL,
                soil_stress REAL NOT NULL,
                plant_disease_risk REAL NOT NULL,
                environmental_stress REAL NOT NULL,
                latent_dim INTEGER NOT NULL,
                latent_embedding TEXT NOT NULL,
                modality_weights TEXT NOT NULL,
                fusion_summary TEXT NOT NULL,
                FOREIGN KEY (request_id) REFERENCES analysis_records(request_id)
            );
            """
        )
        conn.commit()


def extract_fusion_features(
    phase1_record: Dict[str, Any],
    soil_prediction: Optional[Dict[str, Any]],
    plant_prediction: Optional[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct normalized tensors for Soil (10d), Plant (8d), and Climate (3d)."""
    # 1. Soil Tensor: 6 class probabilities + 4 normalized chemistry
    soil_probs = [0.166] * len(SOIL_CLASSES)
    if soil_prediction and "probabilities" in soil_prediction:
        raw_probs = soil_prediction["probabilities"]
        if isinstance(raw_probs, str):
            raw_probs = json.loads(raw_probs)
        soil_probs = [float(raw_probs.get(cls, 0.0)) for cls in SOIL_CLASSES]

    ph = float(phase1_record.get("ph") or 6.5)
    n = float(phase1_record.get("nitrogen") or 60.0)
    p = float(phase1_record.get("phosphorus") or 35.0)
    k = float(phase1_record.get("potassium") or 40.0)

    # Normalize chemistry
    norm_ph = ph / 14.0
    norm_n = min(1.0, max(0.0, n / 150.0))
    norm_p = min(1.0, max(0.0, p / 80.0))
    norm_k = min(1.0, max(0.0, k / 100.0))

    soil_features = soil_probs + [norm_ph, norm_n, norm_p, norm_k]  # len 10
    soil_tensor = torch.tensor([soil_features], dtype=torch.float32)

    # 2. Plant Tensor: 7 disease probabilities + 1 health score
    plant_probs = [0.142] * len(PLANT_DISEASE_CLASSES)
    p_health_score = 80.0
    if plant_prediction:
        raw_p_probs = plant_prediction.get("probabilities", {})
        if isinstance(raw_p_probs, str):
            raw_p_probs = json.loads(raw_p_probs)
        plant_probs = [float(raw_p_probs.get(cls, 0.0)) for cls in PLANT_DISEASE_CLASSES]
        p_health_score = float(plant_prediction.get("plant_health_score") or 80.0)

    norm_plant_health = p_health_score / 100.0
    plant_features = plant_probs + [norm_plant_health]  # len 8
    plant_tensor = torch.tensor([plant_features], dtype=torch.float32)

    # 3. Climate Tensor: 3 normalized environmental variables
    temp = float(phase1_record.get("temperature") or 25.0)
    humidity = float(phase1_record.get("humidity") or 65.0)
    rainfall = float(phase1_record.get("rainfall") or 100.0)

    norm_temp = min(1.0, max(0.0, (temp + 10.0) / 70.0))
    norm_hum = min(1.0, max(0.0, humidity / 100.0))
    norm_rain = min(1.0, max(0.0, rainfall / 300.0))

    climate_features = [norm_temp, norm_hum, norm_rain]  # len 3
    climate_tensor = torch.tensor([climate_features], dtype=torch.float32)

    return soil_tensor, plant_tensor, climate_tensor


def execute_feature_fusion(
    request_id: str,
    db_path: Optional[Path] = None,
) -> FusionResponse:
    """Execute Gated Multi-Modal Feature Fusion for a specified request ID."""
    target_db = db_path or DB_PATH
    init_phase4_db(target_db)

    # 1. Fetch Phase 1, Phase 2, and Phase 3 data from SQLite
    with sqlite3.connect(target_db) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Phase 1
        cur.execute("SELECT * FROM analysis_records WHERE request_id = ?", (request_id,))
        row_p1 = cur.fetchone()
        if not row_p1:
            raise ValueError(f"Request ID '{request_id}' not found in analysis_records.")
        phase1_record = dict(row_p1)

        # Phase 2 (Soil)
        cur.execute("SELECT * FROM soil_health_predictions WHERE request_id = ? ORDER BY id DESC LIMIT 1", (request_id,))
        row_p2 = cur.fetchone()
        soil_record = dict(row_p2) if row_p2 else None

        # Phase 3 (Plant)
        cur.execute("SELECT * FROM plant_health_predictions WHERE request_id = ? ORDER BY id DESC LIMIT 1", (request_id,))
        row_p3 = cur.fetchone()
        plant_record = dict(row_p3) if row_p3 else None

    # 2. Extract normalized tensors
    soil_t, plant_t, climate_t = extract_fusion_features(phase1_record, soil_record, plant_record)

    # 3. Model forward pass through GatedMultiModalFusionNet
    model = get_fusion_model()
    with torch.no_grad():
        latent_t, fvi_t, stress_t, weights_t = model(soil_t, plant_t, climate_t)

    latent_vec = [round(float(x), 5) for x in latent_t.squeeze(0).numpy()]
    fvi_raw = float(fvi_t.squeeze(0).item())
    stress_raw = stress_t.squeeze(0).numpy()
    weights_raw = weights_t.squeeze(0).numpy()

    # Grounding adjustment based on actual domain inputs
    soil_score = float(soil_record.get("soil_health_score", 70) if soil_record else 70)
    plant_score = float(plant_record.get("plant_health_score", 80) if plant_record else 80)
    
    # Weighted composite index: 40% soil health, 40% plant health, 20% model climate vigor
    composite_fvi = int(round(0.40 * soil_score + 0.40 * plant_score + 0.20 * fvi_raw))
    composite_fvi = max(10, min(100, composite_fvi))

    if composite_fvi >= 75:
        vigor_status = "Optimal"
    elif composite_fvi >= 50:
        vigor_status = "Moderate"
    else:
        vigor_status = "Critical"

    # Stress scores
    soil_stress_val = round(max(0.05, min(0.95, (100.0 - soil_score) / 100.0)), 3)
    plant_risk_val = round(max(0.05, min(0.95, (100.0 - plant_score) / 100.0)), 3)
    env_stress_val = round(float(stress_raw[2]), 3)

    stress_obj = DomainStressScores(
        soil_stress=soil_stress_val,
        plant_disease_risk=plant_risk_val,
        environmental_stress=env_stress_val,
    )

    modality_weights_dict = {
        "soil_modality": round(float(weights_raw[0]), 4),
        "plant_modality": round(float(weights_raw[1]), 4),
        "climate_modality": round(float(weights_raw[2]), 4),
    }

    fusion_summary = {
        "detected_soil_type": soil_record.get("soil_type", "Unknown") if soil_record else "Not analyzed",
        "soil_health_score": soil_score,
        "plant_condition": plant_record.get("condition", "Unknown") if plant_record else "Not analyzed",
        "plant_health_score": plant_score,
        "temperature_c": phase1_record.get("temperature"),
        "humidity_pct": phase1_record.get("humidity"),
        "rainfall_mm": phase1_record.get("rainfall"),
        "dominant_driver": max(modality_weights_dict, key=modality_weights_dict.get),
    }

    # 4. Save to SQLite database
    with sqlite3.connect(target_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO feature_fusion_records (
                request_id,
                fused_at,
                farm_vigor_index,
                vigor_status,
                soil_stress,
                plant_disease_risk,
                environmental_stress,
                latent_dim,
                latent_embedding,
                modality_weights,
                fusion_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                datetime.now(timezone.utc).isoformat(),
                composite_fvi,
                vigor_status,
                soil_stress_val,
                plant_risk_val,
                env_stress_val,
                len(latent_vec),
                json.dumps(latent_vec),
                json.dumps(modality_weights_dict),
                json.dumps(fusion_summary),
            )
        )
        conn.commit()

    return FusionResponse(
        status="success",
        message="Multi-modal feature fusion completed successfully",
        request_id=request_id,
        farm_vigor_index=composite_fvi,
        vigor_status=vigor_status,
        stress_scores=stress_obj,
        latent_dim=len(latent_vec),
        latent_embedding=latent_vec,
        modality_weights=modality_weights_dict,
        fusion_summary=fusion_summary,
        ready_for_phase_5=True,
        next_step="Phase 5: Crop Recommendation",
    )
