"""Precision Crop Recommendation Service for Phase 5."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F

from backend.app.models.crop_net import (
    CROPS_LIST,
    CROP_CATALOG,
    get_crop_model,
)
from backend.app.schemas.crop_schema import (
    CropRecommendationItem,
    CropRecommendationResponse,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = BASE_DIR / "data" / "smart_agriculture.db"


def init_phase5_db(db_path: Optional[Path] = None) -> None:
    """Ensure crop_recommendations table exists in SQLite database."""
    target_path = db_path or DB_PATH
    os.makedirs(target_path.parent, exist_ok=True)

    with sqlite3.connect(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS crop_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                recommended_at TEXT NOT NULL,
                primary_crop TEXT NOT NULL,
                primary_suitability REAL NOT NULL,
                top_crops_json TEXT NOT NULL,
                soil_context_json TEXT NOT NULL,
                climate_context_json TEXT NOT NULL,
                FOREIGN KEY (request_id) REFERENCES analysis_records(request_id)
            );
            """
        )
        conn.commit()


def compute_crop_suitability(
    crop_name: str,
    neural_prob: float,
    soil_type: str,
    ph: float,
    temperature: float,
    rainfall: float,
    fvi: int,
) -> Tuple[float, str, str]:
    """Calculate the final blended suitability score, compatibility grade, and projected yield."""
    profile = CROP_CATALOG[crop_name]

    # Heuristic adjustment components
    heuristic_modifier = 1.0

    # 1. Soil Match
    if soil_type in profile["preferred_soils"]:
        heuristic_modifier += 0.25
    else:
        heuristic_modifier -= 0.15

    # 2. pH Match
    min_ph, max_ph = profile["optimal_ph"]
    if min_ph <= ph <= max_ph:
        heuristic_modifier += 0.15
    elif abs(ph - min_ph) <= 0.6 or abs(ph - max_ph) <= 0.6:
        heuristic_modifier += 0.05
    else:
        heuristic_modifier -= 0.20

    # 3. Temperature Match
    min_t, max_t = profile["optimal_temp"]
    if min_t <= temperature <= max_t:
        heuristic_modifier += 0.15
    else:
        heuristic_modifier -= 0.15

    # 4. Rainfall Match
    min_r, max_r = profile["optimal_rainfall"]
    if min_r <= rainfall <= max_r:
        heuristic_modifier += 0.15
    else:
        heuristic_modifier -= 0.10

    # Blend neural confidence (50%) and agronomic rule modifier (50%)
    raw_score = (neural_prob * 100.0 * 0.5) + (max(0.2, heuristic_modifier) * 50.0 * 0.5)
    final_score = round(max(15.0, min(98.5, raw_score)), 1)

    # Grade
    if final_score >= 80.0:
        grade = "Highly Recommended"
    elif final_score >= 60.0:
        grade = "Moderately Suitable"
    else:
        grade = "Marginally Suitable"

    # Yield projection based on Farm Vigor Index
    base_yield = profile["base_yield_tons_ha"]
    vigor_factor = max(0.6, min(1.3, fvi / 75.0))
    low_proj = round(base_yield * vigor_factor * 0.9, 1)
    high_proj = round(base_yield * vigor_factor * 1.1, 1)

    unit = "Tons / Hectare" if crop_name != "Sugarcane" else "Tons Cane / Hectare"
    projected_yield_str = f"{low_proj} - {high_proj} {unit}"

    return final_score, grade, projected_yield_str


def generate_crop_recommendations(
    request_id: str,
    db_path: Optional[Path] = None,
) -> CropRecommendationResponse:
    """Generate precision crop recommendations by synthesizing Phase 1-4 data."""
    target_db = db_path or DB_PATH
    init_phase5_db(target_db)

    # 1. Fetch Phase 1, Phase 2, and Phase 4 from SQLite
    with sqlite3.connect(target_db) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Phase 1
        cur.execute("SELECT * FROM analysis_records WHERE request_id = ?", (request_id,))
        row_p1 = cur.fetchone()
        if not row_p1:
            raise ValueError(f"Request ID '{request_id}' not found in analysis_records.")
        p1 = dict(row_p1)

        # Phase 2 (Soil)
        cur.execute("SELECT * FROM soil_health_predictions WHERE request_id = ? ORDER BY id DESC LIMIT 1", (request_id,))
        row_p2 = cur.fetchone()
        p2 = dict(row_p2) if row_p2 else {}

        # Phase 4 (Fusion)
        cur.execute("SELECT * FROM feature_fusion_records WHERE request_id = ? ORDER BY id DESC LIMIT 1", (request_id,))
        row_p4 = cur.fetchone()
        if not row_p4:
            raise ValueError(f"Feature fusion record for '{request_id}' not found. Please run Phase 4 first.")
        p4 = dict(row_p4)

    # Extract parameters
    soil_type = p2.get("soil_type", "Loamy Soil")
    ph = float(p1.get("ph") or 6.5)
    temperature = float(p1.get("temperature") or 26.0)
    rainfall = float(p1.get("rainfall") or 110.0)
    fvi = int(p4.get("farm_vigor_index") or 75)

    # 2. Model inference using 128-dim latent vector
    latent_vector = json.loads(p4["latent_embedding"])
    tensor_latent = torch.tensor([latent_vector], dtype=torch.float32)

    model = get_crop_model()
    with torch.no_grad():
        logits = model(tensor_latent)
        neural_probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

    # 3. Evaluate each crop candidate with heuristic ranking
    candidates = []
    for idx, crop_name in enumerate(CROPS_LIST):
        n_prob = float(neural_probs[idx])
        score, grade, yield_est = compute_crop_suitability(
            crop_name=crop_name,
            neural_prob=n_prob,
            soil_type=soil_type,
            ph=ph,
            temperature=temperature,
            rainfall=rainfall,
            fvi=fvi,
        )
        profile = CROP_CATALOG[crop_name]
        candidates.append({
            "crop_name": crop_name,
            "suitability_score": score,
            "compatibility_grade": grade,
            "projected_yield": yield_est,
            "growing_duration_days": profile["duration_days"],
            "water_requirement": profile["water_requirement"],
            "ideal_soil_type": profile["preferred_soils"],
            "cultivation_tips": profile["cultivation_tips"],
        })

    # Sort descending by suitability score
    candidates.sort(key=lambda x: x["suitability_score"], reverse=True)

    # Assign ranks and pick Top 5
    top_items: List[CropRecommendationItem] = []
    for rank, item in enumerate(candidates[:5], start=1):
        top_items.append(
            CropRecommendationItem(
                crop_name=item["crop_name"],
                rank=rank,
                suitability_score=item["suitability_score"],
                compatibility_grade=item["compatibility_grade"],
                projected_yield=item["projected_yield"],
                growing_duration_days=item["growing_duration_days"],
                water_requirement=item["water_requirement"],
                ideal_soil_type=item["ideal_soil_type"],
                cultivation_tips=item["cultivation_tips"],
            )
        )

    primary_crop = top_items[0].crop_name
    primary_suitability = top_items[0].suitability_score

    soil_context = {
        "soil_type": soil_type,
        "ph": ph,
        "nitrogen": p1.get("nitrogen"),
        "phosphorus": p1.get("phosphorus"),
        "potassium": p1.get("potassium"),
    }
    climate_context = {
        "temperature_c": temperature,
        "humidity_pct": p1.get("humidity"),
        "rainfall_mm": rainfall,
        "farm_vigor_index": fvi,
    }

    # 4. Save to SQLite database
    with sqlite3.connect(target_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO crop_recommendations (
                request_id,
                recommended_at,
                primary_crop,
                primary_suitability,
                top_crops_json,
                soil_context_json,
                climate_context_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                datetime.now(timezone.utc).isoformat(),
                primary_crop,
                primary_suitability,
                json.dumps([item.model_dump() for item in top_items]),
                json.dumps(soil_context),
                json.dumps(climate_context),
            )
        )
        conn.commit()

    return CropRecommendationResponse(
        status="success",
        message="Precision crop recommendations generated successfully",
        request_id=request_id,
        primary_crop=primary_crop,
        primary_suitability=primary_suitability,
        recommendations=top_items,
        soil_context=soil_context,
        climate_context=climate_context,
        ready_for_phase_6=True,
        next_step="Phase 6: Backend/API Integration & Orchestration",
    )


def recommend_crops_direct(
    ph: float,
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    temperature: float,
    humidity: float,
    rainfall: float,
    soil_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Provide precision crop recommendation directly from soil nutrients & weather readings."""
    candidates = []

    for crop_name, profile in CROP_CATALOG.items():
        score = 100.0

        # 1. pH evaluation
        min_ph, max_ph = profile.get("optimal_ph", (6.0, 7.5))
        if min_ph <= ph <= max_ph:
            score += 0.0
        else:
            diff = min(abs(ph - min_ph), abs(ph - max_ph))
            score -= min(35.0, diff * 18.0)

        # 2. Nitrogen evaluation
        opt_n = profile.get("optimal_n", (80.0, 160.0))
        if opt_n[0] <= nitrogen <= opt_n[1]:
            score += 0.0
        else:
            n_diff = min(abs(nitrogen - opt_n[0]), abs(nitrogen - opt_n[1]))
            score -= min(25.0, (n_diff / max(1.0, opt_n[1])) * 30.0)

        # 3. Phosphorus evaluation
        opt_p = profile.get("optimal_p", (30.0, 70.0))
        if opt_p[0] <= phosphorus <= opt_p[1]:
            score += 0.0
        else:
            p_diff = min(abs(phosphorus - opt_p[0]), abs(phosphorus - opt_p[1]))
            score -= min(20.0, (p_diff / max(1.0, opt_p[1])) * 25.0)

        # 4. Potassium evaluation
        opt_k = profile.get("optimal_k", (40.0, 140.0))
        if opt_k[0] <= potassium <= opt_k[1]:
            score += 0.0
        else:
            k_diff = min(abs(potassium - opt_k[0]), abs(potassium - opt_k[1]))
            score -= min(20.0, (k_diff / max(1.0, opt_k[1])) * 25.0)

        # 5. Temperature evaluation
        opt_t = profile.get("optimal_temp", (18.0, 32.0))
        if opt_t[0] <= temperature <= opt_t[1]:
            score += 0.0
        else:
            t_diff = min(abs(temperature - opt_t[0]), abs(temperature - opt_t[1]))
            score -= min(30.0, t_diff * 4.0)

        # 6. Rainfall evaluation
        opt_r = profile.get("optimal_rainfall", (60.0, 160.0))
        if opt_r[0] <= rainfall <= opt_r[1]:
            score += 0.0
        else:
            r_diff = min(abs(rainfall - opt_r[0]), abs(rainfall - opt_r[1]))
            score -= min(30.0, (r_diff / max(1.0, opt_r[1])) * 35.0)

        # 7. Soil match if provided
        if soil_type:
            preferred = profile.get("preferred_soils", [])
            if any(soil_type.lower() in p.lower() for p in preferred):
                score += 5.0
            else:
                score -= 10.0

        final_suitability = round(max(18.0, min(99.0, score)), 1)

        if final_suitability >= 80.0:
            grade = "Highly Recommended"
        elif final_suitability >= 60.0:
            grade = "Moderately Suitable"
        else:
            grade = "Marginally Suitable"

        base_yield = profile.get("base_yield_tons_ha", 3.0)
        proj_yield = f"{round(base_yield * 0.9, 1)} - {round(base_yield * 1.15, 1)} Tons / Ha"
        if crop_name == "Sugarcane":
            proj_yield = f"{round(base_yield * 0.9, 1)} - {round(base_yield * 1.15, 1)} Tons Cane / Ha"

        candidates.append({
            "crop_name": crop_name,
            "suitability_score": final_suitability,
            "compatibility_grade": grade,
            "projected_yield": proj_yield,
            "growing_duration_days": profile.get("duration_days", "110-130 Days"),
            "water_requirement": profile.get("water_requirement", "Moderate"),
            "ideal_soil_type": profile.get("preferred_soils", []),
            "cultivation_tips": profile.get("cultivation_tips", []),
            "optimal_ranges": {
                "ph": profile.get("optimal_ph"),
                "nitrogen": profile.get("optimal_n"),
                "phosphorus": profile.get("optimal_p"),
                "potassium": profile.get("optimal_k"),
                "temperature": profile.get("optimal_temp"),
                "rainfall": profile.get("optimal_rainfall"),
            }
        })

    # Sort candidates descending
    candidates.sort(key=lambda x: x["suitability_score"], reverse=True)
    for idx, cand in enumerate(candidates, 1):
        cand["rank"] = idx

    top_crop = candidates[0]
    return {
        "status": "success",
        "primary_crop": top_crop["crop_name"],
        "primary_suitability": top_crop["suitability_score"],
        "recommendations": candidates[:5],
        "all_rankings": candidates,
        "input_profile": {
            "nitrogen": nitrogen,
            "phosphorus": phosphorus,
            "potassium": potassium,
            "ph": ph,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "soil_type": soil_type or "Unspecified",
        }
    }

