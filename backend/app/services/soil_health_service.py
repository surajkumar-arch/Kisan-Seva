"""Soil Health Detection and Analysis Service for Phase 2."""

import os
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.app.models.soil_net import (
    SOIL_CLASSES,
    SOIL_METADATA,
    get_soil_model,
)
from backend.app.schemas.input_schema import SoilParameters
from backend.app.schemas.soil_schema import (
    SoilPhysicalProperties,
    ChemicalAssessment,
    SoilPredictionResponse,
)
from backend.app.utils.image_utils import preprocess_image, validate_image_file

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = BASE_DIR / "data" / "smart_agriculture.db"


def init_phase2_db(db_path: Optional[Path] = None) -> None:
    """Ensure soil_health_predictions table exists in SQLite database."""
    target_path = db_path or DB_PATH
    os.makedirs(target_path.parent, exist_ok=True)

    with sqlite3.connect(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS soil_health_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                predicted_at TEXT NOT NULL,
                soil_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT NOT NULL,
                soil_health_score INTEGER NOT NULL,
                fertility_grade TEXT NOT NULL,
                ph_status TEXT,
                nitrogen_status TEXT,
                phosphorus_status TEXT,
                potassium_status TEXT,
                recommendations TEXT NOT NULL,
                FOREIGN KEY (request_id) REFERENCES analysis_records(request_id)
            );
            """
        )
        conn.commit()


def calculate_soil_health_score(
    soil_type: str,
    confidence: float,
    soil_params: Optional[SoilParameters] = None,
) -> Tuple[int, str, ChemicalAssessment, List[str]]:
    """Compute the multi-factor Soil Health Score (0-100), nutrient statuses, and recommendations.

    Factors:
    - Base potential per soil type (Loamy: 85, Alluvial: 80, Black: 78, Red: 65, Clay: 60, Sandy: 52)
    - Chemical sufficiency bonuses / penalties for pH, Nitrogen, Phosphorus, Potassium
    """
    base_scores = {
        "Loamy Soil": 85,
        "Alluvial Soil": 82,
        "Black Soil": 78,
        "Red Soil": 65,
        "Clay Soil": 62,
        "Sandy Soil": 52,
    }
    score = base_scores.get(soil_type, 70)
    recommendations: List[str] = []

    ph_status = None
    n_status = None
    p_status = None
    k_status = None

    if soil_params:
        # pH evaluation
        if soil_params.ph is not None:
            ph = soil_params.ph
            if ph < 5.5:
                ph_status = "Strongly Acidic"
                score -= 15
                recommendations.append(
                    "Soil is strongly acidic (pH < 5.5). Apply agricultural lime (calcium carbonate) or dolomite to raise soil pH."
                )
            elif 5.5 <= ph < 6.5:
                ph_status = "Moderately Acidic"
                score -= 5
                recommendations.append(
                    "Soil is slightly acidic. Consider moderate compost or wood ash amendment."
                )
            elif 6.5 <= ph <= 7.5:
                ph_status = "Optimal / Neutral"
                score += 10
            elif 7.5 < ph <= 8.5:
                ph_status = "Moderately Alkaline"
                score -= 5
                recommendations.append(
                    "Soil is moderately alkaline. Use organic mulches and ammonium-based fertilizers to balance pH."
                )
            else:
                ph_status = "Strongly Alkaline"
                score -= 15
                recommendations.append(
                    "Soil is strongly alkaline (pH > 8.5). Apply agricultural gypsum or elemental sulfur to leach excess sodium."
                )

        # Nitrogen evaluation
        if soil_params.nitrogen is not None:
            n = soil_params.nitrogen
            if n < 50:
                n_status = "Low"
                score -= 10
                recommendations.append(
                    "Nitrogen deficiency detected (< 50 mg/kg). Apply well-rotted farmyard manure, compost, or urea/DAP."
                )
            elif 50 <= n <= 100:
                n_status = "Medium / Sufficient"
                score += 5
            else:
                n_status = "High"
                score += 5

        # Phosphorus evaluation
        if soil_params.phosphorus is not None:
            p = soil_params.phosphorus
            if p < 20:
                p_status = "Low"
                score -= 10
                recommendations.append(
                    "Phosphorus deficiency detected (< 20 mg/kg). Apply Single Super Phosphate (SSP) or rock phosphate to strengthen root development."
                )
            elif 20 <= p <= 50:
                p_status = "Medium / Sufficient"
                score += 5
            else:
                p_status = "High"
                score += 5

        # Potassium evaluation
        if soil_params.potassium is not None:
            k = soil_params.potassium
            if k < 30:
                k_status = "Low"
                score -= 10
                recommendations.append(
                    "Potassium deficiency detected (< 30 mg/kg). Apply Muriate of Potash (MOP) to enhance disease resistance and water regulation."
                )
            elif 30 <= k <= 80:
                k_status = "Medium / Sufficient"
                score += 5
            else:
                k_status = "High"
                score += 5

    # Specific recommendations based on soil physical structure
    if soil_type == "Sandy Soil":
        recommendations.append(
            "High permeability noted: Add vermicompost and biochar regularly to increase moisture and nutrient retention capacity."
        )
    elif soil_type == "Clay Soil":
        recommendations.append(
            "Heavy texture noted: Incorporate coarse organic matter and avoid tilling when wet to prevent severe soil compaction."
        )
    elif soil_type == "Red Soil":
        recommendations.append(
            "Prone to leaching: Practice mulching and fractional split applications of fertilizers."
        )
    elif soil_type in ["Loamy Soil", "Alluvial Soil"]:
        recommendations.append(
            "Excellent native fertility: Maintain soil crumb structure with crop rotation and minimal tillage."
        )

    # Bound score between 0 and 100
    final_score = max(10, min(100, score))

    if final_score >= 75:
        fertility_grade = "High"
    elif final_score >= 50:
        fertility_grade = "Moderate"
    else:
        fertility_grade = "Low"

    chem_assessment = ChemicalAssessment(
        ph_status=ph_status,
        nitrogen_status=n_status,
        phosphorus_status=p_status,
        potassium_status=k_status,
    )

    return final_score, fertility_grade, chem_assessment, recommendations


def predict_soil_health(
    soil_image: Image.Image,
    request_id: str,
    soil_params: Optional[SoilParameters] = None,
    db_path: Optional[Path] = None,
) -> SoilPredictionResponse:
    """Execute EfficientNet-B0 inference and agronomic evaluation."""
    model = get_soil_model()

    # 1. Preprocess with ImageNet standardization
    img_array = preprocess_image(soil_image, target_size=(224, 224), normalize_mode="imagenet")
    
    # 2. Convert to PyTorch Tensor: shape (1, 3, 224, 224)
    # img_array is (224, 224, 3) -> transpose to (3, 224, 224)
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()

    # 3. Model forward pass
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()

    # Calibration heuristic to ensure strong visual domain correlation
    # Evaluate color hue / texture distribution of original RGB
    rgb = np.asarray(soil_image.convert("RGB").resize((64, 64)), dtype=np.float32)
    r_mean = float(np.mean(rgb[:, :, 0]))
    g_mean = float(np.mean(rgb[:, :, 1]))
    b_mean = float(np.mean(rgb[:, :, 2]))
    luminance = 0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean

    prior_bias = np.ones(len(SOIL_CLASSES), dtype=np.float32)
    # Red soil: high R, low G/B
    if r_mean > g_mean * 1.15 and r_mean > b_mean * 1.2:
        prior_bias[SOIL_CLASSES.index("Red Soil")] += 2.0
    # Black soil: very low luminance
    if luminance < 90:
        prior_bias[SOIL_CLASSES.index("Black Soil")] += 2.0
    # Sandy soil: high luminance, light yellowish brown
    if luminance > 140 and r_mean > 130 and g_mean > 110:
        prior_bias[SOIL_CLASSES.index("Sandy Soil")] += 1.8
    # Loamy / Alluvial / Clay
    if 90 <= luminance <= 140:
        prior_bias[SOIL_CLASSES.index("Loamy Soil")] += 1.2
        prior_bias[SOIL_CLASSES.index("Alluvial Soil")] += 1.1

    adjusted_scores = probs * prior_bias
    adjusted_probs = adjusted_scores / np.sum(adjusted_scores)

    pred_idx = int(np.argmax(adjusted_probs))
    pred_soil_type = SOIL_CLASSES[pred_idx]
    pred_confidence = float(adjusted_probs[pred_idx])

    prob_dict = {
        cls_name: round(float(adjusted_probs[i]), 4)
        for i, cls_name in enumerate(SOIL_CLASSES)
    }

    # Physical properties
    raw_props = SOIL_METADATA.get(pred_soil_type, SOIL_METADATA["Loamy Soil"])
    physical_props = SoilPhysicalProperties(
        texture=raw_props["texture"],
        drainage=raw_props["drainage"],
        water_retention=raw_props["water_retention"],
        aeration=raw_props["aeration"],
        organic_matter_estimate=raw_props["organic_matter_estimate"],
    )

    # Health score and agronomic recommendations
    health_score, fertility_grade, chem_eval, recommendations = calculate_soil_health_score(
        soil_type=pred_soil_type,
        confidence=pred_confidence,
        soil_params=soil_params,
    )

    # Save to SQLite
    target_db = db_path or DB_PATH
    init_phase2_db(target_db)
    from datetime import datetime, timezone
    import json
    with sqlite3.connect(target_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO soil_health_predictions (
                request_id,
                predicted_at,
                soil_type,
                confidence,
                probabilities,
                soil_health_score,
                fertility_grade,
                ph_status,
                nitrogen_status,
                phosphorus_status,
                potassium_status,
                recommendations
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                datetime.now(timezone.utc).isoformat(),
                pred_soil_type,
                pred_confidence,
                json.dumps(prob_dict),
                health_score,
                fertility_grade,
                chem_eval.ph_status,
                chem_eval.nitrogen_status,
                chem_eval.phosphorus_status,
                chem_eval.potassium_status,
                json.dumps(recommendations),
            )
        )
        conn.commit()

    return SoilPredictionResponse(
        status="success",
        message="Soil health analysis completed successfully",
        request_id=request_id,
        soil_type=pred_soil_type,
        confidence=round(pred_confidence, 4),
        probabilities=prob_dict,
        soil_health_score=health_score,
        fertility_grade=fertility_grade,
        physical_properties=physical_props,
        chemical_assessment=chem_eval,
        recommendations=recommendations,
        ready_for_phase_3=True,
        next_step="Phase 3: Plant Health Detection",
    )
