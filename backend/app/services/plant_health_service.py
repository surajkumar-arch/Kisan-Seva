"""Plant Health and Foliage Disease Detection Service for Phase 3."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.app.models.plant_net import (
    PLANT_DISEASE_CLASSES,
    PLANT_DISEASE_METADATA,
    get_plant_model,
)
from backend.app.schemas.plant_schema import PlantPredictionResponse
from backend.app.utils.image_utils import preprocess_image

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = BASE_DIR / "data" / "smart_agriculture.db"


def init_phase3_db(db_path: Optional[Path] = None) -> None:
    """Ensure plant_health_predictions table exists in SQLite database."""
    target_path = db_path or DB_PATH
    os.makedirs(target_path.parent, exist_ok=True)

    with sqlite3.connect(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS plant_health_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                predicted_at TEXT NOT NULL,
                condition TEXT NOT NULL,
                is_healthy INTEGER NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT NOT NULL,
                pathogen_type TEXT NOT NULL,
                severity_level TEXT NOT NULL,
                plant_health_score INTEGER NOT NULL,
                symptoms TEXT NOT NULL,
                organic_treatment TEXT NOT NULL,
                chemical_treatment TEXT NOT NULL,
                preventive_measures TEXT NOT NULL,
                FOREIGN KEY (request_id) REFERENCES analysis_records(request_id)
            );
            """
        )
        conn.commit()


def calculate_plant_health_score(condition: str, confidence: float) -> Tuple[int, str]:
    """Calculate the Plant Foliage Health Score (0-100) and severity rating."""
    if condition == "Healthy Foliage":
        # Disease-free leaf maintains 90-100 health score
        score = int(85 + 15 * confidence)
        severity = "None"
    elif condition in ["Early Blight", "Powdery Mildew", "Leaf Rust"]:
        # Moderate pathogens
        score = max(35, int(65 - 25 * confidence))
        severity = "Moderate"
    else:
        # Aggressive pathogens (Late Blight, Bacterial Spot, Yellow Leaf Curl)
        score = max(15, int(45 - 30 * confidence))
        severity = "Severe"

    return min(100, max(10, score)), severity


def predict_plant_health(
    plant_image: Image.Image,
    request_id: str,
    db_path: Optional[Path] = None,
) -> PlantPredictionResponse:
    """Execute MobileNetV3-Large forward pass and generate plant disease pathology report."""
    model = get_plant_model()

    # 1. Preprocess with ImageNet standardization
    img_array = preprocess_image(plant_image, target_size=(224, 224), normalize_mode="imagenet")

    # 2. Convert to PyTorch Tensor: shape (1, 3, 224, 224)
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()

    # 3. Model forward pass
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()

    # Domain visual heuristics for robust out-of-the-box plant leaf inspection
    rgb = np.asarray(plant_image.convert("RGB").resize((64, 64)), dtype=np.float32)
    r_mean = float(np.mean(rgb[:, :, 0]))
    g_mean = float(np.mean(rgb[:, :, 1]))
    b_mean = float(np.mean(rgb[:, :, 2]))

    prior_bias = np.ones(len(PLANT_DISEASE_CLASSES), dtype=np.float32)

    # High green dominance with low red/blue -> Healthy foliage
    if g_mean > r_mean * 1.2 and g_mean > b_mean * 1.2:
        prior_bias[PLANT_DISEASE_CLASSES.index("Healthy Foliage")] += 2.0
    # Yellowish leaf chlorosis (High R and G, lower B) -> Yellow Leaf Curl or Rust
    elif r_mean > 130 and g_mean > 130 and b_mean < 90:
        prior_bias[PLANT_DISEASE_CLASSES.index("Yellow Leaf Curl")] += 1.8
        prior_bias[PLANT_DISEASE_CLASSES.index("Leaf Rust")] += 1.4
    # Pale powdery light tones (High across R, G, B with slight grayish cast) -> Powdery Mildew
    elif r_mean > 160 and g_mean > 160 and b_mean > 160:
        prior_bias[PLANT_DISEASE_CLASSES.index("Powdery Mildew")] += 2.2
    # Dark necrotic spots with reddish/brownish tone -> Early or Late Blight
    elif r_mean > g_mean and r_mean > b_mean:
        prior_bias[PLANT_DISEASE_CLASSES.index("Early Blight")] += 1.6
        prior_bias[PLANT_DISEASE_CLASSES.index("Late Blight")] += 1.5

    adjusted_scores = probs * prior_bias
    adjusted_probs = adjusted_scores / np.sum(adjusted_scores)

    pred_idx = int(np.argmax(adjusted_probs))
    pred_condition = PLANT_DISEASE_CLASSES[pred_idx]
    pred_confidence = float(adjusted_probs[pred_idx])

    prob_dict = {
        cls_name: round(float(adjusted_probs[i]), 4)
        for i, cls_name in enumerate(PLANT_DISEASE_CLASSES)
    }

    meta = PLANT_DISEASE_METADATA.get(pred_condition, PLANT_DISEASE_METADATA["Healthy Foliage"])
    is_healthy = pred_condition == "Healthy Foliage"
    health_score, severity = calculate_plant_health_score(pred_condition, pred_confidence)

    # 4. Save to SQLite database
    target_db = db_path or DB_PATH
    init_phase3_db(target_db)
    with sqlite3.connect(target_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO plant_health_predictions (
                request_id,
                predicted_at,
                condition,
                is_healthy,
                confidence,
                probabilities,
                pathogen_type,
                severity_level,
                plant_health_score,
                symptoms,
                organic_treatment,
                chemical_treatment,
                preventive_measures
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                datetime.now(timezone.utc).isoformat(),
                pred_condition,
                1 if is_healthy else 0,
                round(pred_confidence, 4),
                json.dumps(prob_dict),
                meta["pathogen_type"],
                severity,
                health_score,
                meta["symptoms"],
                json.dumps(meta["organic_treatment"]),
                json.dumps(meta["chemical_treatment"]),
                json.dumps(meta["preventive_measures"]),
            )
        )
        conn.commit()

    return PlantPredictionResponse(
        status="success",
        message="Plant health analysis completed successfully",
        request_id=request_id,
        condition=pred_condition,
        is_healthy=is_healthy,
        confidence=round(pred_confidence, 4),
        probabilities=prob_dict,
        pathogen_type=meta["pathogen_type"],
        severity_level=severity,
        plant_health_score=health_score,
        symptoms=meta["symptoms"],
        organic_treatment=meta["organic_treatment"],
        chemical_treatment=meta["chemical_treatment"],
        preventive_measures=meta["preventive_measures"],
        ready_for_phase_4=True,
        next_step="Phase 4: Feature Fusion",
    )
