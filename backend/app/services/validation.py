"""Validation and database persistence services for Phase 1."""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from backend.app.schemas.input_schema import (
    SoilParameters,
    EnvironmentalParameters,
    ImageMetadata,
    AnalysisRecord,
)

# SQLite database file path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DB_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "smart_agriculture.db"


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize the SQLite database and create tables if not present."""
    target_path = db_path or DB_PATH
    os.makedirs(target_path.parent, exist_ok=True)

    with sqlite3.connect(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                soil_image_filename TEXT NOT NULL,
                soil_image_path TEXT NOT NULL,
                soil_image_size INTEGER NOT NULL,
                soil_image_width INTEGER NOT NULL,
                soil_image_height INTEGER NOT NULL,
                plant_image_filename TEXT NOT NULL,
                plant_image_path TEXT NOT NULL,
                plant_image_size INTEGER NOT NULL,
                plant_image_width INTEGER NOT NULL,
                plant_image_height INTEGER NOT NULL,
                ph REAL,
                nitrogen REAL,
                phosphorus REAL,
                potassium REAL,
                temperature REAL,
                humidity REAL,
                rainfall REAL,
                status TEXT NOT NULL DEFAULT 'VALIDATED'
            );
            """
        )
        conn.commit()


def generate_request_id() -> str:
    """Generate a clean, readable unique request ID (e.g. REQ-A1B2C3D4)."""
    uid = uuid.uuid4().hex[:8].upper()
    return f"REQ-{uid}"


def validate_soil_and_environment_inputs(
    ph_raw: Optional[Any] = None,
    nitrogen_raw: Optional[Any] = None,
    phosphorus_raw: Optional[Any] = None,
    potassium_raw: Optional[Any] = None,
    temperature_raw: Optional[Any] = None,
    humidity_raw: Optional[Any] = None,
    rainfall_raw: Optional[Any] = None,
) -> Tuple[Optional[SoilParameters], Optional[EnvironmentalParameters], Dict[str, str]]:
    """Validate numerical inputs from raw strings or numbers.

    Returns:
        (SoilParameters, EnvironmentalParameters, errors_dict)
    """
    errors: Dict[str, str] = {}

    def parse_float(val: Any, field_name: str) -> Optional[float]:
        if val is None or val == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            errors[field_name] = f"Invalid number format for {field_name}."
            return None

    ph = parse_float(ph_raw, "ph")
    nitrogen = parse_float(nitrogen_raw, "nitrogen")
    phosphorus = parse_float(phosphorus_raw, "phosphorus")
    potassium = parse_float(potassium_raw, "potassium")
    temperature = parse_float(temperature_raw, "temperature")
    humidity = parse_float(humidity_raw, "humidity")
    rainfall = parse_float(rainfall_raw, "rainfall")

    # Range and domain validations
    if ph is not None:
        if ph < 0.0 or ph > 14.0:
            errors["ph"] = "pH must be between 0 and 14."

    if nitrogen is not None:
        if nitrogen < 0.0:
            errors["nitrogen"] = "Nitrogen content must be greater than or equal to 0."

    if phosphorus is not None:
        if phosphorus < 0.0:
            errors["phosphorus"] = "Phosphorus content must be greater than or equal to 0."

    if potassium is not None:
        if potassium < 0.0:
            errors["potassium"] = "Potassium content must be greater than or equal to 0."

    if humidity is not None:
        if humidity < 0.0 or humidity > 100.0:
            errors["humidity"] = "Humidity must be between 0 and 100."

    if temperature is not None:
        if temperature < -10.0 or temperature > 60.0:
            errors["temperature"] = "Temperature must be between -10°C and 60°C."

    if rainfall is not None:
        if rainfall < 0.0:
            errors["rainfall"] = "Rainfall must be greater than or equal to 0."

    soil_params = SoilParameters(
        ph=ph,
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
    ) if not any(k in errors for k in ["ph", "nitrogen", "phosphorus", "potassium"]) else None

    env_params = EnvironmentalParameters(
        temperature=temperature,
        humidity=humidity,
        rainfall=rainfall,
    ) if not any(k in errors for k in ["temperature", "humidity", "rainfall"]) else None

    return soil_params, env_params, errors


def save_analysis_record(
    record: AnalysisRecord,
    db_path: Optional[Path] = None
) -> None:
    """Store complete Phase 1 analysis record into SQLite."""
    target_path = db_path or DB_PATH
    init_db(target_path)

    with sqlite3.connect(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO analysis_records (
                request_id,
                created_at,
                soil_image_filename,
                soil_image_path,
                soil_image_size,
                soil_image_width,
                soil_image_height,
                plant_image_filename,
                plant_image_path,
                plant_image_size,
                plant_image_width,
                plant_image_height,
                ph,
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                rainfall,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.request_id,
                record.created_at,
                record.soil_image.filename,
                record.soil_image.path,
                record.soil_image.size_bytes,
                record.soil_image.dimensions[0],
                record.soil_image.dimensions[1],
                record.plant_image.filename,
                record.plant_image.path,
                record.plant_image.size_bytes,
                record.plant_image.dimensions[0],
                record.plant_image.dimensions[1],
                record.soil_parameters.ph,
                record.soil_parameters.nitrogen,
                record.soil_parameters.phosphorus,
                record.soil_parameters.potassium,
                record.environment.temperature,
                record.environment.humidity,
                record.environment.rainfall,
                "READY_FOR_PHASE_2",
            )
        )
        conn.commit()


def get_analysis_record_by_request_id(
    request_id: str,
    db_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Retrieve an analysis record from SQLite by request_id."""
    target_path = db_path or DB_PATH
    if not target_path.exists():
        return None

    with sqlite3.connect(target_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_records WHERE request_id = ?", (request_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)
