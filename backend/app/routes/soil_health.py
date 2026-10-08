"""API routes for Phase 2: Soil Health Detection."""

import io
import json
import sqlite3
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image

from backend.app.schemas.input_schema import SoilParameters, ValidationErrorDetail
from backend.app.schemas.soil_schema import (
    SoilPredictionResponse,
    SoilPhysicalProperties,
)
from backend.app.models.soil_net import SOIL_CLASSES, SOIL_METADATA
from backend.app.utils.image_utils import (
    validate_image_file,
    ImageValidationError,
)
from backend.app.services.validation import (
    get_analysis_record_by_request_id,
    validate_soil_and_environment_inputs,
    generate_request_id,
)
from backend.app.services.soil_health_service import (
    predict_soil_health,
    DB_PATH,
)

router = APIRouter(prefix="/api/soil", tags=["Soil Health Detection"])
BASE_DIR = Path(__file__).resolve().parent.parent.parent


@router.get("/classes")
async def get_soil_classes():
    """Retrieve list of supported soil classes and their physical profiles."""
    return {
        "status": "success",
        "classes": SOIL_CLASSES,
        "metadata": SOIL_METADATA,
    }


@router.post(
    "/predict",
    response_model=SoilPredictionResponse,
    responses={400: {"model": ValidationErrorDetail}, 404: {"description": "Record not found"}}
)
async def predict_soil_health_endpoint(
    request_id: Optional[str] = Form(None, description="Optional Phase 1 Request ID"),
    soil_image: Optional[UploadFile] = File(None, description="Optional direct soil image upload"),
    ph: Optional[str] = Form(None, description="Optional soil pH"),
    nitrogen: Optional[str] = Form(None, description="Optional Nitrogen (mg/kg)"),
    phosphorus: Optional[str] = Form(None, description="Optional Phosphorus (mg/kg)"),
    potassium: Optional[str] = Form(None, description="Optional Potassium (mg/kg)"),
):
    """Predict soil type, assess fertility, and generate soil health recommendations.

    Can consume either an existing Phase 1 request_id or a direct image upload.
    """
    image_obj: Optional[Image.Image] = None
    target_request_id: str = ""
    effective_soil_params: Optional[SoilParameters] = None

    # Parse provided optional parameters
    soil_params, _, val_errors = validate_soil_and_environment_inputs(
        ph_raw=ph,
        nitrogen_raw=nitrogen,
        phosphorus_raw=phosphorus,
        potassium_raw=potassium,
    )
    if val_errors:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "message": "Invalid soil parameter values", "errors": val_errors}
        )
    effective_soil_params = soil_params

    # Case 1: Existing request_id provided
    if request_id:
        target_request_id = request_id.strip()
        record = get_analysis_record_by_request_id(target_request_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Phase 1 record with Request ID '{target_request_id}' was not found."
            )
        
        # Load saved soil image
        saved_rel_path = record["soil_image_path"]
        full_image_path = BASE_DIR / saved_rel_path
        if not full_image_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Soil image file for {target_request_id} is missing from disk ({saved_rel_path})."
            )

        try:
            image_obj = Image.open(full_image_path).convert("RGB")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read stored soil image: {str(exc)}"
            )

        # Merge stored parameters if not overridden
        if effective_soil_params is None:
            effective_soil_params = SoilParameters(
                ph=record.get("ph"),
                nitrogen=record.get("nitrogen"),
                phosphorus=record.get("phosphorus"),
                potassium=record.get("potassium"),
            )
        else:
            # Fill missing with stored record
            if effective_soil_params.ph is None and record.get("ph") is not None:
                effective_soil_params.ph = record["ph"]
            if effective_soil_params.nitrogen is None and record.get("nitrogen") is not None:
                effective_soil_params.nitrogen = record["nitrogen"]
            if effective_soil_params.phosphorus is None and record.get("phosphorus") is not None:
                effective_soil_params.phosphorus = record["phosphorus"]
            if effective_soil_params.potassium is None and record.get("potassium") is not None:
                effective_soil_params.potassium = record["potassium"]

    # Case 2: Direct soil_image provided
    elif soil_image:
        target_request_id = generate_request_id()
        try:
            file_bytes = await soil_image.read()
            image_obj = validate_image_file(file_bytes, soil_image.filename or "soil.jpg", field_name="soil_image")
        except ImageValidationError as exc:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "message": exc.message, "errors": exc.errors}
            )
        except Exception as exc:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "message": f"Failed to parse uploaded soil image: {str(exc)}", "errors": {"soil_image": str(exc)}}
            )

    else:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "message": "Either 'request_id' from Phase 1 or a 'soil_image' file must be provided.",
                "errors": {"input": "Missing request_id or soil_image"}
            }
        )

    # Execute prediction & evaluation
    result = predict_soil_health(
        soil_image=image_obj,
        request_id=target_request_id,
        soil_params=effective_soil_params,
    )
    return result


@router.get("/history/{request_id}")
async def get_soil_history(request_id: str):
    """Retrieve historical soil health predictions for a request_id."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail="No database records found.")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM soil_health_predictions WHERE request_id = ?", (request_id,))
        rows = cursor.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No soil health predictions found for request ID {request_id}.")

    results = []
    for row in rows:
        d = dict(row)
        d["probabilities"] = json.loads(d["probabilities"])
        d["recommendations"] = json.loads(d["recommendations"])
        results.append(d)

    return {"status": "success", "data": results}
