"""API routes for Phase 3: Plant Health & Foliage Disease Detection."""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image

from backend.app.schemas.input_schema import ValidationErrorDetail
from backend.app.schemas.plant_schema import PlantPredictionResponse
from backend.app.models.plant_net import PLANT_DISEASE_CLASSES, PLANT_DISEASE_METADATA
from backend.app.utils.image_utils import validate_image_file, ImageValidationError
from backend.app.services.validation import (
    get_analysis_record_by_request_id,
    generate_request_id,
)
from backend.app.services.plant_health_service import (
    predict_plant_health,
    DB_PATH,
)
from backend.app.services.cloudinary_service import upload_to_cloudinary
from backend.app.services.mongo_service import save_plant_diagnosis_to_mongo

router = APIRouter(prefix="/api/plant", tags=["Plant Health Detection"])
BASE_DIR = Path(__file__).resolve().parent.parent.parent


@router.get("/classes")
async def get_plant_disease_classes():
    """Retrieve list of supported plant foliage disease classes and treatment metadata."""
    return {
        "status": "success",
        "classes": PLANT_DISEASE_CLASSES,
        "metadata": PLANT_DISEASE_METADATA,
    }


@router.post(
    "/predict",
    response_model=PlantPredictionResponse,
    responses={400: {"model": ValidationErrorDetail}, 404: {"description": "Record not found"}}
)
async def predict_plant_health_endpoint(
    request_id: Optional[str] = Form(None, description="Optional Phase 1 Request ID"),
    plant_image: Optional[UploadFile] = File(None, description="Optional direct plant image upload"),
):
    """Diagnose leaf disease, identify pathogen, evaluate severity, and prescribe agronomic treatments.

    Can consume either an existing Phase 1 request_id or a direct image upload.
    """
    image_obj: Optional[Image.Image] = None
    target_request_id: str = ""
    cloudinary_url: Optional[str] = None

    # Case 1: Existing request_id provided
    if request_id:
        target_request_id = request_id.strip()
        record = get_analysis_record_by_request_id(target_request_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Phase 1 record with Request ID '{target_request_id}' was not found."
            )

        saved_rel_path = record["plant_image_path"]
        full_image_path = BASE_DIR / saved_rel_path
        if not full_image_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Plant image file for {target_request_id} is missing from disk ({saved_rel_path})."
            )

        try:
            image_obj = Image.open(full_image_path).convert("RGB")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read stored plant image: {str(exc)}"
            )

    # Case 2: Direct plant_image uploaded
    elif plant_image:
        target_request_id = generate_request_id()
        try:
            file_bytes = await plant_image.read()
            image_obj = validate_image_file(file_bytes, plant_image.filename or "plant.jpg", field_name="plant_image")
            
            # Upload to Cloudinary CDN
            c_res = upload_to_cloudinary(file_bytes, folder="kisanseva/plant", public_id=f"leaf_{target_request_id}")
            if c_res and c_res.get("secure_url"):
                cloudinary_url = c_res.get("secure_url")
        except ImageValidationError as exc:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "message": exc.message, "errors": exc.errors}
            )
        except Exception as exc:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "message": f"Failed to parse uploaded plant image: {str(exc)}", "errors": {"plant_image": str(exc)}}
            )

    else:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "message": "Either 'request_id' from Phase 1 or a 'plant_image' file must be provided.",
                "errors": {"input": "Missing request_id or plant_image"}
            }
        )

    # Execute MobileNetV3 inference
    result = predict_plant_health(
        plant_image=image_obj,
        request_id=target_request_id,
    )
    if cloudinary_url:
        result.image_url = cloudinary_url

    # Persist diagnosis document to MongoDB Atlas
    save_plant_diagnosis_to_mongo(result.model_dump())

    return result


@router.get("/history/{request_id}")
async def get_plant_history(request_id: str):
    """Retrieve historical plant health diagnoses for a request_id."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail="No database records found.")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM plant_health_predictions WHERE request_id = ?", (request_id,))
        rows = cursor.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No plant health predictions found for request ID {request_id}.")

    results = []
    for row in rows:
        d = dict(row)
        d["probabilities"] = json.loads(d["probabilities"])
        d["organic_treatment"] = json.loads(d["organic_treatment"])
        d["chemical_treatment"] = json.loads(d["chemical_treatment"])
        d["preventive_measures"] = json.loads(d["preventive_measures"])
        results.append(d)

    return {"status": "success", "data": results}
