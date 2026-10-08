"""Routes for image uploads and multi-modal input acquisition for Phase 1."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from backend.app.schemas.input_schema import (
    AnalysisResponse,
    AnalysisRecord,
    SingleImageUploadResponse,
    ImageMetadata,
    SoilParameters,
    EnvironmentalParameters,
    ValidationErrorDetail,
)
from backend.app.utils.image_utils import (
    validate_image_file,
    preprocess_image,
    save_uploaded_image,
    ImageValidationError,
)
from backend.app.services.validation import (
    generate_request_id,
    validate_soil_and_environment_inputs,
    save_analysis_record,
    get_analysis_record_by_request_id,
)
from backend.app.services.cloudinary_service import upload_to_cloudinary
from backend.app.services.mongo_service import save_analysis_record_to_mongo

router = APIRouter(prefix="/api", tags=["Input Acquisition"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
SOIL_UPLOADS_DIR = UPLOADS_DIR / "soil"
PLANT_UPLOADS_DIR = UPLOADS_DIR / "plant"


@router.post(
    "/soil/upload",
    response_model=SingleImageUploadResponse,
    responses={400: {"model": ValidationErrorDetail}, 422: {"model": ValidationErrorDetail}}
)
async def upload_soil_image(file: UploadFile = File(...)):
    """Upload and validate a soil image."""
    try:
        content = await file.read()
        pil_image = validate_image_file(content, file.filename or "soil_image.jpg", field_name="soil_image")
        
        # Test preprocessing pipeline for Phase 2 readiness
        preprocessed = preprocess_image(pil_image)
        
        # Save image
        saved_filename, full_path = save_uploaded_image(
            content,
            file.filename or "soil_image.jpg",
            str(SOIL_UPLOADS_DIR),
            prefix="soil"
        )
        
        metadata = ImageMetadata(
            filename=saved_filename,
            path=str(Path("uploads/soil") / saved_filename),
            content_type=file.content_type or "image/jpeg",
            size_bytes=len(content),
            dimensions=pil_image.size,
        )
        
        return SingleImageUploadResponse(
            status="success",
            message="Soil image uploaded and validated successfully.",
            image=metadata
        )
    except ImageValidationError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "message": exc.message, "errors": exc.errors}
        )
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": "Unexpected error processing soil image.", "errors": {"soil_image": str(exc)}}
        )


@router.post(
    "/plant/upload",
    response_model=SingleImageUploadResponse,
    responses={400: {"model": ValidationErrorDetail}, 422: {"model": ValidationErrorDetail}}
)
async def upload_plant_image(file: UploadFile = File(...)):
    """Upload and validate a plant/leaf image."""
    try:
        content = await file.read()
        pil_image = validate_image_file(content, file.filename or "plant_image.jpg", field_name="plant_image")
        
        # Test preprocessing pipeline for Phase 3 readiness
        preprocessed = preprocess_image(pil_image)
        
        # Save image
        saved_filename, full_path = save_uploaded_image(
            content,
            file.filename or "plant_image.jpg",
            str(PLANT_UPLOADS_DIR),
            prefix="plant"
        )
        
        metadata = ImageMetadata(
            filename=saved_filename,
            path=str(Path("uploads/plant") / saved_filename),
            content_type=file.content_type or "image/jpeg",
            size_bytes=len(content),
            dimensions=pil_image.size,
        )
        
        return SingleImageUploadResponse(
            status="success",
            message="Plant image uploaded and validated successfully.",
            image=metadata
        )
    except ImageValidationError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "message": exc.message, "errors": exc.errors}
        )
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": "Unexpected error processing plant image.", "errors": {"plant_image": str(exc)}}
        )


@router.post(
    "/analyze/input",
    response_model=AnalysisResponse,
    responses={400: {"model": ValidationErrorDetail}, 422: {"model": ValidationErrorDetail}}
)
async def analyze_input(
    soil_image: UploadFile = File(..., description="Soil image (JPG/PNG)"),
    plant_image: UploadFile = File(..., description="Plant or leaf image (JPG/PNG)"),
    ph: Optional[str] = Form(None, description="Soil pH (0.0 to 14.0)"),
    nitrogen: Optional[str] = Form(None, description="Nitrogen content N in mg/kg (>= 0)"),
    phosphorus: Optional[str] = Form(None, description="Phosphorus content P in mg/kg (>= 0)"),
    potassium: Optional[str] = Form(None, description="Potassium content K in mg/kg (>= 0)"),
    temperature: Optional[str] = Form(None, description="Temperature in °C (-10 to 60)"),
    humidity: Optional[str] = Form(None, description="Relative humidity % (0 to 100)"),
    rainfall: Optional[str] = Form(None, description="Rainfall in mm (>= 0)"),
):
    """Validate, preprocess, and persist Phase 1 inputs.
    
    Validates images, soil chemistry, and environmental parameters without running ML inference.
    Prepares input for Phase 2: Soil Health Detection.
    """
    validation_errors = {}

    # 1. Validate numerical inputs
    soil_params, env_params, num_errors = validate_soil_and_environment_inputs(
        ph_raw=ph,
        nitrogen_raw=nitrogen,
        phosphorus_raw=phosphorus,
        potassium_raw=potassium,
        temperature_raw=temperature,
        humidity_raw=humidity,
        rainfall_raw=rainfall,
    )
    validation_errors.update(num_errors)

    # 2. Validate Soil Image
    soil_bytes = b""
    soil_pil = None
    try:
        soil_bytes = await soil_image.read()
        soil_pil = validate_image_file(soil_bytes, soil_image.filename or "soil.jpg", field_name="soil_image")
    except ImageValidationError as exc:
        validation_errors.update(exc.errors)
    except Exception as exc:
        validation_errors["soil_image"] = f"Failed to read soil image: {str(exc)}"

    # 3. Validate Plant Image
    plant_bytes = b""
    plant_pil = None
    try:
        plant_bytes = await plant_image.read()
        plant_pil = validate_image_file(plant_bytes, plant_image.filename or "plant.jpg", field_name="plant_image")
    except ImageValidationError as exc:
        validation_errors.update(exc.errors)
    except Exception as exc:
        validation_errors["plant_image"] = f"Failed to read plant image: {str(exc)}"

    # If any validation errors exist, reject with 400 Bad Request
    if validation_errors:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "message": "Input validation failed. Please correct the highlighted errors.",
                "errors": validation_errors,
            }
        )

    # 4. Generate unique Request ID
    req_id = generate_request_id()

    # 5. Preprocess images (validating pipeline for Phase 2 & Phase 3)
    # Both arrays are verified to be shape (224, 224, 3)
    soil_preprocessed = preprocess_image(soil_pil)
    plant_preprocessed = preprocess_image(plant_pil)

    # 6. Save images safely
    soil_saved_name, soil_saved_path = save_uploaded_image(
        soil_bytes,
        soil_image.filename or "soil.jpg",
        str(SOIL_UPLOADS_DIR),
        prefix=req_id
    )
    plant_saved_name, plant_saved_path = save_uploaded_image(
        plant_bytes,
        plant_image.filename or "plant.jpg",
        str(PLANT_UPLOADS_DIR),
        prefix=req_id
    )

    soil_meta = ImageMetadata(
        filename=soil_saved_name,
        path=f"uploads/soil/{soil_saved_name}",
        content_type=soil_image.content_type or "image/jpeg",
        size_bytes=len(soil_bytes),
        dimensions=soil_pil.size,
    )

    plant_meta = ImageMetadata(
        filename=plant_saved_name,
        path=f"uploads/plant/{plant_saved_name}",
        content_type=plant_image.content_type or "image/jpeg",
        size_bytes=len(plant_bytes),
        dimensions=plant_pil.size,
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    record = AnalysisRecord(
        request_id=req_id,
        created_at=now_iso,
        soil_image=soil_meta,
        plant_image=plant_meta,
        soil_parameters=soil_params or SoilParameters(),
        environment=env_params or EnvironmentalParameters(),
    )

    # 7. Persist to SQLite
    save_analysis_record(record)

    # 8. Upload to Cloudinary CDN & Persist to MongoDB Atlas
    try:
        c_soil = upload_to_cloudinary(soil_bytes, folder="kisanseva/soil", public_id=f"soil_{req_id}")
        c_plant = upload_to_cloudinary(plant_bytes, folder="kisanseva/plant", public_id=f"plant_{req_id}")
        doc = record.model_dump()
        if c_soil and c_soil.get("secure_url"):
            doc["soil_image_cdn_url"] = c_soil["secure_url"]
        if c_plant and c_plant.get("secure_url"):
            doc["plant_image_cdn_url"] = c_plant["secure_url"]
        save_analysis_record_to_mongo(doc)
    except Exception:
        pass

    return AnalysisResponse(
        status="success",
        message="Input validation successful",
        request_id=req_id,
        data=record,
        ready_for_phase_2=True,
        next_step="Phase 2: Soil Health Detection",
    )


@router.get("/records/{request_id}")
async def get_record(request_id: str):
    """Retrieve details of an existing Phase 1 acquisition record."""
    row = get_analysis_record_by_request_id(request_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Record {request_id} not found.")
    return {"status": "success", "data": row}
