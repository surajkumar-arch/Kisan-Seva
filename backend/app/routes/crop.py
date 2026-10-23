"""API routes for Phase 5: Precision Crop Recommendation."""

import json
import sqlite3
from pathlib import Path
from fastapi import APIRouter, Form, HTTPException, status
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field
from backend.app.schemas.input_schema import ValidationErrorDetail
from backend.app.schemas.crop_schema import CropRecommendationResponse
from backend.app.models.crop_net import CROP_CATALOG, CROPS_LIST
from backend.app.services.crop_service import (
    generate_crop_recommendations,
    recommend_crops_direct,
    DB_PATH,
)
from typing import Optional
from backend.app.services.mongo_service import save_crop_recommendation_to_mongo
from backend.app.services.realtime_weather_crop_service import realtime_weather_crop_service

router = APIRouter(prefix="/api/crop", tags=["Crop Recommendation"])


class DirectCropRequest(BaseModel):
    ph: float = Field(6.5, ge=0.0, le=14.0, description="Soil pH value")
    nitrogen: float = Field(140.0, ge=0.0, le=300.0, description="Soil Nitrogen (N) kg/ha")
    phosphorus: float = Field(35.0, ge=0.0, le=150.0, description="Soil Phosphorus (P) kg/ha")
    potassium: float = Field(150.0, ge=0.0, le=350.0, description="Soil Potassium (K) kg/ha")
    temperature: float = Field(26.0, ge=-10.0, le=60.0, description="Mean temperature in Celsius")
    humidity: float = Field(65.0, ge=0.0, le=100.0, description="Relative humidity percentage")
    rainfall: float = Field(120.0, ge=0.0, le=500.0, description="Annual/seasonal rainfall in mm")
    soil_type: str = Field(default="Loamy Soil", description="Optional soil classification")


@router.get("/catalog")
async def get_crop_catalog():
    """Retrieve full catalog of 12 supported crops and their agronomic profiles."""
    crop_details = []
    for crop_name in CROPS_LIST:
        profile = CROP_CATALOG.get(crop_name, {})
        crop_details.append({"crop_name": crop_name, **profile})

    return {
        "status": "success",
        "total_crops": len(CROPS_LIST),
        "crops": CROPS_LIST,
        "crop_details": crop_details,
        "catalog": CROP_CATALOG,
    }


@router.post("/predict-direct")
async def direct_crop_recommendation_endpoint(payload: DirectCropRequest):
    """Directly evaluate soil nutrients & meteorological conditions to rank optimal crops."""
    result = recommend_crops_direct(
        ph=payload.ph,
        nitrogen=payload.nitrogen,
        phosphorus=payload.phosphorus,
        potassium=payload.potassium,
        temperature=payload.temperature,
        humidity=payload.humidity,
        rainfall=payload.rainfall,
        soil_type=payload.soil_type,
    )
    # Persist to MongoDB Atlas
    save_crop_recommendation_to_mongo(result)
    return result


@router.post(
    "/recommend",
    response_model=CropRecommendationResponse,
    responses={400: {"model": ValidationErrorDetail}, 404: {"description": "Record not found"}}
)
async def recommend_crop_endpoint(
    request_id: str = Form(..., description="Unique request ID from Phase 1")
):
    """Generate precision ranked crop recommendations from fused multi-modal latent features."""
    clean_id = request_id.strip()
    if not clean_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "message": "request_id is required", "errors": {"request_id": "Field is required."}}
        )

    try:
        response = generate_crop_recommendations(clean_id)
        # Persist to MongoDB Atlas
        save_crop_recommendation_to_mongo(response.model_dump())
        return response
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Crop recommendation pipeline failed: {str(exc)}"
        )


@router.get("/history/{request_id}")
async def get_crop_history(request_id: str):
    """Retrieve historical crop recommendations for a request_id."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail="No database records found.")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM crop_recommendations WHERE request_id = ?", (request_id,))
        rows = cursor.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No crop recommendations found for request ID {request_id}.")

    results = []
    for row in rows:
        d = dict(row)
        d["top_crops"] = json.loads(d["top_crops_json"])
        d["soil_context"] = json.loads(d["soil_context_json"])
        d["climate_context"] = json.loads(d["climate_context_json"])
        del d["top_crops_json"]
        del d["soil_context_json"]
        del d["climate_context_json"]
        results.append(d)

    return {"status": "success", "data": results}


class RealtimeWeatherCropRequest(BaseModel):
    ph: float = Field(6.8, ge=0.0, le=14.0, description="Soil pH value")
    nitrogen: float = Field(120.0, ge=0.0, le=300.0, description="Soil Nitrogen (N) kg/ha")
    phosphorus: float = Field(35.0, ge=0.0, le=150.0, description="Soil Phosphorus (P) kg/ha")
    potassium: float = Field(110.0, ge=0.0, le=350.0, description="Soil Potassium (K) kg/ha")
    soil_type: str = Field("Alluvial Soil", description="Soil classification")
    location_name: Optional[str] = Field("Ludhiana, Punjab", description="District or farming hub name")
    lat: Optional[float] = Field(None, description="GPS Latitude")
    lon: Optional[float] = Field(None, description="GPS Longitude")
    override_temperature: Optional[float] = Field(None, description="Manual override for temp")
    override_rainfall: Optional[float] = Field(None, description="Manual override for rain")


@router.get("/live-weather-telemetry")
def get_live_weather_telemetry(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    location: Optional[str] = "Ludhiana"
):
    """Retrieve real-time meteorological conditions from live NWP radar."""
    return realtime_weather_crop_service.fetch_live_weather(lat=lat, lon=lon, location_name=location)


@router.post("/predict-with-live-weather")
def predict_with_live_weather_endpoint(req: RealtimeWeatherCropRequest):
    """
    Evaluate crop suitability by fetching real-time meteorological conditions
    and adapting neural rankings with climate sensitivity factors.
    """
    return realtime_weather_crop_service.predict_crops_with_live_weather(
        ph=req.ph,
        nitrogen=req.nitrogen,
        phosphorus=req.phosphorus,
        potassium=req.potassium,
        soil_type=req.soil_type,
        lat=req.lat,
        lon=req.lon,
        location_name=req.location_name,
        override_temperature=req.override_temperature,
        override_rainfall=req.override_rainfall
    )


@router.get("/demo-live-weather")
def get_demo_live_weather_crop(district: str = "ludhiana"):
    """
    Instant benchmark demo evaluating real-time weather crop recommendations.
    """
    return realtime_weather_crop_service.predict_crops_with_live_weather(
        ph=6.8,
        nitrogen=130.0,
        phosphorus=38.0,
        potassium=115.0,
        soil_type="Alluvial Soil",
        location_name=district
    )

