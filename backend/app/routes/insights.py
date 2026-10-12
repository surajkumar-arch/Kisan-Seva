"""Explainability (Grad-CAM), Multi-Agent, AI Agronomist, Weather & PDF Endpoints."""

import io
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Response
from PIL import Image
from pydantic import BaseModel, Field

from backend.app.services.agronomy_agents import run_agents
from backend.app.services.explain_service import explain_image
from backend.app.services.enterprise_service import (
    ask_agronomist_ai,
    get_hyperlocal_weather_forecast,
    generate_agronomic_pdf_prescription,
    run_yolov8_lesion_detection
)

router = APIRouter(prefix="/api", tags=["enterprise-features"])

MAX_UPLOAD_BYTES = 12 * 1024 * 1024


@router.post("/explain/{kind}")
async def explain(kind: str, image: UploadFile = File(...), top_k: int = Form(3)):
    """Grad-CAM heatmap + top-k predictions. kind = plant | soil."""
    if kind not in ("plant", "soil"):
        raise HTTPException(status_code=404, detail="kind must be 'plant' or 'soil'")
    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image larger than 12 MB")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return explain_image(img, kind, top_k=max(1, min(top_k, 10)))


class AdviseRequest(BaseModel):
    soil_type: Optional[str] = None
    ph: Optional[float] = Field(None, ge=0, le=14)
    nitrogen: Optional[float] = Field(None, ge=0)
    phosphorus: Optional[float] = Field(None, ge=0)
    potassium: Optional[float] = Field(None, ge=0)
    condition: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    temperature_c: Optional[float] = None
    humidity: Optional[float] = Field(None, ge=0, le=100)
    rainfall_mm: Optional[float] = Field(None, ge=0)


@router.post("/agents/advise")
async def advise(req: AdviseRequest):
    """Run the soil, disease and weather agents and merge into a prioritised plan."""
    return run_agents(**req.model_dump())


# 1. AI Conversational Chat Endpoint
class AIChatRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    language: Optional[str] = "English"

@router.post("/ai/chat")
async def ai_chat(req: AIChatRequest):
    """Conversational agronomist answering farmer queries."""
    return ask_agronomist_ai(req.query, req.context, req.language or "English")


# 2. Live Microclimate & Weather Outbreak Endpoint
@router.get("/weather/telemetry")
async def weather_telemetry(lat: float = 20.59, lon: float = 78.96, city: Optional[str] = None):
    """Get live weather and disease outbreak probability."""
    return get_hyperlocal_weather_forecast(lat, lon, city)


# 3. YOLOv8 Edge Lesion Detection Endpoint
@router.post("/vision/yolo-detect")
async def yolo_detect(image: UploadFile = File(...)):
    """Run YOLOv8 object detection on leaf."""
    data = await image.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return run_yolov8_lesion_detection(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 4. PDF Health Card Prescription Download
class PDFRequest(BaseModel):
    request_id: str
    patient_crop: Optional[str] = "Farm Foliage"
    diagnosis_condition: str
    confidence: float
    pathogen: str
    severity: str
    health_score: int
    organic_rx: List[str]
    chemical_rx: List[str]
    soil_type: Optional[str] = "Loamy Soil"

@router.post("/prescription/pdf")
async def download_pdf(req: PDFRequest):
    """Generate high-resolution printable PDF report."""
    pdf_bytes = generate_agronomic_pdf_prescription(
        request_id=req.request_id,
        patient_crop=req.patient_crop or "Farm Foliage",
        diagnosis_condition=req.diagnosis_condition,
        confidence=req.confidence,
        pathogen=req.pathogen,
        severity=req.severity,
        health_score=req.health_score,
        organic_rx=req.organic_rx,
        chemical_rx=req.chemical_rx,
        soil_type=req.soil_type
    )
    return Response(content=pdf_bytes, media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=KisanSeva_Rx_{req.request_id}.pdf"
    })
