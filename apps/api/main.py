"""EDAPHNET Production REST API Gateway.

Built with FastAPI, featuring:
- Strict OpenAPI documentation & typed schemas
- Correlation Request-IDs and structured logging
- Input validation & Image Quality Gates
- Model Registry catalog endpoints
- Dedicated Soil, Foliar Disease, YOLO Localization, U-Net Segmentation & Recommendation routes
- Health, Readiness & Version probes for Kubernetes / Docker Compose
"""

from __future__ import annotations

import io
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from src.edaphnet.application.pipeline_service import EdaphNetPipelineService
from src.edaphnet.config.settings import settings
from src.edaphnet.domain.models import (
    CropRecommendation,
    DiseasePrediction,
    InferenceResponse,
    LocalizationResult,
    ModelMetadata,
    SegmentationResult,
    SoilPrediction,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("edaphnet.api")

app = FastAPI(
    title="EDAPHNET Agricultural Intelligence API",
    description="Enterprise dual-vision & agronomic recommendation system with calibrated uncertainty and explainable AI.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_hosts,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline Service Singleton
pipeline_service = EdaphNetPipelineService()


# -------------------------------------------------------------------------
# MIDDLEWARE
# -------------------------------------------------------------------------
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Enforce request correlation ID and log structured latency."""
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()
    
    # Attach to request state
    request.state.request_id = req_id

    response: Response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000.0

    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
    
    logger.info(
        "Handled %s %s [Status: %d] Latency: %.2fms (Request-ID: %s)",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        req_id,
    )
    return response


# -------------------------------------------------------------------------
# HEALTH & PROBES
# -------------------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe confirming API service availability."""
    return {"status": "healthy", "service": "edaphnet-api", "timestamp": time.time()}


@app.get("/ready", tags=["System"])
async def readiness_check():
    """Readiness probe confirming ML models and registries are loaded."""
    ready = pipeline_service is not None and pipeline_service.registry is not None
    if not ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model services not ready")
    return {
        "status": "ready",
        "models_loaded": len(pipeline_service.registry.list_models()),
        "environment": settings.environment,
    }


@app.get("/version", tags=["System"])
async def version_info():
    """Version metadata and active deployment configuration."""
    return {
        "version": "1.0.0",
        "api_name": "EDAPHNET",
        "framework": "PyTorch / FastAPI",
        "environment": settings.environment,
    }


# -------------------------------------------------------------------------
# UNIFIED MULTI-MODAL ANALYSIS
# -------------------------------------------------------------------------
@app.post("/api/v1/analyze", response_model=InferenceResponse, tags=["Inference"])
async def analyze_multimodal(
    request: Request,
    soil_image: Optional[UploadFile] = File(None, description="Soil photograph (JPEG/PNG)"),
    leaf_image: Optional[UploadFile] = File(None, description="Plant foliage photograph (JPEG/PNG)"),
    n: Optional[float] = Form(None, description="Soil Nitrogen (mg/kg or kg/ha)"),
    p: Optional[float] = Form(None, description="Soil Phosphorus (mg/kg or kg/ha)"),
    k: Optional[float] = Form(None, description="Soil Potassium (mg/kg or kg/ha)"),
    ph: Optional[float] = Form(None, description="Soil pH value (0 - 14)"),
    temperature: Optional[float] = Form(None, description="Ambient temperature (°C)"),
    humidity: Optional[float] = Form(None, description="Relative humidity (%)"),
    rainfall: Optional[float] = Form(None, description="Annual/Seasonal rainfall (mm)"),
    include_explainability: bool = Form(True, description="Compute Grad-CAM & SHAP attributions"),
    include_segmentation: bool = Form(True, description="Compute U-Net lesion mask"),
    include_localization: bool = Form(True, description="Compute YOLO lesion bounding boxes"),
):
    """Run complete end-to-end multi-modal vision and agronomic recommendation pipeline."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    soil_bytes = await soil_image.read() if soil_image else None
    leaf_bytes = await leaf_image.read() if leaf_image else None

    # Validate file sizes
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if soil_bytes and len(soil_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Soil image exceeds {settings.max_upload_size_mb}MB limit")
    if leaf_bytes and len(leaf_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Leaf image exceeds {settings.max_upload_size_mb}MB limit")

    agronomic_features: Dict[str, Any] = {
        "N": n,
        "P": p,
        "K": k,
        "ph": ph,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
    }

    try:
        response = pipeline_service.analyze(
            soil_image_bytes=soil_bytes,
            leaf_image_bytes=leaf_bytes,
            agronomic_features=agronomic_features,
            request_id=req_id,
            include_explainability=include_explainability,
            include_segmentation=include_segmentation,
            include_localization=include_localization,
        )
        return response
    except Exception as exc:
        logger.exception("Pipeline analysis failure: %s", exc)
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(exc)}")


# -------------------------------------------------------------------------
# DEDICATED SOIL CLASSIFIER ROUTE
# -------------------------------------------------------------------------
@app.post("/api/v1/soil/predict", response_model=SoilPrediction, tags=["Soil"])
async def predict_soil(file: UploadFile = File(...)):
    """Classify soil type with temperature calibration and Mahalanobis OOD gating."""
    content = await file.read()
    quality = pipeline_service.quality_gate.validate_soil_image(content)
    if not quality.is_acceptable:
        return SoilPrediction(
            soil_class="Rejected",
            confidence=0.0,
            raw_confidence=0.0,
            calibrated=False,
            probabilities={},
            model_version="1.3.0",
        )

    pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    pred, _ = pipeline_service.soil_model.predict(pil_img)
    return pred


# -------------------------------------------------------------------------
# DEDICATED DISEASE CLASSIFIER ROUTE
# -------------------------------------------------------------------------
@app.post("/api/v1/disease/predict", response_model=DiseasePrediction, tags=["Disease"])
async def predict_disease(file: UploadFile = File(...)):
    """Detect plant foliar disease with temperature scaling and reliability assessment."""
    content = await file.read()
    quality = pipeline_service.quality_gate.validate_leaf_image(content)
    if not quality.is_acceptable:
        return DiseasePrediction(
            disease_class="Rejected",
            confidence=0.0,
            raw_confidence=0.0,
            probabilities={},
            model_version="1.3.0",
        )

    pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    pred, _ = pipeline_service.disease_model.predict(pil_img)
    return pred


# -------------------------------------------------------------------------
# DEDICATED LOCALIZATION ROUTE (YOLO)
# -------------------------------------------------------------------------
@app.post("/api/v1/disease/localize", response_model=LocalizationResult, tags=["Disease"])
async def localize_disease(file: UploadFile = File(...)):
    """Detect lesion bounding boxes using YOLOv8 / YOLO11 architecture."""
    content = await file.read()
    pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    result, _ = pipeline_service.localizer.localize(pil_img)
    return result


# -------------------------------------------------------------------------
# DEDICATED SEGMENTATION ROUTE (U-NET)
# -------------------------------------------------------------------------
@app.post("/api/v1/disease/segment", response_model=SegmentationResult, tags=["Disease"])
async def segment_disease(file: UploadFile = File(...)):
    """Segment foliar necrotic lesions and compute affected area percentage via U-Net."""
    content = await file.read()
    pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    result, _ = pipeline_service.segmenter.segment_leaf(pil_img)
    return result


# -------------------------------------------------------------------------
# DEDICATED CROP RECOMMENDATION ROUTE
# -------------------------------------------------------------------------
@app.post("/api/v1/recommendations", response_model=List[CropRecommendation], tags=["Recommendation"])
async def get_crop_recommendations(
    soil_type: str = Form("alluvial"),
    n: Optional[float] = Form(None),
    p: Optional[float] = Form(None),
    k: Optional[float] = Form(None),
    ph: Optional[float] = Form(None),
    temperature: Optional[float] = Form(None),
    humidity: Optional[float] = Form(None),
    rainfall: Optional[float] = Form(None),
):
    """Generate calibrated Top-5 crop recommendations based on agronomic features."""
    recs, _, _ = pipeline_service.recommender.recommend(
        soil_pred=None,
        disease_pred=None,
        ph=ph,
        nitrogen=n,
        phosphorus=p,
        potassium=k,
        temperature=temperature,
        humidity=humidity,
        rainfall=rainfall,
    )
    return recs



# -------------------------------------------------------------------------
# MODEL REGISTRY
# -------------------------------------------------------------------------
@app.get("/api/v1/models", response_model=List[ModelMetadata], tags=["Registry"])
async def list_registered_models():
    """List all certified models and their production versions, metrics, and checksums."""
    return pipeline_service.registry.list_models()


# -------------------------------------------------------------------------
# HISTORICAL REQUEST AUDIT
# -------------------------------------------------------------------------
@app.get("/api/v1/requests/{request_id}", tags=["Audit"])
async def get_analysis_request(request_id: str):
    """Retrieve historical inference telemetry and results by request UUID."""
    if not pipeline_service.db:
        raise HTTPException(status_code=501, detail="Persistence service not configured")
    record = pipeline_service.db.get_record_by_request_id(request_id)
    if not record:
        raise HTTPException(status_code=404, detail="Request ID not found in telemetry store")
    return record
