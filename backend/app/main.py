"""FastAPI Main Entrypoint for Smart Agriculture / Crop Management System (Phase 1)."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.routes.health import router as health_router
from backend.app.routes.upload import router as upload_router
from backend.app.routes.soil_health import router as soil_health_router
from backend.app.routes.plant_health import router as plant_health_router
from backend.app.routes.fusion import router as fusion_router
from backend.app.routes.crop import router as crop_router
from backend.app.routes.pipeline import router as pipeline_router
from backend.app.routes.evaluation import router as evaluation_router
from backend.app.routes.environment import router as environment_router
from backend.app.routes.insights import router as insights_router
from backend.app.routes.community import router as community_router
from backend.app.routes.advisor import router as advisor_router
from backend.app.routes.voice import router as voice_router
from backend.app.routes.spectral import router as spectral_router
from backend.app.routes.video_scanner import router as video_scanner_router
from backend.app.routes.council import router as council_router
from backend.app.routes.edge import router as edge_router
from backend.app.routes.satellite import router as satellite_router
from backend.app.routes.tank_mix import router as tank_mix_router
from backend.app.routes.yield_forecast import router as yield_forecast_router
from backend.app.routes.mandi import router as mandi_router
from backend.app.routes.vernacular_voice import router as vernacular_voice_router
from backend.app.routes.insurance import router as insurance_router
from backend.app.routes.microclimate import router as microclimate_router
from backend.app.routes.carbon_ledger import router as carbon_ledger_router
from backend.app.routes.precision_irrigation import router as precision_irrigation_router
from backend.app.routes.epidemic_swarm import router as epidemic_swarm_router
from backend.app.services.community_service import init_community_db
from backend.app.services.validation import init_db
from backend.app.services.soil_health_service import init_phase2_db
from backend.app.services.plant_health_service import init_phase3_db
from backend.app.services.fusion_service import init_phase4_db
from backend.app.services.crop_service import init_phase5_db
from backend.app.services.pipeline_service import init_phase6_db

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOADS_DIR = BASE_DIR / "backend" / "uploads"
DATA_DIR = BASE_DIR / "data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event for initializing databases and directories."""
    # Ensure upload, data, and model directories exist
    os.makedirs(UPLOADS_DIR / "soil", exist_ok=True)
    os.makedirs(UPLOADS_DIR / "plant", exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(BASE_DIR / "models" / "soil_health", exist_ok=True)
    os.makedirs(BASE_DIR / "models" / "plant_health", exist_ok=True)
    os.makedirs(BASE_DIR / "models" / "fusion", exist_ok=True)
    os.makedirs(BASE_DIR / "models" / "crop", exist_ok=True)
    
    # Initialize SQLite database schemas
    init_db()
    init_phase2_db()
    init_phase3_db()
    init_phase4_db()
    init_phase5_db()
    init_phase6_db()
    init_community_db()
    yield


app = FastAPI(
    title="🌱 Smart Agriculture System - API",
    description="Multi-Modal Agricultural Intelligence Engine: Acquisition, Diagnostics, Fusion & Crop Recommendation.",
    version="8.0.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit and React Native (Expo) clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom validation error handler for consistent error formats
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = {}
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"] if loc != "body")
        errors[field or "request"] = error["msg"]

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": "error",
            "message": "Invalid request parameters",
            "errors": errors,
        },
    )


# Include Routers
app.include_router(health_router)
app.include_router(upload_router)
app.include_router(soil_health_router)
app.include_router(plant_health_router)
app.include_router(fusion_router)
app.include_router(crop_router)
app.include_router(pipeline_router)
app.include_router(evaluation_router)
app.include_router(environment_router)
app.include_router(insights_router)
app.include_router(community_router)
app.include_router(advisor_router)
app.include_router(voice_router)
app.include_router(spectral_router)
app.include_router(video_scanner_router)
app.include_router(council_router)
app.include_router(edge_router)
app.include_router(satellite_router)
app.include_router(tank_mix_router)
app.include_router(yield_forecast_router)
app.include_router(mandi_router)
app.include_router(vernacular_voice_router)
app.include_router(insurance_router)
app.include_router(microclimate_router)
app.include_router(carbon_ledger_router)
app.include_router(precision_irrigation_router)
app.include_router(epidemic_swarm_router)


@app.get("/")
async def root():
    """Welcome endpoint providing system overview and links."""
    return {
        "system": "Smart Agriculture / Crop Management System",
        "active_phases": [
            "PHASE 1: Input Acquisition",
            "PHASE 2: Soil Health Detection",
            "PHASE 3: Plant Health Detection",
            "PHASE 4: Feature Fusion",
            "PHASE 5: Crop Recommendation",
            "PHASE 6: Backend/API Integration",
            "PHASE 7: Frontend/Dashboard",
            "PHASE 8: Testing, Evaluation & Deployment",
        ],
        "status": "production_ready",
        "endpoints": {
            "health": "/health",
            "soil_upload": "/api/soil/upload",
            "plant_upload": "/api/plant/upload",
            "analyze_input": "/api/analyze/input",
            "soil_classes": "/api/soil/classes",
            "soil_predict": "/api/soil/predict",
            "plant_classes": "/api/plant/classes",
            "plant_predict": "/api/plant/predict",
            "feature_fusion": "/api/fusion/fuse",
            "crop_catalog": "/api/crop/catalog",
            "crop_recommend": "/api/crop/recommend",
            "pipeline_analyze_all": "/api/pipeline/analyze-all",
            "pipeline_prescription": "/api/pipeline/prescription/{req_id}",
            "pipeline_history": "/api/pipeline/history",
            "pipeline_stats": "/api/pipeline/stats",
            "pipeline_batch": "/api/pipeline/batch-analyze",
            "evaluation_metrics": "/api/evaluation/metrics",
            "community_outbreaks": "/api/community/outbreaks",
            "community_report": "/api/community/report",
            "advisor_economics": "/api/advisor/economics",
            "voice_status": "/api/ai/voice/status",
            "docs": "/docs",
        },
        "all_phases_completed": True,
    }
