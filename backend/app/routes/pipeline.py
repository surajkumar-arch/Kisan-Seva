"""FastAPI router for Phase 6: Unified Pipeline Orchestration & Farm Diagnostics."""

import io
from typing import Optional, List
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Query, status
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from PIL import Image

from backend.app.schemas.input_schema import SoilParameters, EnvironmentalParameters
from backend.app.schemas.pipeline_schema import (
    PipelineAnalyzeResponse,
    PipelineHistoryResponse,
    PipelineStatsResponse,
    BatchAnalyzeResponse,
    RegionalSummary,
)
from backend.app.services.pipeline_service import (
    orchestrate_pipeline,
    get_pipeline_history_records,
    get_pipeline_statistics,
    generate_markdown_report,
    generate_html_report,
    generate_agronomic_prescription,
)
from backend.app.utils.image_utils import validate_image_file, ImageValidationError

router = APIRouter(prefix="/api/pipeline", tags=["Pipeline Orchestration & Diagnostics"])


@router.post(
    "/analyze-all",
    response_model=PipelineAnalyzeResponse,
    summary="1-Click Multi-Modal Farm Diagnostic Pipeline",
    description="Executes Phases 1 through 5 in a single unified atomic pipeline and returns executive farm diagnostics.",
)
async def run_pipeline_analyze_all(
    soil_image: Optional[UploadFile] = File(None, description="Soil photograph (JPEG/PNG)"),
    plant_image: Optional[UploadFile] = File(None, description="Plant foliage photograph (JPEG/PNG)"),
    request_id: Optional[str] = Form(None, description="Existing Phase 1 request ID to process"),
    ph: Optional[float] = Form(None, ge=0.0, le=14.0, description="Soil pH level"),
    nitrogen: Optional[float] = Form(None, ge=0.0, description="Available Nitrogen (kg/ha)"),
    phosphorus: Optional[float] = Form(None, ge=0.0, description="Available Phosphorus (kg/ha)"),
    potassium: Optional[float] = Form(None, ge=0.0, description="Available Potassium (kg/ha)"),
    temperature: Optional[float] = Form(None, description="Ambient temperature (°C)"),
    humidity: Optional[float] = Form(None, ge=0.0, le=100.0, description="Relative humidity (%)"),
    rainfall: Optional[float] = Form(None, ge=0.0, description="Precipitation level (mm)"),
    force_refresh: bool = Form(False, description="Bypass cache and force model re-inference"),
):
    """Execute end-to-end pipeline analysis."""
    soil_pil: Optional[Image.Image] = None
    plant_pil: Optional[Image.Image] = None

    # Validate and decode soil image if uploaded
    if soil_image:
        content = await soil_image.read()
        try:
            soil_pil = validate_image_file(content, soil_image.filename or "soil.jpg", field_name="soil_image")
        except ImageValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid soil image: {exc.message}",
            )

    # Validate and decode plant image if uploaded
    if plant_image:
        content = await plant_image.read()
        try:
            plant_pil = validate_image_file(content, plant_image.filename or "plant.jpg", field_name="plant_image")
        except ImageValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid plant image: {exc.message}",
            )

    soil_params = SoilParameters(
        ph=ph,
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
    )
    env_params = EnvironmentalParameters(
        temperature=temperature,
        humidity=humidity,
        rainfall=rainfall,
    )

    try:
        response = orchestrate_pipeline(
            request_id=request_id,
            soil_image=soil_pil,
            plant_image=plant_pil,
            soil_params=soil_params,
            env_params=env_params,
            force_refresh=force_refresh,
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failure: {str(e)}",
        )


@router.post(
    "/analyze-by-id/{req_id}",
    response_model=PipelineAnalyzeResponse,
    summary="Execute Pipeline for Pre-Acquired Request ID",
    description="Run the end-to-end pipeline on inputs previously saved in Phase 1.",
)
async def run_pipeline_by_request_id(
    req_id: str,
    force_refresh: bool = Query(False, description="Force re-inference without using cache"),
):
    """Execute pipeline from request_id."""
    try:
        return orchestrate_pipeline(request_id=req_id, force_refresh=force_refresh)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failure: {str(e)}",
        )


@router.get(
    "/report/{req_id}",
    summary="Export Agronomic Diagnostic Report",
    description="Generates a diagnostic report in JSON, Markdown, or printable HTML format.",
)
async def get_diagnostic_report(
    req_id: str,
    format: str = Query("json", pattern="^(json|markdown|html)$", description="Report format: json, markdown, or html"),
):
    """Export agronomic diagnostic report."""
    try:
        analysis = orchestrate_pipeline(request_id=req_id, force_refresh=False)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analysis record found for request ID: {req_id}",
        )

    if format == "markdown":
        md_content = generate_markdown_report(analysis)
        return PlainTextResponse(content=md_content, media_type="text/markdown")
    elif format == "html":
        html_content = generate_html_report(analysis)
        return HTMLResponse(content=html_content)
    else:
        return analysis


@router.get(
    "/prescription/{req_id}",
    summary="Generate Agronomic Prescription Sheet",
    description="Generates an agronomic medical-grade prescription for farmers and extension officers.",
)
async def get_agronomic_prescription_endpoint(req_id: str):
    """Retrieve structured prescription for farm field."""
    try:
        analysis = orchestrate_pipeline(request_id=req_id, force_refresh=False)
        return generate_agronomic_prescription(analysis)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get(
    "/history",
    response_model=PipelineHistoryResponse,
    summary="Query Pipeline Diagnostic History",
    description="Returns paginated historical farm diagnostic runs with optional filters.",
)
async def get_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    min_health_score: Optional[int] = Query(None, ge=0, le=100),
    soil_type: Optional[str] = Query(None),
):
    """Retrieve history of pipeline executions."""
    return get_pipeline_history_records(
        limit=limit,
        offset=offset,
        min_health_score=min_health_score,
        soil_type=soil_type,
    )


@router.get(
    "/stats",
    response_model=PipelineStatsResponse,
    summary="Global Agriculture Pipeline Statistics",
    description="Aggregated platform statistics including farm health scores, disease frequencies, and soil distributions.",
)
async def get_statistics():
    """Retrieve system analytics."""
    return get_pipeline_statistics()


@router.post(
    "/batch-analyze",
    response_model=BatchAnalyzeResponse,
    summary="Batch Multi-Farm Analysis",
    description="Execute pipeline on multiple request IDs in batch and compute regional summary intelligence.",
)
async def run_batch_pipeline(request_ids: List[str]):
    """Execute pipeline for multiple farm records in batch."""
    if not request_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The request_ids list must not be empty.",
        )

    results: List[PipelineAnalyzeResponse] = []
    failed = 0

    for req_id in request_ids:
        try:
            resp = orchestrate_pipeline(request_id=req_id)
            results.append(resp)
        except Exception:
            failed += 1

    # Aggregate regional metrics
    total_processed = len(results)
    avg_health = sum(r.summary.farm_health_score for r in results) / total_processed if total_processed else 0.0
    avg_vigor = sum(r.summary.farm_vigor_index for r in results) / total_processed if total_processed else 0.0

    soil_counts: dict = {}
    disease_counts: dict = {}
    crop_counts: dict = {}

    for r in results:
        st = r.summary.soil_type
        soil_counts[st] = soil_counts.get(st, 0) + 1

        dis = r.summary.plant_condition
        disease_counts[dis] = disease_counts.get(dis, 0) + 1

        cr = r.summary.primary_crop
        crop_counts[cr] = crop_counts.get(cr, 0) + 1

    regional = RegionalSummary(
        total_scans=total_processed,
        avg_farm_health=round(avg_health, 1),
        avg_vigor_index=round(avg_vigor, 1),
        common_soil_types=soil_counts,
        dominant_diseases=disease_counts,
        top_recommended_crops=crop_counts,
    )

    return BatchAnalyzeResponse(
        status="success",
        total_submitted=len(request_ids),
        total_processed=total_processed,
        failed_count=failed,
        results=results,
        regional_summary=regional,
    )
