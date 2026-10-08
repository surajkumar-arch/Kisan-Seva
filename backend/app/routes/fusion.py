"""API routes for Phase 4: Multi-Modal Feature Fusion."""

import json
import sqlite3
from pathlib import Path
from fastapi import APIRouter, Form, HTTPException, status
from fastapi.responses import JSONResponse

from backend.app.schemas.input_schema import ValidationErrorDetail
from backend.app.schemas.fusion_schema import FusionResponse
from backend.app.services.fusion_service import (
    execute_feature_fusion,
    DB_PATH,
)

router = APIRouter(prefix="/api/fusion", tags=["Feature Fusion"])


@router.post(
    "/fuse",
    response_model=FusionResponse,
    responses={400: {"model": ValidationErrorDetail}, 404: {"description": "Record not found"}}
)
async def fuse_features_endpoint(
    request_id: str = Form(..., description="Unique request ID from Phase 1")
):
    """Execute Gated Multi-Modal Feature Fusion across Soil, Plant, and Environmental domains.

    Produces a 128-dimensional dense latent vector and computes the Farm Vigor Index (FVI).
    """
    clean_id = request_id.strip()
    if not clean_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "message": "request_id is required", "errors": {"request_id": "Field is required."}}
        )

    try:
        response = execute_feature_fusion(clean_id)
        return response
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature fusion pipeline failed: {str(exc)}"
        )


@router.get("/history/{request_id}")
async def get_fusion_history(request_id: str):
    """Retrieve historical multi-modal fusion records and latent embeddings."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail="No database records found.")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feature_fusion_records WHERE request_id = ?", (request_id,))
        rows = cursor.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No fusion records found for request ID {request_id}.")

    results = []
    for row in rows:
        d = dict(row)
        d["latent_embedding"] = json.loads(d["latent_embedding"])
        d["modality_weights"] = json.loads(d["modality_weights"])
        d["fusion_summary"] = json.loads(d["fusion_summary"])
        results.append(d)

    return {"status": "success", "data": results}
