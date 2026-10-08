"""Health check route."""

from fastapi import APIRouter
from backend.app.services.cloudinary_service import is_cloudinary_configured
from backend.app.services.mongo_service import is_mongo_connected

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        JSON confirming API health and cloud services status.
    """
    return {
        "status": "healthy",
        "service": "smart-agriculture-api",
        "cloudinary": "active" if is_cloudinary_configured() else "inactive",
        "mongodb": "connected" if is_mongo_connected() else "fallback_sqlite",
    }
