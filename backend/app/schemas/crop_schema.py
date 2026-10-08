"""Pydantic schemas for Phase 5: Precision Crop Recommendation."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class CropRecommendationItem(BaseModel):
    """Detailed agronomic recommendation for an individual crop candidate."""
    crop_name: str = Field(..., description="Name of the recommended crop")
    rank: int = Field(..., ge=1, description="Recommendation priority rank (1 = Top Match)")
    suitability_score: float = Field(..., ge=0.0, le=100.0, description="Overall compatibility score (0-100%)")
    compatibility_grade: str = Field(..., description="Categorical recommendation grade (e.g. Highly Recommended)")
    projected_yield: str = Field(..., description="Estimated yield potential based on farm vigor index")
    growing_duration_days: str = Field(..., description="Approximate days from sowing to harvest")
    water_requirement: str = Field(..., description="Crop water demand level (Low, Moderate, High, Very High)")
    ideal_soil_type: List[str] = Field(default_factory=list, description="Optimal soil classifications")
    cultivation_tips: List[str] = Field(default_factory=list, description="Key agronomic management best practices")


class CropRecommendationResponse(BaseModel):
    """Complete response payload for Phase 5 Crop Recommendation."""
    status: str = "success"
    message: str = "Precision crop recommendations generated successfully"
    request_id: str
    primary_crop: str = Field(..., description="Top-ranked primary crop choice")
    primary_suitability: float = Field(..., description="Top crop compatibility percentage")
    recommendations: List[CropRecommendationItem] = Field(..., description="Ranked list of top candidate crops")
    soil_context: Dict[str, Any] = Field(..., description="Summary of soil type and chemistry evaluated")
    climate_context: Dict[str, Any] = Field(..., description="Summary of environmental factors evaluated")
    ready_for_phase_6: bool = True
    next_step: str = "Phase 6: Backend/API Integration & Orchestration"
