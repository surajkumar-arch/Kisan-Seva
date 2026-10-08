"""Pydantic schemas for Phase 4: Multi-Modal Feature Fusion."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DomainStressScores(BaseModel):
    """Normalized risk and stress indicators (0.0 to 1.0)."""
    soil_stress: float = Field(..., ge=0.0, le=1.0, description="Nutrient and pH stress factor")
    plant_disease_risk: float = Field(..., ge=0.0, le=1.0, description="Pathogen infection & foliage degradation factor")
    environmental_stress: float = Field(..., ge=0.0, le=1.0, description="Thermal, drought, or humidity stress factor")


class FusionFeatures(BaseModel):
    """Raw vector inputs channeled into the fusion neural network."""
    soil_vector: List[float] = Field(..., description="Soil probabilities (6) + normalized chemistry (4)")
    plant_vector: List[float] = Field(..., description="Plant disease probabilities (7) + health score (1)")
    climate_vector: List[float] = Field(..., description="Normalized Temperature, Humidity, Rainfall (3)")


class FusionResponse(BaseModel):
    """Structured response containing the 128-dimensional latent vector and composite indices."""
    status: str = "success"
    message: str = "Multi-modal feature fusion completed successfully"
    request_id: str
    farm_vigor_index: int = Field(..., ge=0, le=100, description="Composite Farm Vigor Index (FVI: 0-100)")
    vigor_status: str = Field(..., description="Categorical farm vigor status (Optimal, Moderate, Critical)")
    stress_scores: DomainStressScores
    latent_dim: int = Field(default=128, description="Dimensionality of unified latent space")
    latent_embedding: List[float] = Field(..., description="128-dimensional dense latent representation vector")
    modality_weights: Dict[str, float] = Field(..., description="Learned attention gating weights for each domain")
    fusion_summary: Dict[str, Any] = Field(..., description="High-level synthesis of soil, plant, and weather states")
    ready_for_phase_5: bool = True
    next_step: str = "Phase 5: Crop Recommendation"
