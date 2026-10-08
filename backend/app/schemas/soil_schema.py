"""Pydantic schemas for Phase 2: Soil Health Detection."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from backend.app.schemas.input_schema import SoilParameters


class SoilPhysicalProperties(BaseModel):
    """Agronomic and physical soil properties."""
    texture: str = Field(..., description="Soil texture description")
    drainage: str = Field(..., description="Drainage capacity (e.g. Well-drained, Poor, Excessive)")
    water_retention: str = Field(..., description="Water holding capacity")
    aeration: str = Field(..., description="Root aeration quality")
    organic_matter_estimate: str = Field(..., description="Estimated organic carbon content")


class ChemicalAssessment(BaseModel):
    """Evaluation of soil chemical nutrients and pH."""
    ph_status: Optional[str] = Field(None, description="pH interpretation (e.g. Acidic, Neutral, Alkaline)")
    nitrogen_status: Optional[str] = Field(None, description="Nitrogen sufficiency level (Low, Medium, High)")
    phosphorus_status: Optional[str] = Field(None, description="Phosphorus sufficiency level (Low, Medium, High)")
    potassium_status: Optional[str] = Field(None, description="Potassium sufficiency level (Low, Medium, High)")


class SoilPredictionRequest(BaseModel):
    """Request payload to predict soil health from an existing Phase 1 request ID."""
    request_id: str = Field(..., description="Unique request ID from Phase 1")
    soil_parameters: Optional[SoilParameters] = Field(None, description="Optional override/update of soil parameters")


class SoilPredictionResponse(BaseModel):
    """Comprehensive response for Phase 2 Soil Health Detection."""
    status: str = "success"
    message: str = "Soil health analysis completed successfully"
    request_id: str
    soil_type: str = Field(..., description="Predicted primary soil classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model prediction confidence score")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution across all soil classes")
    soil_health_score: int = Field(..., ge=0, le=100, description="Overall Soil Quality Index / Health Score (0-100)")
    fertility_grade: str = Field(..., description="Fertility evaluation (High, Moderate, Low)")
    physical_properties: SoilPhysicalProperties
    chemical_assessment: Optional[ChemicalAssessment] = None
    recommendations: List[str] = Field(default_factory=list, description="Actionable agronomic soil amendment recommendations")
    ready_for_phase_3: bool = True
    next_step: str = "Phase 3: Plant Health Detection"
