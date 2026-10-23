"""Pydantic schemas for Phase 3: Plant Health & Foliage Disease Detection."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PlantDiseaseDetail(BaseModel):
    """Agronomic metadata for a plant pathology class."""
    condition: str
    pathogen_type: str = Field(..., description="Pathogen category (Fungal, Bacterial, Viral, or None)")
    symptoms: str = Field(..., description="Typical visible symptoms on foliage")
    severity_level: str = Field(..., description="Default virulence/severity rating")
    organic_treatment: List[str] = Field(default_factory=list, description="Organic & biological remedies")
    chemical_treatment: List[str] = Field(default_factory=list, description="Conventional chemical crop protection")
    preventive_measures: List[str] = Field(default_factory=list, description="Cultural & preventive agronomic practices")


class PlantPredictionResponse(BaseModel):
    """Diagnostic response for Phase 3 Plant Disease Detection."""
    status: str = "success"
    message: str = "Plant health analysis completed successfully"
    request_id: str
    plant_name: Optional[str] = Field(None, description="Crop or plant name identified by the classifier")
    specific_condition: Optional[str] = Field(None, description="Crop-specific disease label from the classifier")
    condition: str = Field(..., description="Detected plant health condition / disease name")
    is_healthy: bool = Field(..., description="Flag indicating if foliage is disease-free")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model prediction confidence score")
    probabilities: Dict[str, float] = Field(..., description="Class probability distribution")
    pathogen_type: str = Field(..., description="Pathogen category (Fungal, Bacterial, Viral, or None)")
    severity_level: str = Field(..., description="Infection severity (None, Low, Moderate, Severe)")
    plant_health_score: int = Field(..., ge=0, le=100, description="Plant Foliage Health Score (0-100)")
    symptoms: str = Field(..., description="Observed symptom indicators")
    organic_treatment: List[str] = Field(default_factory=list, description="Organic/biological treatments")
    chemical_treatment: List[str] = Field(default_factory=list, description="Chemical treatments")
    preventive_measures: List[str] = Field(default_factory=list, description="Preventive practices")
    image_url: Optional[str] = Field(None, description="Cloudinary CDN hosted image URL")
    ready_for_phase_4: bool = True
    next_step: str = "Phase 4: Feature Fusion"
