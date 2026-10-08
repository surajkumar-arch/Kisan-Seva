"""Pydantic schemas for request validation, metadata tracking, and response formatting."""

from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, field_validator


class SoilParameters(BaseModel):
    """Soil chemical and nutrient parameters."""
    ph: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=14.0,
        description="Soil pH scale between 0.0 and 14.0"
    )
    nitrogen: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Nitrogen content (N) in mg/kg or ppm (>= 0)"
    )
    phosphorus: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Phosphorus content (P) in mg/kg or ppm (>= 0)"
    )
    potassium: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Potassium content (K) in mg/kg or ppm (>= 0)"
    )


class EnvironmentalParameters(BaseModel):
    """Environmental and climatic parameters."""
    temperature: Optional[float] = Field(
        default=None,
        ge=-10.0,
        le=60.0,
        description="Ambient temperature in °C (-10°C to 60°C reasonable agricultural range)"
    )
    humidity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Relative air humidity percentage (0% to 100%)"
    )
    rainfall: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Precipitation / rainfall in mm (>= 0)"
    )


class ImageMetadata(BaseModel):
    """Metadata recorded for an uploaded image."""
    filename: str
    path: str
    content_type: str
    size_bytes: int
    dimensions: Tuple[int, int]


class AnalysisRecord(BaseModel):
    """Complete structured record for Phase 1 inputs."""
    request_id: str
    created_at: str
    soil_image: ImageMetadata
    plant_image: ImageMetadata
    soil_parameters: SoilParameters
    environment: EnvironmentalParameters


class AnalysisResponse(BaseModel):
    """Response returned upon successful Phase 1 input validation and storage."""
    status: str = "success"
    message: str = "Input validation successful"
    request_id: str
    data: AnalysisRecord
    ready_for_phase_2: bool = True
    next_step: str = "Phase 2: Soil Health Detection"


class SingleImageUploadResponse(BaseModel):
    """Response returned when uploading a single soil or plant image."""
    status: str = "success"
    message: str
    image: ImageMetadata


class ValidationErrorDetail(BaseModel):
    """Structured error payload for validation failures."""
    status: str = "error"
    message: str
    errors: Dict[str, str]
