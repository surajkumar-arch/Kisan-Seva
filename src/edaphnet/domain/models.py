"""Domain Models and Explicit Data Contracts for EDAPHNET Platform."""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from src.edaphnet.domain.enums import (
    ReliabilityLevel,
    AbstainReason,
    MeasurementStatus,
    ModelStatus,
)


class QualityIssue(BaseModel):
    check_name: str = Field(..., description="Name of validation test that flagged an anomaly")
    severity: str = Field("WARNING", description="Severity level: CRITICAL, WARNING, INFO")
    details: str = Field(..., description="Actionable technical explanation of the quality issue")


class ImageQualityReport(BaseModel):
    passed: bool = Field(..., description="Whether the image meets the production inference quality threshold")
    blur_score: float = Field(..., description="Laplacian variance blur metric (lower = blurrier)")
    brightness: float = Field(..., description="Normalized mean intensity (0-255)")
    contrast: float = Field(..., description="Normalized standard deviation of luminance")
    aspect_ratio: float = Field(..., description="Width-to-height ratio")
    resolution: List[int] = Field(..., description="[width, height] in pixels")
    issues: List[QualityIssue] = Field(default_factory=list, description="List of quality defects detected")
    recommendation: str = Field("PROCEED", description="PROCEED, RETAKE_IMAGE, or REJECT")

    @property
    def is_acceptable(self) -> bool:
        return self.passed

    @property
    def reasons(self) -> List[str]:
        return [i.details for i in self.issues] if self.issues else []



class ModelMetadata(BaseModel):
    model_name: str
    version: str
    framework: str = "pytorch"
    checksum: Optional[str] = None
    dataset_version: str = "v1.0.0"
    preprocessing_version: str = "v1.0.0"
    status: ModelStatus = ModelStatus.READY
    metrics: Dict[str, float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SoilPrediction(BaseModel):
    soil_class: str = Field(..., alias="class", description="Predicted soil taxonomic classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Temperature-calibrated posterior confidence")
    raw_confidence: float = Field(..., ge=0.0, le=1.0, description="Uncalibrated model softmax probability")
    calibrated: bool = Field(True, description="Whether temperature scaling was applied")
    probabilities: Dict[str, float] = Field(default_factory=dict, description="Full posterior class distribution")
    model_version: str = Field("1.3.0", description="Model version used for inference")

    class Config:
        populate_by_name = True


class DiseasePrediction(BaseModel):
    disease_class: str = Field(..., alias="class", description="Diagnosed foliar pathology")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Top-class probability; validation calibrated only when calibrated is true",
    )
    raw_confidence: float = Field(..., ge=0.0, le=1.0, description="Raw softmax probability")
    calibrated: bool = Field(False, description="Whether validation-fitted temperature scaling was applied")
    pathogen_type: str = Field("None", description="Causal agent: Fungal, Oomycete, Bacterial, Viral, None")
    severity_level: str = Field("None", description="None, Low, Moderate, Severe")
    probabilities: Dict[str, float] = Field(default_factory=dict, description="Full class posterior distribution")
    model_version: str = Field("1.3.0", description="Version of disease classifier model")

    class Config:
        populate_by_name = True


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    label: str


class LocalizationResult(BaseModel):
    detected: bool
    num_lesions: int
    boxes: List[BoundingBox] = Field(default_factory=list)
    annotated_image_url: Optional[str] = None
    model_version: str = "yolov8-1.0.0"


class SegmentationResult(BaseModel):
    segmented: bool
    affected_area_percentage: float = Field(..., ge=0.0, le=100.0, description="% of leaf surface damaged")
    total_leaf_pixels: int = 0
    lesion_pixels: int = 0
    mask_image_url: Optional[str] = None
    model_version: str = "unet-1.0.0"


class CropRecommendation(BaseModel):
    crop_name: str
    rank: int = Field(..., ge=1)
    suitability_score: float = Field(..., ge=0.0, le=100.0)
    compatibility_grade: str
    projected_yield: str
    water_requirement: str
    growing_duration_days: str
    feature_contributions: Dict[str, float] = Field(default_factory=dict, description="SHAP feature attribution weights")
    cultivation_tips: List[str] = Field(default_factory=list)


class MissingMeasurement(BaseModel):
    parameter: str
    status: MeasurementStatus
    estimated_value: Optional[float] = None
    impact_on_reliability: str = "MODERATE"
    recommended_action: str


class ReliabilityReport(BaseModel):
    level: ReliabilityLevel
    decision: str = "PROCEED"  # PROCEED, BEST_EFFORT, ABSTAIN
    ood_detected: bool = False
    mahalanobis_distance: float = 0.0
    threshold_distance: float = 12.5
    reason: AbstainReason = AbstainReason.NONE
    message: str = "Sample is within calibrated in-distribution envelope."
    conformal_coverage: float = 0.95


class ExplainabilityArtifacts(BaseModel):
    gradcam_soil_url: Optional[str] = None
    gradcam_leaf_url: Optional[str] = None
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="SHAP feature importances")
    explanation_summary: str = ""


class InferenceRequest(BaseModel):
    request_id: Optional[str] = None
    ph: Optional[float] = None
    nitrogen: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rainfall: Optional[float] = None
    location: Optional[str] = None
    season: Optional[str] = None
    include_localization: bool = False
    include_segmentation: bool = False
    include_explainability: bool = True


class InferenceResponse(BaseModel):
    request_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    soil: Optional[SoilPrediction] = None
    disease: Optional[DiseasePrediction] = None
    localization: Optional[LocalizationResult] = None
    segmentation: Optional[SegmentationResult] = None
    recommendations: List[CropRecommendation] = Field(default_factory=list)
    reliability: ReliabilityReport
    missing_measurements: List[MissingMeasurement] = Field(default_factory=list)
    explainability: Optional[ExplainabilityArtifacts] = None
    model_versions: Dict[str, str] = Field(default_factory=dict)
    processing_time_ms: float = 0.0

    @property
    def inference_latency_ms(self) -> float:
        return self.processing_time_ms

