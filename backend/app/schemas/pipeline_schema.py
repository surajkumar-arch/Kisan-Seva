"""Pydantic schemas for Phase 6: Unified Pipeline Orchestration & Farm Diagnostics."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from backend.app.schemas.soil_schema import SoilPredictionResponse
from backend.app.schemas.plant_schema import PlantPredictionResponse
from backend.app.schemas.fusion_schema import FusionResponse
from backend.app.schemas.crop_schema import CropRecommendationResponse, CropRecommendationItem


class FarmAlert(BaseModel):
    """Urgent alert regarding farm disease, soil degradation, or climate stress."""
    level: str = Field(..., description="Alert severity: CRITICAL, WARNING, or INFO")
    category: str = Field(..., description="Domain: DISEASE, SOIL, NUTRIENT, or CLIMATE")
    title: str = Field(..., description="Brief alert title")
    message: str = Field(..., description="Detailed explanation of the alert condition")
    recommended_action: str = Field(..., description="Prescribed immediate response")


class ActionPlanItem(BaseModel):
    """Prioritized agronomic management step for farmer execution."""
    priority: int = Field(..., ge=1, description="Execution priority order (1 = highest)")
    category: str = Field(..., description="Category: Plant Protection, Soil Amendment, Irrigation, Crop Sowing")
    action: str = Field(..., description="Core actionable command")
    timeframe: str = Field(..., description="Target completion window (e.g. Immediate 0-48h, Within 7 Days)")
    details: str = Field(..., description="Agronomic specifics, dosages, and notes")


class FarmDiagnosticsSummary(BaseModel):
    """Aggregated executive diagnostic summary combining all multi-modal phases."""
    farm_health_score: int = Field(..., ge=0, le=100, description="Overall weighted farm health score (0-100)")
    farm_status: str = Field(..., description="Qualitative farm status (Thriving, Fair, Degraded, Critical)")
    soil_health_score: int = Field(..., ge=0, le=100, description="Soil fertility health score")
    plant_health_score: int = Field(..., ge=0, le=100, description="Plant foliage health score")
    farm_vigor_index: int = Field(..., ge=0, le=100, description="Composite multi-modal Farm Vigor Index (FVI)")
    soil_type: str = Field(..., description="Detected soil classification")
    plant_condition: str = Field(..., description="Detected plant foliage condition / disease")
    primary_crop: str = Field(..., description="Top recommended crop variety")
    primary_suitability: float = Field(..., description="Top crop suitability percentage")
    projected_yield: str = Field(..., description="Projected yield range (t/ha)")
    critical_alerts: List[FarmAlert] = Field(default_factory=list, description="Active farm hazard alerts")
    action_plan: List[ActionPlanItem] = Field(default_factory=list, description="Prioritized farmer task checklist")
    execution_time_ms: Dict[str, float] = Field(default_factory=dict, description="Latency breakdown per phase")


class PipelineAnalyzeResponse(BaseModel):
    """Unified master response containing complete end-to-end multi-modal farm intelligence."""
    status: str = "success"
    message: str = "End-to-end agricultural intelligence pipeline executed successfully"
    request_id: str
    timestamp: str
    summary: FarmDiagnosticsSummary
    phase1_input: Dict[str, Any]
    phase2_soil: SoilPredictionResponse
    phase3_plant: PlantPredictionResponse
    phase4_fusion: FusionResponse
    phase5_crop: CropRecommendationResponse


class PipelineHistoryItem(BaseModel):
    """Compact summary of a previous pipeline run for history and dashboard lists."""
    request_id: str
    created_at: str
    farm_health_score: int
    farm_vigor_index: int
    soil_type: str
    plant_condition: str
    primary_crop: str
    primary_suitability: float
    projected_yield: str
    critical_alerts_count: int


class PipelineHistoryResponse(BaseModel):
    """Paginated list of historical farm diagnostic runs."""
    status: str = "success"
    total_records: int
    limit: int
    offset: int
    records: List[PipelineHistoryItem]


class RegionalSummary(BaseModel):
    """Aggregated regional intelligence across batch scans."""
    total_scans: int
    avg_farm_health: float
    avg_vigor_index: float
    common_soil_types: Dict[str, int]
    dominant_diseases: Dict[str, int]
    top_recommended_crops: Dict[str, int]


class BatchAnalyzeResponse(BaseModel):
    """Response payload for multi-sample batch farm analysis."""
    status: str = "success"
    total_submitted: int
    total_processed: int
    failed_count: int
    results: List[PipelineAnalyzeResponse]
    regional_summary: RegionalSummary


class PipelineStatsResponse(BaseModel):
    """High-level system metrics and agronomic distributions."""
    status: str = "success"
    total_farms_analyzed: int
    avg_farm_health_score: float
    avg_farm_vigor_index: float
    soil_distribution: Dict[str, int]
    disease_distribution: Dict[str, int]
    crop_recommendation_distribution: Dict[str, int]
    health_grade_distribution: Dict[str, int]
