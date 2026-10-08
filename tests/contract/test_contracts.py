"""Contract tests verifying EDAPHNET typed Pydantic models and schemas."""

import uuid
import pytest
from src.edaphnet.domain.enums import ModelStatus, ReliabilityLevel, MeasurementStatus
from src.edaphnet.domain.models import (
    CropRecommendation,
    DiseasePrediction,
    ImageQualityReport,
    InferenceResponse,
    LocalizationResult,
    MissingMeasurement,
    ModelMetadata,
    ReliabilityReport,
    SegmentationResult,
    SoilPrediction,
)


def test_inference_response_contract_schema():
    response = InferenceResponse(
        request_id=str(uuid.uuid4()),
        soil=SoilPrediction(
            soil_class="Red Soil",
            confidence=0.87,
            raw_confidence=0.91,
            calibrated=True,
            probabilities={"Red Soil": 0.87, "Alluvial Soil": 0.10},
        ),
        disease=DiseasePrediction(
            disease_class="Early Blight",
            confidence=0.92,
            raw_confidence=0.95,
            pathogen_type="Fungal",
            severity_level="Moderate",
            probabilities={"Early Blight": 0.92},
        ),
        recommendations=[
            CropRecommendation(
                rank=1,
                crop_name="cotton",
                suitability_score=91.0,
                compatibility_grade="A",
                projected_yield="2.8 - 3.4 Tons/Hectare",
                water_requirement="500 - 700 mm",
                growing_duration_days="150 - 180 Days",
            )
        ],
        reliability=ReliabilityReport(
            decision="PROCEED",
            level=ReliabilityLevel.MODERATE,
            ood_detected=False,
        ),
        missing_measurements=[
            MissingMeasurement(
                parameter="ph",
                status=MeasurementStatus.MISSING,
                estimated_value=6.5,
                impact_on_reliability="HIGH",
                recommended_action="Conduct soil test",
            )
        ],
        model_versions={"soil_classifier": "1.3.0", "disease_classifier": "1.3.0"},
    )

    data = response.model_dump(by_alias=True)
    assert "request_id" in data
    assert data["soil"]["class"] == "Red Soil"
    assert data["disease"]["class"] == "Early Blight"
    assert data["reliability"]["level"] in [ReliabilityLevel.MODERATE, "MODERATE", "moderate"]
    assert len(data["recommendations"]) == 1
    assert len(data["missing_measurements"]) == 1



def test_model_metadata_contract():
    meta = ModelMetadata(
        model_name="soil_classifier",
        version="1.3.0",
        framework="pytorch",
        dataset_version="soil-types-v2.1",
        preprocessing_version="norm-imagenet-224",
        metrics={"accuracy": 0.942, "f1": 0.938},
        status=ModelStatus.READY,
    )
    assert meta.model_name == "soil_classifier"
    assert meta.status == ModelStatus.READY
    assert meta.metrics["accuracy"] == 0.942
