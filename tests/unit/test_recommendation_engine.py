"""Unit tests for EDAPHNET Crop Recommendation Engine and Missing Data Strategy."""

import pytest
from src.edaphnet.domain.enums import MeasurementStatus
from src.edaphnet.recommendation.engine import CropRecommendationModel


@pytest.fixture
def recommender():
    return CropRecommendationModel()


def test_recommendation_all_features_present(recommender):
    recs, missing, rel_status = recommender.recommend(
        soil_pred=None,
        disease_pred=None,
        nitrogen=80.0,
        phosphorus=40.0,
        potassium=40.0,
        temperature=28.0,
        humidity=75.0,
        ph=6.8,
        rainfall=200.0,
    )
    assert len(recs) >= 5
    assert len(missing) == 0  # No missing measurements
    assert recs[0].rank == 1
    assert 0.0 <= recs[0].suitability_score <= 100.0
    assert rel_status == "HIGH"


def test_recommendation_missing_data_strategy(recommender):
    # Only rainfall provided
    recs, missing, rel_status = recommender.recommend(
        soil_pred=None,
        disease_pred=None,
        rainfall=180.0,
    )
    assert len(recs) >= 5
    assert len(missing) > 0

    # Ensure missing measurements are identified
    missing_names = [m.parameter for m in missing]
    assert "nitrogen" in missing_names
    assert "phosphorus" in missing_names
    assert "potassium" in missing_names
    assert "ph" in missing_names

    # Check status is ESTIMATED
    for m in missing:
        assert m.status in [MeasurementStatus.MISSING, MeasurementStatus.ESTIMATED]
        assert m.estimated_value is not None



def test_recommendation_invalid_values_handled(recommender):
    # Pass out-of-bounds pH (e.g. 25.0)
    recs, missing, rel_status = recommender.recommend(
        soil_pred=None,
        disease_pred=None,
        ph=25.0,
        nitrogen=100.0,
    )
    invalid_m = [m for m in missing if m.parameter == "ph"]
    assert len(invalid_m) == 1
    assert invalid_m[0].status == MeasurementStatus.INVALID
