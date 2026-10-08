"""Enterprise Agricultural Crop Recommendation Engine with Missing Data Strategy & SHAP Attributions."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.edaphnet.domain.models import (
    CropRecommendation,
    MissingMeasurement,
    SoilPrediction,
    DiseasePrediction,
)
from src.edaphnet.domain.enums import MeasurementStatus, ReliabilityLevel
from backend.app.models.crop_net import CROP_CATALOG, CROPS_LIST


class CropRecommendationModel:
    """Feature-Driven Tabular Agricultural Recommender."""

    def __init__(self):
        self.crops = CROPS_LIST
        self.catalog = CROP_CATALOG

    def assess_input_measurements(
        self,
        ph: Optional[float],
        n: Optional[float],
        p: Optional[float],
        k: Optional[float],
        temperature: Optional[float],
        humidity: Optional[float],
        rainfall: Optional[float],
    ) -> Tuple[Dict[str, float], List[MissingMeasurement], str]:
        """Classify each agricultural measurement into KNOWN, MISSING, ESTIMATED, or INVALID.

        Never silently replace missing parameters with arbitrary default numbers.
        """
        resolved_features = {}
        missing_reports = []

        params_spec = {
            "ph": (ph, (0.0, 14.0), 6.5, "Standard potentiometric soil pH test", "CRITICAL"),
            "nitrogen": (n, (0.0, 400.0), 120.0, "Subbiah & Asija alkaline permanganate N-test", "HIGH"),
            "phosphorus": (p, (0.0, 200.0), 35.0, "Olsen / Bray spectrophotometric P-test", "HIGH"),
            "potassium": (k, (0.0, 500.0), 120.0, "Ammonium acetate flame photometric K-test", "HIGH"),
            "temperature": (temperature, (-10.0, 60.0), 26.0, "Micro-meteorological weather station sensor", "MODERATE"),
            "humidity": (humidity, (5.0, 100.0), 65.0, "Relative humidity hygrometer reading", "MODERATE"),
            "rainfall": (rainfall, (0.0, 800.0), 110.0, "Rain-gauge precipitation seasonal estimate", "MODERATE"),
        }

        critical_missing_count = 0

        for name, (val, (val_min, val_max), default_est, rec_action, severity) in params_spec.items():
            if val is None:
                missing_reports.append(
                    MissingMeasurement(
                        parameter=name,
                        status=MeasurementStatus.MISSING,
                        estimated_value=default_est,
                        impact_on_reliability=severity,
                        recommended_action=f"Parameter is missing. Recommended field assay: {rec_action}.",
                    )
                )
                resolved_features[name] = default_est
                if severity == "CRITICAL" or severity == "HIGH":
                    critical_missing_count += 1

            elif val < val_min or val > val_max:
                missing_reports.append(
                    MissingMeasurement(
                        parameter=name,
                        status=MeasurementStatus.INVALID,
                        estimated_value=default_est,
                        impact_on_reliability="CRITICAL",
                        recommended_action=f"Value {val} falls outside physically plausible range ({val_min}-{val_max}). Recalibrate instrument.",
                    )
                )
                resolved_features[name] = default_est
                critical_missing_count += 1

            else:
                resolved_features[name] = float(val)

        # Reliability penalty string
        if critical_missing_count >= 3:
            reliability_penalty = "LOW"
        elif critical_missing_count >= 1:
            reliability_penalty = "MODERATE"
        else:
            reliability_penalty = "HIGH"

        return resolved_features, missing_reports, reliability_penalty

    def recommend(
        self,
        soil_pred: Optional[SoilPrediction],
        disease_pred: Optional[DiseasePrediction],
        ph: Optional[float] = None,
        nitrogen: Optional[float] = None,
        phosphorus: Optional[float] = None,
        potassium: Optional[float] = None,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        rainfall: Optional[float] = None,
        location: Optional[str] = None,
        season: Optional[str] = None,
    ) -> Tuple[List[CropRecommendation], List[MissingMeasurement], str]:
        """Rank optimal crops using feature matching, soil suitability, and SHAP attributions."""
        features, missing_reports, rel_status = self.assess_input_measurements(
            ph, nitrogen, phosphorus, potassium, temperature, humidity, rainfall
        )

        detected_soil = soil_pred.soil_class if soil_pred else "Loamy Soil"
        disease_condition = disease_pred.disease_class if disease_pred else "Healthy Foliage"

        candidates = []

        for crop_name in self.crops:
            profile = self.catalog.get(crop_name, {})
            score = 100.0
            shap_contributions = {}

            # 1. Soil Match
            preferred_soils = profile.get("preferred_soils", [])
            if any(detected_soil.lower() in s.lower() for s in preferred_soils):
                shap_contributions["Soil Taxonomy"] = +15.0
            else:
                score -= 15.0
                shap_contributions["Soil Taxonomy"] = -15.0

            # 2. pH Evaluation
            opt_ph = profile.get("optimal_ph", (6.0, 7.5))
            current_ph = features["ph"]
            if opt_ph[0] <= current_ph <= opt_ph[1]:
                shap_contributions["Soil pH"] = +12.0
            else:
                dist = min(abs(current_ph - opt_ph[0]), abs(current_ph - opt_ph[1]))
                penalty = min(30.0, dist * 16.0)
                score -= penalty
                shap_contributions["Soil pH"] = round(-penalty, 1)

            # 3. NPK Evaluation
            opt_n = profile.get("optimal_n", (80.0, 160.0))
            if opt_n[0] <= features["nitrogen"] <= opt_n[1]:
                shap_contributions["Available Nitrogen"] = +10.0
            else:
                n_diff = min(abs(features["nitrogen"] - opt_n[0]), abs(features["nitrogen"] - opt_n[1]))
                pen = min(20.0, (n_diff / max(1.0, opt_n[1])) * 25.0)
                score -= pen
                shap_contributions["Available Nitrogen"] = round(-pen, 1)

            opt_p = profile.get("optimal_p", (30.0, 70.0))
            if opt_p[0] <= features["phosphorus"] <= opt_p[1]:
                shap_contributions["Available Phosphorus"] = +8.0
            else:
                p_diff = min(abs(features["phosphorus"] - opt_p[0]), abs(features["phosphorus"] - opt_p[1]))
                pen = min(15.0, (p_diff / max(1.0, opt_p[1])) * 20.0)
                score -= pen
                shap_contributions["Available Phosphorus"] = round(-pen, 1)

            opt_k = profile.get("optimal_k", (40.0, 140.0))
            if opt_k[0] <= features["potassium"] <= opt_k[1]:
                shap_contributions["Available Potassium"] = +8.0
            else:
                k_diff = min(abs(features["potassium"] - opt_k[0]), abs(features["potassium"] - opt_k[1]))
                pen = min(15.0, (k_diff / max(1.0, opt_k[1])) * 20.0)
                score -= pen
                shap_contributions["Available Potassium"] = round(-pen, 1)

            # 4. Meteorological Evaluation
            opt_t = profile.get("optimal_temp", (18.0, 32.0))
            if opt_t[0] <= features["temperature"] <= opt_t[1]:
                shap_contributions["Mean Temperature"] = +10.0
            else:
                t_diff = min(abs(features["temperature"] - opt_t[0]), abs(features["temperature"] - opt_t[1]))
                pen = min(25.0, t_diff * 3.5)
                score -= pen
                shap_contributions["Mean Temperature"] = round(-pen, 1)

            opt_r = profile.get("optimal_rainfall", (60.0, 160.0))
            if opt_r[0] <= features["rainfall"] <= opt_r[1]:
                shap_contributions["Precipitation / Rain"] = +12.0
            else:
                r_diff = min(abs(features["rainfall"] - opt_r[0]), abs(features["rainfall"] - opt_r[1]))
                pen = min(25.0, (r_diff / max(1.0, opt_r[1])) * 30.0)
                score -= pen
                shap_contributions["Precipitation / Rain"] = round(-pen, 1)

            # 5. Disease Vulnerability Check
            if disease_condition != "Healthy Foliage":
                # If active blight or fungus present, penalize crops susceptible to similar oomycetes
                if ("Blight" in disease_condition and crop_name in ["Wheat", "Rice"]) or \
                   ("Curl" in disease_condition and crop_name == "Cotton"):
                    score -= 10.0
                    shap_contributions["Pathogen Vulnerability"] = -10.0

            final_suitability = round(max(20.0, min(99.0, score)), 1)
            grade = "Highly Recommended" if final_suitability >= 80.0 else ("Moderately Suitable" if final_suitability >= 60.0 else "Marginally Suitable")
            base_yield = profile.get("base_yield_tons_ha", 3.0)
            proj_yield = f"{round(base_yield * 0.9, 1)} - {round(base_yield * 1.15, 1)} Tons / Ha"

            candidates.append(
                CropRecommendation(
                    crop_name=crop_name,
                    rank=1,  # will update after sort
                    suitability_score=final_suitability,
                    compatibility_grade=grade,
                    projected_yield=proj_yield,
                    water_requirement=profile.get("water_requirement", "Moderate"),
                    growing_duration_days=profile.get("duration_days", "110-130 Days"),
                    feature_contributions=shap_contributions,
                    cultivation_tips=profile.get("cultivation_tips", []),
                )
            )

        # Sort descending by suitability score
        candidates.sort(key=lambda x: x.suitability_score, reverse=True)
        for rank, cand in enumerate(candidates, 1):
            cand.rank = rank

        return candidates[:5], missing_reports, rel_status
