"""PyTorch Neural Crop Recommendation Model and Agronomic Rules for Phase 5."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn

CROPS_LIST: List[str] = [
    "Rice",
    "Wheat",
    "Cotton",
    "Maize",
    "Chickpea",
    "Kidney Beans",
    "Pigeon Peas",
    "Moth Beans",
    "Groundnut",
    "Sugarcane",
    "Coffee",
    "Tea",
]

# Agronomic profiles, environmental boundaries, and management guidelines
CROP_CATALOG: Dict[str, Dict[str, Any]] = {
    "Rice": {
        "preferred_soils": ["Alluvial Soil", "Clay Soil", "Loamy Soil"],
        "optimal_ph": (5.5, 7.2),
        "optimal_n": (120.0, 200.0),
        "optimal_p": (30.0, 60.0),
        "optimal_k": (120.0, 200.0),
        "optimal_temp": (20.0, 35.0),
        "optimal_rainfall": (120.0, 300.0),
        "water_requirement": "Very High",
        "duration_days": "115 - 135 Days",
        "base_yield_tons_ha": 4.5,
        "cultivation_tips": [
            "Maintain 2-5 cm standing water layer during early tillering.",
            "Apply nitrogen in 3 split doses: basal, active tillering, and panicle initiation.",
            "Practice alternate wetting and drying (AWD) to conserve water without penalizing yield.",
        ],
    },
    "Wheat": {
        "preferred_soils": ["Alluvial Soil", "Loamy Soil", "Black Soil"],
        "optimal_ph": (6.0, 7.5),
        "optimal_n": (100.0, 160.0),
        "optimal_p": (40.0, 70.0),
        "optimal_k": (30.0, 60.0),
        "optimal_temp": (12.0, 25.0),
        "optimal_rainfall": (40.0, 100.0),
        "water_requirement": "Moderate",
        "duration_days": "120 - 140 Days",
        "base_yield_tons_ha": 4.2,
        "cultivation_tips": [
            "Ensure first critical irrigation at Crown Root Initiation (CRI) stage (20-25 days after sowing).",
            "Apply balanced fertilizer dose (120:60:40 NPK kg/ha).",
            "Maintain cool, weed-free seedbed conditions during early vegetative growth.",
        ],
    },
    "Cotton": {
        "preferred_soils": ["Black Soil", "Alluvial Soil", "Red Soil"],
        "optimal_ph": (6.0, 8.2),
        "optimal_n": (80.0, 150.0),
        "optimal_p": (30.0, 60.0),
        "optimal_k": (60.0, 120.0),
        "optimal_temp": (22.0, 38.0),
        "optimal_rainfall": (60.0, 120.0),
        "water_requirement": "Moderate",
        "duration_days": "150 - 180 Days",
        "base_yield_tons_ha": 2.2,
        "cultivation_tips": [
            "Deep black vertisols with high moisture retention are ideal.",
            "Avoid waterlogging during boll formation; monitor closely for sucking pests.",
            "Apply foliar potassium nitrate (KNO3) during peak blooming to increase boll weight.",
        ],
    },
    "Maize": {
        "preferred_soils": ["Loamy Soil", "Alluvial Soil", "Red Soil"],
        "optimal_ph": (5.8, 7.5),
        "optimal_n": (100.0, 160.0),
        "optimal_p": (40.0, 70.0),
        "optimal_k": (80.0, 140.0),
        "optimal_temp": (18.0, 32.0),
        "optimal_rainfall": (60.0, 140.0),
        "water_requirement": "Moderate",
        "duration_days": "90 - 110 Days",
        "base_yield_tons_ha": 5.5,
        "cultivation_tips": [
            "Heavy feeder: ensure adequate basal phosphorus and potassium.",
            "Knee-high and tasseling/silking stages are critical for moisture availability.",
            "Ridge planting is recommended to facilitate drainage during heavy rains.",
        ],
    },
    "Chickpea": {
        "preferred_soils": ["Sandy Soil", "Loamy Soil", "Black Soil"],
        "optimal_ph": (6.0, 8.0),
        "optimal_n": (20.0, 50.0),
        "optimal_p": (40.0, 80.0),
        "optimal_k": (20.0, 50.0),
        "optimal_temp": (15.0, 28.0),
        "optimal_rainfall": (30.0, 80.0),
        "water_requirement": "Low",
        "duration_days": "95 - 115 Days",
        "base_yield_tons_ha": 1.8,
        "cultivation_tips": [
            "Inoculate seeds with Rhizobium culture to fix atmospheric nitrogen.",
            "Requires light, well-drained soil; highly sensitive to waterlogging.",
            "One pre-flowering and one pod development irrigation gives optimal return.",
        ],
    },
    "Kidney Beans": {
        "preferred_soils": ["Loamy Soil", "Alluvial Soil"],
        "optimal_ph": (5.5, 6.8),
        "optimal_n": (20.0, 45.0),
        "optimal_p": (55.0, 80.0),
        "optimal_k": (20.0, 45.0),
        "optimal_temp": (15.0, 26.0),
        "optimal_rainfall": (70.0, 130.0),
        "water_requirement": "Moderate",
        "duration_days": "100 - 120 Days",
        "base_yield_tons_ha": 2.0,
        "cultivation_tips": [
            "Incorporate organic matter; kidney beans thrive in high organic carbon loams.",
            "Avoid alkaline soils with free calcium carbonate.",
            "Irrigate regularly to maintain uniform root zone moisture.",
        ],
    },
    "Pigeon Peas": {
        "preferred_soils": ["Red Soil", "Black Soil", "Alluvial Soil"],
        "optimal_ph": (6.0, 7.8),
        "optimal_n": (20.0, 40.0),
        "optimal_p": (50.0, 75.0),
        "optimal_k": (20.0, 40.0),
        "optimal_temp": (20.0, 35.0),
        "optimal_rainfall": (50.0, 110.0),
        "water_requirement": "Low",
        "duration_days": "140 - 180 Days",
        "base_yield_tons_ha": 1.9,
        "cultivation_tips": [
            "Extremely deep taproot provides excellent drought resilience.",
            "Ideal for intercropping with cereals (e.g. Maize or Sorghum).",
            "Apply starter nitrogen (20 kg/ha) and adequate phosphorus (50 kg/ha SSP).",
        ],
    },
    "Moth Beans": {
        "preferred_soils": ["Sandy Soil", "Red Soil"],
        "optimal_ph": (6.2, 8.5),
        "optimal_n": (15.0, 35.0),
        "optimal_p": (35.0, 60.0),
        "optimal_k": (15.0, 35.0),
        "optimal_temp": (24.0, 40.0),
        "optimal_rainfall": (20.0, 60.0),
        "water_requirement": "Very Low",
        "duration_days": "75 - 90 Days",
        "base_yield_tons_ha": 1.0,
        "cultivation_tips": [
            "Highly drought-hardy legume suitable for arid and sandy tracts.",
            "Spreading habit provides natural ground cover against wind erosion.",
            "Requires minimal fertilizer; fixes soil nitrogen effectively.",
        ],
    },
    "Groundnut": {
        "preferred_soils": ["Sandy Soil", "Red Soil", "Loamy Soil"],
        "optimal_ph": (6.0, 7.2),
        "optimal_n": (20.0, 40.0),
        "optimal_p": (40.0, 70.0),
        "optimal_k": (40.0, 80.0),
        "optimal_temp": (22.0, 32.0),
        "optimal_rainfall": (50.0, 100.0),
        "water_requirement": "Low to Moderate",
        "duration_days": "105 - 125 Days",
        "base_yield_tons_ha": 2.5,
        "cultivation_tips": [
            "Friable, light sandy loam is mandatory for easy peg penetration and pod development.",
            "Apply gypsum @ 400 kg/ha at flowering to supply calcium and sulfur for kernel filling.",
            "Avoid soils prone to crusting or water stagnation.",
        ],
    },
    "Sugarcane": {
        "preferred_soils": ["Alluvial Soil", "Black Soil", "Loamy Soil"],
        "optimal_ph": (6.5, 7.8),
        "optimal_n": (180.0, 300.0),
        "optimal_p": (60.0, 100.0),
        "optimal_k": (120.0, 250.0),
        "optimal_temp": (24.0, 38.0),
        "optimal_rainfall": (140.0, 280.0),
        "water_requirement": "Very High",
        "duration_days": "300 - 365 Days",
        "base_yield_tons_ha": 75.0,
        "cultivation_tips": [
            "High biomass crop requiring deep, well-drained, nutrient-dense soils.",
            "Heavy applications of nitrogen and potash produce superior cane girth and sucrose.",
            "Trash mulching conserves moisture and suppresses weed germination.",
        ],
    },
    "Coffee": {
        "preferred_soils": ["Red Soil", "Loamy Soil"],
        "optimal_ph": (5.0, 6.5),
        "optimal_n": (80.0, 140.0),
        "optimal_p": (30.0, 60.0),
        "optimal_k": (80.0, 150.0),
        "optimal_temp": (15.0, 28.0),
        "optimal_rainfall": (120.0, 250.0),
        "water_requirement": "High",
        "duration_days": "Perennial (Annual harvest)",
        "base_yield_tons_ha": 1.5,
        "cultivation_tips": [
            "Subtropical hill slopes with good drainage and shade canopy (two-tier shade).",
            "Acidic soil rich in humus and organic leaf litter is mandatory.",
            "Maintain soil mulch and prevent erosion on sloping topography.",
        ],
    },
    "Tea": {
        "preferred_soils": ["Red Soil", "Loamy Soil"],
        "optimal_ph": (4.5, 5.5),
        "optimal_n": (100.0, 160.0),
        "optimal_p": (25.0, 50.0),
        "optimal_k": (60.0, 120.0),
        "optimal_temp": (14.0, 27.0),
        "optimal_rainfall": (150.0, 300.0),
        "water_requirement": "Very High",
        "duration_days": "Perennial (Regular plucking)",
        "base_yield_tons_ha": 2.2,
        "cultivation_tips": [
            "Strictly requires acidic, well-aerated, calcium-free soils (pH 4.5-5.5).",
            "Gentle hill slopes with well-drained subsoil prevent stagnant moisture.",
            "Regular light prunings and nitrogen-potash top-dressings sustain flushes.",
        ],
    },
}

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_WEIGHTS_PATH = BASE_DIR / "models" / "crop" / "crop_recommendation_net.pt"


class CropRecommendationNet(nn.Module):
    """Neural Multi-Class Crop Suitability Classifier.

    Maps 128-dimensional multi-modal latent space into 12 crop affinity logits.
    """

    def __init__(self, in_features: int = 128, num_crops: int = len(CROPS_LIST)):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_crops),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized compatibility logits."""
        return self.classifier(x)


_CROP_MODEL_INSTANCE: Optional[CropRecommendationNet] = None


def get_crop_model(weights_path: Optional[Path] = None) -> CropRecommendationNet:
    """Retrieve singleton instance of CropRecommendationNet."""
    global _CROP_MODEL_INSTANCE
    if _CROP_MODEL_INSTANCE is not None:
        return _CROP_MODEL_INSTANCE

    target_weights = weights_path or DEFAULT_WEIGHTS_PATH
    model = CropRecommendationNet(in_features=128, num_crops=len(CROPS_LIST))
    model.eval()

    if target_weights.exists():
        try:
            state_dict = torch.load(target_weights, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
        except Exception:
            pass

    _CROP_MODEL_INSTANCE = model
    return _CROP_MODEL_INSTANCE
