"""PyTorch EfficientNet-B0 Soil Classification Architecture for Phase 2."""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

SOIL_CLASSES: List[str] = [
    "Alluvial Soil",
    "Black Soil",
    "Clay Soil",
    "Red Soil",
    "Sandy Soil",
    "Loamy Soil",
]

# Physical & Agronomic characteristics mapping for each soil class
SOIL_METADATA: Dict[str, Dict[str, str]] = {
    "Alluvial Soil": {
        "texture": "Silty Loam to Clayey",
        "drainage": "Well-drained to Moderate",
        "water_retention": "High",
        "aeration": "Good",
        "organic_matter_estimate": "High (1.2% - 2.5%)",
        "typical_fertility": "High",
    },
    "Black Soil": {
        "texture": "Heavy Clay / Vertisol",
        "drainage": "Slow / Poor",
        "water_retention": "Very High",
        "aeration": "Moderate to Low",
        "organic_matter_estimate": "Moderate (0.8% - 1.5%)",
        "typical_fertility": "High",
    },
    "Clay Soil": {
        "texture": "Dense Fine Clay",
        "drainage": "Poor / Impermeable",
        "water_retention": "Extremely High",
        "aeration": "Poor",
        "organic_matter_estimate": "Moderate (0.6% - 1.2%)",
        "typical_fertility": "Moderate",
    },
    "Red Soil": {
        "texture": "Sandy Loam to Clay Loam (Rich in Iron)",
        "drainage": "Rapid / Well-drained",
        "water_retention": "Low to Moderate",
        "aeration": "Excellent",
        "organic_matter_estimate": "Low to Moderate (0.4% - 1.0%)",
        "typical_fertility": "Moderate",
    },
    "Sandy Soil": {
        "texture": "Coarse Gritty Sand",
        "drainage": "Excessive / Very Rapid",
        "water_retention": "Very Low",
        "aeration": "Excessive",
        "organic_matter_estimate": "Low (< 0.5%)",
        "typical_fertility": "Low",
    },
    "Loamy Soil": {
        "texture": "Balanced Sand, Silt, and Clay",
        "drainage": "Optimal / Well-drained",
        "water_retention": "Optimal",
        "aeration": "Optimal",
        "organic_matter_estimate": "High (1.5% - 3.0%)",
        "typical_fertility": "High",
    },
}

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_WEIGHTS_PATH = BASE_DIR / "models" / "soil_health" / "soil_efficientnet_b0.pt"


class SoilEfficientNet(nn.Module):
    """EfficientNet-B0 adapted for multi-class Soil Health and Type classification."""

    def __init__(self, num_classes: int = len(SOIL_CLASSES), pretrained_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # Load EfficientNet-B0 backbone
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained_backbone else None
            self.backbone = models.efficientnet_b0(weights=weights)
        except Exception:
            self.backbone = models.efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features
        # Replace classifier head with custom dropout + linear head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits."""
        return self.backbone(x)


_MODEL_INSTANCE: Optional[SoilEfficientNet] = None


def get_soil_model(weights_path: Optional[Path] = None) -> SoilEfficientNet:
    """Retrieve or initialize singleton instance of the Soil Classification model."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    target_weights = weights_path or DEFAULT_WEIGHTS_PATH
    model = SoilEfficientNet(num_classes=len(SOIL_CLASSES), pretrained_backbone=True)
    model.eval()

    if target_weights.exists():
        try:
            state_dict = torch.load(target_weights, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
        except Exception:
            pass  # Fallback to backbone initialized model

    _MODEL_INSTANCE = model
    return _MODEL_INSTANCE
