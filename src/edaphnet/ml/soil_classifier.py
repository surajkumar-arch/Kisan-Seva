"""Load and run the trained EfficientNet-B0 soil classifier."""

import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from src.edaphnet.domain.models import SoilPrediction
from src.edaphnet.ml.calibration import TemperatureScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOIL_MODEL_DIR = PROJECT_ROOT / "models" / "soil_health" / "soil_dataset_experiment"
DEFAULT_SOIL_CHECKPOINT = DEFAULT_SOIL_MODEL_DIR / "soil_efficientnet_b0.pt"
DEFAULT_SOIL_METADATA = DEFAULT_SOIL_MODEL_DIR / "class_labels.json"

SOIL_CLASSES: List[str] = ["Alluvial Soil", "Black Soil", "Clay Soil", "Red Soil"]


class EfficientNetSoilClassifier(nn.Module):
    """EfficientNet-B0 Soil Taxonomy Classifier with Feature Extractor."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_SOIL_CHECKPOINT,
        metadata_path: Path = DEFAULT_SOIL_METADATA,
    ):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        metadata_path = Path(metadata_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Trained soil checkpoint not found: {checkpoint_path}. "
                "Train it with train_soil.py before starting the app."
            )
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Soil class metadata not found: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        class_names = metadata.get("classes")
        if not isinstance(class_names, list) or not class_names:
            raise ValueError(f"No soil classes listed in {metadata_path}")
        if len(set(class_names)) != len(class_names):
            raise ValueError(f"Duplicate soil classes listed in {metadata_path}")
        self.class_names = [str(name) for name in class_names]
        self.num_classes = len(self.class_names)
        self.checkpoint_path = checkpoint_path

        # The full trained checkpoint contains the backbone and classifier, so
        # don't create an ImageNet-only backbone with a fresh random head.
        backbone = models.efficientnet_b0(weights=None)

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, self.num_classes),
        )

        self.temp_scaler = TemperatureScaler(initial_temperature=1.0)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported soil checkpoint format: {checkpoint_path}")
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        incompatible = self.load_state_dict(state_dict, strict=False)
        unexpected = incompatible.unexpected_keys
        missing = [key for key in incompatible.missing_keys if key != "temp_scaler.temperature"]
        if missing or unexpected:
            raise ValueError(
                "Soil checkpoint does not match the configured model. "
                f"Missing keys: {missing}; unexpected keys: {unexpected}"
            )

        # This training artifact has not been temperature-calibrated. Keep the
        # raw softmax confidence honest until a validation-fitted temperature is saved.
        self.is_calibrated = "temperature" in metadata
        if self.is_calibrated:
            temperature = float(metadata["temperature"])
            if temperature <= 0:
                raise ValueError("Soil calibration temperature must be positive")
            self.temp_scaler.temperature.data.fill_(temperature)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled 1280-dimensional latent embedding."""
        feat = self.features(x)
        pooled = self.avgpool(feat)
        return torch.flatten(pooled, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and latent embedding."""
        embeddings = self.forward_features(x)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def predict(self, pil_image: Image.Image) -> Tuple[SoilPrediction, np.ndarray]:
        """Execute calibrated inference on a PIL image."""
        self.eval()
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, embedding = self.forward(img_tensor)
            raw_probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            calibrated_probs = self.temp_scaler.get_calibrated_probabilities(logits).squeeze(0).cpu().numpy()

        top_idx = int(np.argmax(calibrated_probs))
        top_class = self.class_names[top_idx]
        calibrated_conf = float(calibrated_probs[top_idx])
        raw_conf = float(raw_probs[top_idx])

        prob_dist = {self.class_names[i]: round(float(calibrated_probs[i]), 4) for i in range(self.num_classes)}
        embedding_np = embedding.squeeze(0).cpu().numpy()

        prediction = SoilPrediction(
            soil_class=top_class,
            confidence=round(calibrated_conf, 4),
            raw_confidence=round(raw_conf, 4),
            calibrated=self.is_calibrated,
            probabilities=prob_dist,
            model_version="1.3.0",
        )

        return prediction, embedding_np


# Production alias
SoilClassifier = EfficientNetSoilClassifier

