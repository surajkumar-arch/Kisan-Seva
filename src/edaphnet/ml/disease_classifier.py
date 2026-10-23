"""Load and run the trained multi-crop MobileNetV3 disease classifier."""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.edaphnet.domain.models import DiseasePrediction
from src.edaphnet.ml.calibration import TemperatureScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DISEASE_MODEL_DIR = PROJECT_ROOT / "models" / "plant_health" / "multi_crop_69_class"
DEFAULT_DISEASE_CHECKPOINT = DEFAULT_DISEASE_MODEL_DIR / "plant_mobilenet_v3.pt"
DEFAULT_DISEASE_METADATA = DEFAULT_DISEASE_MODEL_DIR / "class_labels.json"

def _metadata_for_class(class_name: str) -> Dict[str, str]:
    """Return conservative display metadata inferred from a detailed class label."""
    normalized = class_name.casefold()
    if "healthy" in normalized:
        return {"pathogen": "None", "severity": "None"}
    if "tylc" in normalized or "yellow_leaf_curl" in normalized:
        return {"pathogen": "Viral (Tomato yellow leaf curl virus)", "severity": "Not assessed"}
    if any(term in normalized for term in ("virus", "mosaic", "leafcurl", "moko")):
        return {"pathogen": "Viral (specific agent not established)", "severity": "Not assessed"}
    if "bacterial" in normalized:
        return {"pathogen": "Bacterial (species not established)", "severity": "Not assessed"}
    if "late_blight" in normalized:
        return {"pathogen": "Oomycete (class label)", "severity": "Not assessed"}
    if any(term in normalized for term in ("blight", "rust", "mildew", "scab", "esca")):
        return {"pathogen": "Likely fungal (class label)", "severity": "Not assessed"}
    if any(term in normalized for term in ("spot", "rot")):
        return {"pathogen": "Not specified by the dataset label", "severity": "Not assessed"}
    return {"pathogen": "Not specified by the dataset label", "severity": "Not assessed"}


class EfficientNetDiseaseClassifier(nn.Module):
    """MobileNetV3-Large classifier using the saved multi-crop checkpoint."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_DISEASE_CHECKPOINT,
        metadata_path: Path = DEFAULT_DISEASE_METADATA,
    ):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        metadata_path = Path(metadata_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Trained disease checkpoint not found: {checkpoint_path}. "
                "Train it with train_plant_disease.py before starting the app."
            )
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Disease class metadata not found: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        class_names = metadata.get("classes")
        if not isinstance(class_names, list) or not class_names:
            raise ValueError(f"No disease classes listed in {metadata_path}")
        if len(set(class_names)) != len(class_names):
            raise ValueError(f"Duplicate disease classes listed in {metadata_path}")
        self.class_names = [str(name) for name in class_names]
        self.num_classes = len(self.class_names)
        self.checkpoint_path = checkpoint_path

        # Match train_plant_disease.py's MobileNetV3-Large head exactly.
        self.backbone = models.mobilenet_v3_large(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=self.num_classes),
        )
        self.temp_scaler = TemperatureScaler(initial_temperature=1.0)

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported disease checkpoint format: {checkpoint_path}")
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        incompatible = self.backbone.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                "Disease checkpoint does not match the configured MobileNetV3 model. "
                f"Missing keys: {incompatible.missing_keys}; "
                f"unexpected keys: {incompatible.unexpected_keys}"
            )

        self.is_calibrated = "temperature" in metadata
        if self.is_calibrated:
            temperature = float(metadata["temperature"])
            if temperature <= 0:
                raise ValueError("Disease calibration temperature must be positive")
            self.temp_scaler.temperature.data.fill_(temperature)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return classifier logits and pooled 960-dimensional MobileNet features."""
        features = self.backbone.features(x)
        pooled = self.backbone.avgpool(features)
        embeddings = torch.flatten(pooled, 1)
        logits = self.backbone.classifier(embeddings)
        return logits, embeddings

    def predict(self, pil_image: Image.Image) -> Tuple[DiseasePrediction, np.ndarray]:
        """Classify one leaf image with the trained model and optional calibration."""
        self.eval()
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, embedding = self.forward(img_tensor)
            raw_probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            probabilities = (
                self.temp_scaler.get_calibrated_probabilities(logits).squeeze(0).cpu().numpy()
                if self.is_calibrated
                else raw_probs
            )

        top_idx = int(np.argmax(probabilities))
        top_class = self.class_names[top_idx]
        meta = _metadata_for_class(top_class)
        probability_map = {
            self.class_names[index]: round(float(probabilities[index]), 4)
            for index in range(self.num_classes)
        }
        prediction = DiseasePrediction(
            disease_class=top_class,
            confidence=round(float(probabilities[top_idx]), 4),
            raw_confidence=round(float(raw_probs[top_idx]), 4),
            calibrated=self.is_calibrated,
            pathogen_type=meta["pathogen"],
            severity_level=meta["severity"],
            probabilities=probability_map,
            model_version="2.1.0",
        )
        return prediction, embedding.squeeze(0).cpu().numpy()


# Production alias
PlantDiseaseClassifier = EfficientNetDiseaseClassifier
