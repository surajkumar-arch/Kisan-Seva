"""EfficientNet-B0 Plant Foliage Disease Classifier with Post-Hoc Calibration."""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from src.edaphnet.domain.models import DiseasePrediction
from src.edaphnet.ml.calibration import TemperatureScaler

DISEASE_CLASSES: List[str] = [
    "Healthy Foliage",
    "Early Blight",
    "Late Blight",
    "Powdery Mildew",
    "Leaf Rust",
    "Bacterial Spot",
    "Yellow Leaf Curl",
]

DISEASE_METADATA: Dict[str, Dict[str, str]] = {
    "Healthy Foliage": {"pathogen": "None", "severity": "None"},
    "Early Blight": {"pathogen": "Fungal (Alternaria solani)", "severity": "Moderate"},
    "Late Blight": {"pathogen": "Oomycete (Phytophthora infestans)", "severity": "Severe"},
    "Powdery Mildew": {"pathogen": "Fungal (Erysiphe / Podosphaera)", "severity": "Moderate"},
    "Leaf Rust": {"pathogen": "Fungal (Puccinia spp.)", "severity": "Moderate"},
    "Bacterial Spot": {"pathogen": "Bacterial (Xanthomonas spp.)", "severity": "Severe"},
    "Yellow Leaf Curl": {"pathogen": "Viral (Begomovirus / TYLCV)", "severity": "Severe"},
}


class EfficientNetDiseaseClassifier(nn.Module):
    """EfficientNet-B0 Foliar Pathology Classifier with Feature Extractor."""

    def __init__(self, num_classes: int = len(DISEASE_CLASSES), pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
        except Exception:
            backbone = models.efficientnet_b0(weights=None)

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        self.temp_scaler = TemperatureScaler(initial_temperature=1.08)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.avgpool(feat)
        return torch.flatten(pooled, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.forward_features(x)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def predict(self, pil_image: Image.Image) -> Tuple[DiseasePrediction, np.ndarray]:
        """Perform calibrated disease prediction and extract latent representation."""
        self.eval()
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, embedding = self.forward(img_tensor)
            raw_probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            calibrated_probs = self.temp_scaler.get_calibrated_probabilities(logits).squeeze(0).cpu().numpy()

        top_idx = int(np.argmax(calibrated_probs))
        top_class = DISEASE_CLASSES[top_idx]
        calibrated_conf = float(calibrated_probs[top_idx])
        raw_conf = float(raw_probs[top_idx])

        meta = DISEASE_METADATA.get(top_class, {"pathogen": "Unknown", "severity": "Moderate"})
        prob_dist = {DISEASE_CLASSES[i]: round(float(calibrated_probs[i]), 4) for i in range(self.num_classes)}
        embedding_np = embedding.squeeze(0).cpu().numpy()

        prediction = DiseasePrediction(
            disease_class=top_class,
            confidence=round(calibrated_conf, 4),
            raw_confidence=round(raw_conf, 4),
            pathogen_type=meta["pathogen"],
            severity_level=meta["severity"],
            probabilities=prob_dist,
            model_version="1.3.0",
        )

        return prediction, embedding_np


# Production alias
PlantDiseaseClassifier = EfficientNetDiseaseClassifier

