"""PyTorch MobileNetV3-Large Plant Disease Classification Architecture for Phase 3."""

import os
from pathlib import Path
from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torchvision import models

PLANT_DISEASE_CLASSES: List[str] = [
    "Healthy Foliage",
    "Early Blight",
    "Late Blight",
    "Powdery Mildew",
    "Leaf Rust",
    "Bacterial Spot",
    "Yellow Leaf Curl",
]

# Agronomic metadata and treatment protocols
PLANT_DISEASE_METADATA: Dict[str, Dict[str, any]] = {
    "Healthy Foliage": {
        "pathogen_type": "None",
        "severity_level": "None",
        "symptoms": "Vibrant green lamina, uniform pigmentation, intact cuticle, and normal vein structure with no necrotic lesions.",
        "organic_treatment": [
            "Maintain current organic regimen with regular compost tea or seaweed extract foliar spray.",
            "Apply beneficial microbial inoculants (e.g., Trichoderma) to preserve root zone immunity.",
        ],
        "chemical_treatment": [
            "No chemical intervention needed.",
        ],
        "preventive_measures": [
            "Maintain balanced N-P-K fertilization to avoid excess vegetative lushness susceptible to fungi.",
            "Ensure adequate crop spacing for uniform canopy air circulation.",
        ],
    },
    "Early Blight": {
        "pathogen_type": "Fungal (Alternaria solani)",
        "severity_level": "Moderate",
        "symptoms": "Dark brown to black necrotic spots with characteristic concentric rings ('bullseye' target pattern) starting on older lower foliage.",
        "organic_treatment": [
            "Foliar spray of 5% Neem Seed Kernel Extract (NSKE) or cold-pressed Neem oil (3 ml/L).",
            "Apply bio-fungicide Bacillus subtilis or Trichoderma harzianum at first symptom appearance.",
            "Remove and incinerate severely infected lower leaves.",
        ],
        "chemical_treatment": [
            "Foliar application of Mancozeb 75% WP @ 2.5 g/L or Chlorothalonil 75% WP @ 2 g/L.",
            "In progressive infection, spray Azoxystrobin 23% SC @ 1 ml/L or Difenoconazole 25% EC @ 0.5 ml/L.",
        ],
        "preventive_measures": [
            "Avoid overhead sprinkler irrigation; wet foliage promotes fungal sporulation.",
            "Practice 3-year crop rotation with non-solanaceous crops.",
            "Apply straw mulch to prevent soil-splashing onto lower leaves.",
        ],
    },
    "Late Blight": {
        "pathogen_type": "Oomycete / Fungal (Phytophthora infestans)",
        "severity_level": "Severe",
        "symptoms": "Large, irregular water-soaked pale-green to brown lesions rapidly expanding from leaf tips/margins, with white velvety fungal growth on leaf undersides in high humidity.",
        "organic_treatment": [
            "Foliar spray of Copper Hydroxide or Bordeaux mixture (1%) as a protective barrier.",
            "Apply fermented horsetail extract or compost tea rich in antagonistic microbes.",
            "Immediately excise and destroy infected plant debris outside the field perimeter.",
        ],
        "chemical_treatment": [
            "Apply systemic fungicide Metalaxyl 8% + Mancozeb 64% WP @ 2.5 g/L.",
            "Alternate with Dimethomorph 50% WP @ 1 g/L or Cymoxanil 8% + Mancozeb 64% WP @ 2 g/L.",
            "Repeat at 7-10 day intervals during cool, humid, overcast weather.",
        ],
        "preventive_measures": [
            "Plant certified disease-resistant seeds and certified tuber stock.",
            "Ensure wide row spacing to maximize sunlight penetration and foliage drying.",
            "Eliminate volunteer host plants and weed reservoirs (e.g., wild nightshades).",
        ],
    },
    "Powdery Mildew": {
        "pathogen_type": "Fungal (Erysiphe / Podosphaera)",
        "severity_level": "Moderate",
        "symptoms": "Distinct white or grayish talcum-powder-like circular patches on upper leaf surfaces, progressing to yellowing, curling, and premature defoliation.",
        "organic_treatment": [
            "Foliar spray of Potassium Bicarbonate (3 g/L) or diluted milk solution (1:9 with water).",
            "Apply wettable sulfur @ 2 g/L during early morning (avoid spraying above 32°C).",
            "Spray Ampelomyces quisqualis bio-fungicide.",
        ],
        "chemical_treatment": [
            "Foliar spray of Hexaconazole 5% EC @ 1 ml/L or Myclobutanil 10% WP @ 1 g/L.",
            "Apply Penconazole 10% EC @ 0.5 ml/L or Triadimefon 25% WP @ 1 g/L.",
        ],
        "preventive_measures": [
            "Prune overcrowded stems to enhance sunlight penetration and reduce microclimate humidity.",
            "Avoid excessive synthetic nitrogen fertilizers which stimulate soft, susceptible growth.",
        ],
    },
    "Leaf Rust": {
        "pathogen_type": "Fungal (Puccinia spp.)",
        "severity_level": "Moderate",
        "symptoms": "Small, raised orange, reddish-brown, or yellow powdery pustules scattered over the lower and upper leaf surfaces that rupture the epidermis.",
        "organic_treatment": [
            "Spray sulfur-based organic dust or wettable sulfur @ 2 g/L.",
            "Apply bio-agent Verticillium lecanii or Bacillus pumilus.",
            "Prune infected branches and destroy fallen diseased foliage.",
        ],
        "chemical_treatment": [
            "Apply Propiconazole 25% EC @ 1 ml/L or Tebuconazole 25.9% EC @ 1.5 ml/L.",
            "Apply Mancozeb 75% WP @ 2.5 g/L as a protective preventative spray.",
        ],
        "preventive_measures": [
            "Select rust-resistant crop cultivars.",
            "Eradicate alternate host weeds nearby.",
            "Maintain balanced potash nutrition to thicken leaf epidermal cell walls.",
        ],
    },
    "Bacterial Spot": {
        "pathogen_type": "Bacterial (Xanthomonas spp.)",
        "severity_level": "Severe",
        "symptoms": "Small, angular, water-soaked dark brown spots often surrounded by a chlorotic yellow halo; centers may become brittle and drop out ('shot-hole' appearance).",
        "organic_treatment": [
            "Spray Copper oxychloride (COC) @ 2.5 g/L mixed with bio-bactericide Bacillus amyloliquefaciens.",
            "Apply fresh garlic-chili extract spray for natural bactericidal suppression.",
            "Sterilize all pruning shears and farming implements in 10% bleach between plants.",
        ],
        "chemical_treatment": [
            "Foliar spray of Streptomycin sulphate 9% + Tetracycline hydrochloride 1% (Plantomycin/Streptocycline) @ 100 ppm (1 g / 10 L water) mixed with Copper Oxychloride (2.5 g/L).",
        ],
        "preventive_measures": [
            "Use only certified hot-water-treated pathogen-free seeds.",
            "Never cultivate, touch, or harvest plants while leaves are wet with dew or rain.",
            "Install drip irrigation directly at the soil level.",
        ],
    },
    "Yellow Leaf Curl": {
        "pathogen_type": "Viral (Begomovirus / TYLCV)",
        "severity_level": "Severe",
        "symptoms": "Severe upward rolling and cupping of leaflets, marked interveinal yellowing (chlorosis), stunting of plants, and flower drop.",
        "organic_treatment": [
            "Deploy yellow sticky traps (15-20 traps/acre) to monitor and mass-trap insect vectors (Whiteflies - Bemisia tabaci).",
            "Spray Neem oil (5 ml/L) or pongamia soap to repel whitefly populations.",
            "Rogue out and bury or burn virus-infected plants immediately to stop secondary spread.",
        ],
        "chemical_treatment": [
            "Target vector insects: Spray Imidacloprid 17.8% SL @ 0.3 ml/L or Thiamethoxam 25% WG @ 0.4 g/L.",
            "Alternate with Diafenthiuron 50% WP @ 1 g/L or Spiromesifen 22.9% SC @ 1 ml/L.",
        ],
        "preventive_measures": [
            "Use 40-50 mesh insect-proof netting in nursery beds.",
            "Grow barrier crops like maize, sorghum, or pearl millet around the main crop boundary.",
            "Plant virus-resistant hybrid varieties.",
        ],
    },
}

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_WEIGHTS_PATH = BASE_DIR / "models" / "plant_health" / "multi_crop_30_class" / "plant_mobilenet_v3.pt"
LEGACY_WEIGHTS_PATH = BASE_DIR / "models" / "plant_health" / "plant_mobilenet_v3.pt"


class PlantMobileNetV3(nn.Module):
    """MobileNetV3-Large neural network for plant disease diagnosis."""

    def __init__(self, num_classes: int = len(PLANT_DISEASE_CLASSES), pretrained_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes

        try:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
            self.backbone = models.mobilenet_v3_large(weights=weights)
        except Exception:
            self.backbone = models.mobilenet_v3_large(weights=None)

        in_features = self.backbone.classifier[3].in_features
        # Replace final classifier layer (same head layout as train_plant_disease.py)
        self.backbone.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits."""
        return self.backbone(x)


_PLANT_MODEL_INSTANCE: Optional[PlantMobileNetV3] = None
_PLANT_CLASS_NAMES: List[str] = list(PLANT_DISEASE_CLASSES)
_PLANT_MODEL_TRAINED: bool = False


def get_plant_class_names() -> List[str]:
    """Class names matching the loaded model's output layer."""
    get_plant_model()
    return _PLANT_CLASS_NAMES


def is_plant_model_trained() -> bool:
    """True when trained weights were successfully loaded."""
    get_plant_model()
    return _PLANT_MODEL_TRAINED


def get_plant_model(weights_path: Optional[Path] = None) -> PlantMobileNetV3:
    """Retrieve singleton instance of the Plant Disease Classification model."""
    global _PLANT_MODEL_INSTANCE, _PLANT_CLASS_NAMES, _PLANT_MODEL_TRAINED
    if _PLANT_MODEL_INSTANCE is not None:
        return _PLANT_MODEL_INSTANCE

    target_weights = weights_path or DEFAULT_WEIGHTS_PATH
    if not target_weights.exists() and LEGACY_WEIGHTS_PATH.exists():
        target_weights = LEGACY_WEIGHTS_PATH
    labels_file = target_weights.parent / "class_labels.json"
    num_classes = len(PLANT_DISEASE_CLASSES)
    class_names: List[str] = list(PLANT_DISEASE_CLASSES)
    if labels_file.exists():
        try:
            import json
            with open(labels_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if "classes" in meta:
                    class_names = [str(c) for c in meta["classes"]]
                    num_classes = len(class_names)
                elif "num_classes" in meta:
                    num_classes = meta["num_classes"]
        except Exception:
            pass

    model = PlantMobileNetV3(num_classes=num_classes, pretrained_backbone=False)
    model.eval()

    if target_weights.exists():
        try:
            state_dict = torch.load(target_weights, map_location=torch.device("cpu"))
            try:
                # Checkpoints from train_plant_disease.py store raw torchvision keys
                model.backbone.load_state_dict(state_dict)
            except Exception:
                model.load_state_dict(state_dict)
            _PLANT_MODEL_TRAINED = True
            if len(class_names) == num_classes:
                _PLANT_CLASS_NAMES = class_names
        except Exception:
            _PLANT_MODEL_TRAINED = False
            _PLANT_CLASS_NAMES = list(PLANT_DISEASE_CLASSES) if num_classes == len(PLANT_DISEASE_CLASSES) else class_names
    else:
        _PLANT_CLASS_NAMES = class_names

    _PLANT_MODEL_INSTANCE = model
    return _PLANT_MODEL_INSTANCE


def get_disease_metadata(class_name: str) -> Dict[str, any]:
    """Resolve agronomic metadata for any class label (legacy or crop_disease style)."""
    if class_name in PLANT_DISEASE_METADATA:
        return PLANT_DISEASE_METADATA[class_name]
    name = class_name.lower()
    if "healthy" in name:
        base = "Healthy Foliage"
    elif any(k in name for k in ("mosaic", "virus", "leafcurl", "curl", "yellowish", "yb_", "moko", "whitefly")):
        base = "Yellow Leaf Curl"
    elif "rust" in name:
        base = "Leaf Rust"
    elif "mildew" in name:
        base = "Powdery Mildew"
    elif any(k in name for k in ("bacterial", "blackrot", "black_rot", "rot")):
        base = "Bacterial Spot"
    elif "late" in name:
        base = "Late Blight"
    else:
        base = "Early Blight"
    return PLANT_DISEASE_METADATA[base]
