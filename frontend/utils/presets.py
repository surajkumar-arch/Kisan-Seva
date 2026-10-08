"""Preset scenarios and demo samples for Phase 7 Farmer Guidance Wizard."""

import io
from typing import Dict, Any, Tuple
from PIL import Image, ImageDraw


def generate_synthetic_field_image(img_type: str, condition_or_soil: str) -> io.BytesIO:
    """Generate representative synthetic test imagery for presets."""
    img = Image.new("RGB", (300, 300), color=(100, 100, 100))
    draw = ImageDraw.Draw(img)

    if img_type == "soil":
        if "Red" in condition_or_soil:
            base_color = (185, 65, 45)
        elif "Black" in condition_or_soil:
            base_color = (45, 40, 38)
        elif "Sandy" in condition_or_soil:
            base_color = (215, 185, 120)
        elif "Alluvial" in condition_or_soil:
            base_color = (135, 110, 80)
        elif "Clay" in condition_or_soil:
            base_color = (110, 95, 75)
        else:
            base_color = (95, 80, 60)

        # Draw textured soil
        for y in range(0, 300, 10):
            for x in range(0, 300, 10):
                jitter = (x * 7 + y * 13) % 25 - 12
                c = tuple(max(0, min(255, val + jitter)) for val in base_color)
                draw.rectangle([x, y, x + 10, y + 10], fill=c)
    else:
        # Plant foliage
        base_green = (35, 140, 45)
        draw.rectangle([0, 0, 300, 300], fill=base_green)

        # Draw leaf vein
        draw.line([(150, 20), (150, 280)], fill=(70, 180, 80), width=4)
        for i in range(40, 260, 30):
            draw.line([(150, i), (60, i + 35)], fill=(60, 170, 70), width=2)
            draw.line([(150, i), (240, i + 35)], fill=(60, 170, 70), width=2)

        # Add disease spots if not healthy
        if "Blight" in condition_or_soil or "Spot" in condition_or_soil or "Rust" in condition_or_soil:
            spot_color = (75, 45, 25) if "Blight" in condition_or_soil else (175, 95, 20)
            draw.ellipse([80, 90, 130, 140], fill=spot_color)
            draw.ellipse([180, 140, 230, 190], fill=spot_color)
            draw.ellipse([100, 190, 140, 230], fill=spot_color)
        elif "Powdery" in condition_or_soil:
            # White fungal patches
            draw.ellipse([80, 90, 140, 140], fill=(230, 235, 230))
            draw.ellipse([170, 150, 220, 200], fill=(225, 230, 225))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf


FARM_PRESETS: Dict[str, Dict[str, Any]] = {
    "🌾 Optimal Alluvial Rice Paddy": {
        "description": "Fertile alluvial basin with high water availability, balanced NPK, and healthy foliage.",
        "soil_type": "Alluvial Soil",
        "plant_condition": "Healthy Foliage",
        "ph": 6.6,
        "nitrogen": 175.0,
        "phosphorus": 45.0,
        "potassium": 180.0,
        "temperature": 28.0,
        "humidity": 75.0,
        "rainfall": 170.0,
    },
    "🍂 Tomato Field with Early Blight Outbreak": {
        "description": "Slightly acidic red soil showing target-board concentric leaf lesions (Early Blight).",
        "soil_type": "Red Soil",
        "plant_condition": "Early Blight",
        "ph": 5.4,
        "nitrogen": 95.0,
        "phosphorus": 22.0,
        "potassium": 115.0,
        "temperature": 29.0,
        "humidity": 82.0,
        "rainfall": 85.0,
    },
    "🥔 Potato Crop with Late Blight Hazard": {
        "description": "Cool humid clay plot experiencing water-soaked fungal lesions (Late Blight).",
        "soil_type": "Clay Soil",
        "plant_condition": "Late Blight",
        "ph": 5.9,
        "nitrogen": 130.0,
        "phosphorus": 36.0,
        "potassium": 140.0,
        "temperature": 20.5,
        "humidity": 92.0,
        "rainfall": 155.0,
    },
    "🥜 Semi-Arid Sandy Groundnut Plot": {
        "description": "Light sandy soil in semi-arid conditions, optimal for pod penetration but prone to moisture deficit.",
        "soil_type": "Sandy Soil",
        "plant_condition": "Healthy Foliage",
        "ph": 6.8,
        "nitrogen": 85.0,
        "phosphorus": 50.0,
        "potassium": 95.0,
        "temperature": 32.0,
        "humidity": 48.0,
        "rainfall": 45.0,
    },
    "🍃 Subtropical Highland Tea Estate": {
        "description": "Acidic highland loam prone to powdery mildew under high morning mist.",
        "soil_type": "Loamy Soil",
        "plant_condition": "Powdery Mildew",
        "ph": 4.9,
        "nitrogen": 155.0,
        "phosphorus": 28.0,
        "potassium": 165.0,
        "temperature": 22.0,
        "humidity": 85.0,
        "rainfall": 210.0,
    },
}
