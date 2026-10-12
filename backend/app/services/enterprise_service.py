"""Advanced Enterprise Services for KisanSeva.

1. Multilingual LLM Agronomist (Gemini 1.5 with Intelligent Fallback)
2. Hyperlocal Live Weather & Microclimate Disease Outbreak Risk
3. High-Resolution PDF Prescription / Health Card Generator (ReportLab)
4. YOLOv8 Edge Foliage Lesion & Spot Detector
"""

import os
import io
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image

import numpy as np
import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# ==============================================================================
# 1. MULTILINGUAL AI AGRONOMIST (GEMINI 1.5 + EXPERT FALLBACK)
# ==============================================================================
def ask_agronomist_ai(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    language: str = "English"
) -> Dict[str, Any]:
    """Conversational AI agronomist using Gemini 1.5 with fallback."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    ctx_str = json.dumps(context or {}, indent=2)

    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prompt = f"""You are 'KisanSeva Agronomist AI', an elite agricultural scientist and field expert for farmers in India.
Current Farm & Diagnostic Context:
{ctx_str}

Farmer Query: "{query}"
Target Language: {language}

Instructions:
1. Provide practical, step-by-step agronomic advice tailored to Indian farming conditions (organic bio-control, specific chemical dosages if severe, irrigation, fertilizer management).
2. Answer clearly and respectfully in the requested language ({language}).
3. If pathology is detected in the context, reference it directly.
4. Keep the response crisp, structured with bullet points.
"""
            response = model.generate_content(prompt)
            return {
                "source": "gemini-1.5-flash",
                "language": language,
                "reply": response.text,
                "has_api_key": True
            }
        except Exception as e:
            pass  # Fall back gracefully

    # Offline Rule-Based Fallback
    cond = (context or {}).get("condition", "General Crop Care")
    ph = (context or {}).get("ph", "Normal")
    risk = (context or {}).get("overall_risk_level", "Moderate")

    fallback_replies = {
        "English": f"**[KisanSeva Offline Knowledge Base]**\n\n• **Field Status**: Condition is currently identified as **{cond}** with a risk profile of **{risk}** (Soil pH: {ph}).\n• **Recommended Immediate Steps**:\n  - Maintain 15-20 cm plant spacing to improve canopy ventilation and reduce relative humidity in the foliage.\n  - Spray 5% cold-pressed Neem Oil or Bacillus subtilis bio-fungicide during early morning hours.\n  - Avoid high-nitrogen top dressing while plants are under active fungal or bacterial stress.\n• *Tip: Add your `GEMINI_API_KEY` in `.env` to unlock live conversational multi-turn chat in Hindi, Telugu, Tamil, and other regional languages.*",
        "Hindi": f"**[किसान सेवा ऑफलाइन ज्ञान केंद्र]**\n\n• **फसल की स्थिति**: वर्तमान में **{cond}** का प्रभाव देखा गया है (जोखिम स्तर: **{risk}**).\n• **तुरंत किए जाने वाले उपाय**:\n  - 5% नीम तेल (Neem Oil) का छिड़काव सुबह के समय करें।\n  - खेत में जलभराव न होने दें और अतिरिक्त यूरिया के प्रयोग से बचें।\n  - ट्राइकोडर्मा (Trichoderma) या स्यूडोमोनास (Pseudomonas) का प्रयोग रोग प्रतिरोधक क्षमता बढ़ाने के लिए करें।\n• *सूचना: लाइव बहुभाषी एआई चैट के लिए `.env` फाइल में `GEMINI_API_KEY` जोड़ें।*",
    }

    return {
        "source": "offline-expert-knowledge-base",
        "language": language,
        "reply": fallback_replies.get(language, fallback_replies["English"]),
        "has_api_key": False
    }


# ==============================================================================
# 2. HYPERLOCAL WEATHER & MICROCLIMATE OUTBREAK FORECASTER
# ==============================================================================
def get_hyperlocal_weather_forecast(
    latitude: float = 20.5937,
    longitude: float = 78.9629,
    city_name: Optional[str] = None
) -> Dict[str, Any]:
    """Fetch live weather and calculate disease outbreak risk index."""
    api_key = os.getenv("WEATHER_API_KEY", "").strip()

    # Try OpenWeather if key exists
    if api_key:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
            if city_name:
                url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                temp = data["main"]["temp"]
                humidity = data["main"]["humidity"]
                wind_speed = data["wind"]["speed"]
                rain = data.get("rain", {}).get("1h", 0.0)
                desc = data["weather"][0]["description"].title()

                # Fungal Spore Spread Risk Calculation
                disease_risk = "Low"
                if humidity > 80 and 18 <= temp <= 30:
                    disease_risk = "CRITICAL (High Fungal & Blight Sporulation Index)"
                elif humidity > 65 and 15 <= temp <= 32:
                    disease_risk = "MODERATE (Monitor lower canopy leaves)"

                return {
                    "source": "OpenWeather API (Live)",
                    "city": data.get("name", "Field Coordinates"),
                    "temperature_c": temp,
                    "humidity": humidity,
                    "wind_speed_kmh": round(wind_speed * 3.6, 1),
                    "rainfall_mm": rain,
                    "condition": desc,
                    "disease_outbreak_risk": disease_risk,
                    "spray_window": "Optimal for foliar spray" if wind_speed < 4.0 and rain == 0 else "Avoid spraying (wind or rain risk)"
                }
        except Exception:
            pass

    # Open-Meteo Free Fallback (No API key needed, global coverage)
    try:
        om_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        om_resp = requests.get(om_url, timeout=4)
        if om_resp.status_code == 200:
            c = om_resp.json().get("current", {})
            temp = c.get("temperature_2m", 26.5)
            hum = c.get("relative_humidity_2m", 68.0)
            rain = c.get("precipitation", 0.0)
            wind = c.get("wind_speed_10m", 8.5)

            disease_risk = "High" if (hum > 75 and 18 <= temp <= 29) else "Moderate"
            return {
                "source": "Open-Meteo Telemetry (Free / Live)",
                "city": f"Lat {latitude:.2f}, Lon {longitude:.2f}",
                "temperature_c": temp,
                "humidity": hum,
                "wind_speed_kmh": round(wind, 1),
                "rainfall_mm": rain,
                "condition": "Cloudy & Humid" if hum > 70 else "Clear / Sunny",
                "disease_outbreak_risk": f"{disease_risk} (Microclimate spore propagation indicator)",
                "spray_window": "Safe to spray" if wind < 15 and rain == 0 else "High drift or wash-off hazard"
            }
    except Exception:
        pass

    # Default offline synthetic simulation
    return {
        "source": "KisanSeva Microclimate Sensor Simulation",
        "city": "Agronomic Test Station",
        "temperature_c": 27.0,
        "humidity": 72.0,
        "wind_speed_kmh": 6.4,
        "rainfall_mm": 0.0,
        "condition": "Partly Overcast",
        "disease_outbreak_risk": "Moderate (Monitor Leaf Undersides for Mildew)",
        "spray_window": "Optimal Morning Application Window"
    }


# ==============================================================================
# 3. HIGH-RESOLUTION PDF PRESCRIPTION GENERATOR (REPORTLAB)
# ==============================================================================
def generate_agronomic_pdf_prescription(
    request_id: str,
    patient_crop: str,
    diagnosis_condition: str,
    confidence: float,
    pathogen: str,
    severity: str,
    health_score: int,
    organic_rx: List[str],
    chemical_rx: List[str],
    soil_type: Optional[str] = "Loamy Soil",
    field_readings: Optional[Dict[str, Any]] = None,
    heatmap_b64: Optional[str] = None
) -> bytes:
    """Generate high-quality farmer prescription PDF."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    elements = []
    styles = getSampleStyleSheet()

    # Custom Header
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor("#065f46"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor("#047857"),
        spaceAfter=14
    )
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor("#064e3b"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b")
    )

    elements.append(Paragraph("🌱 KisanSeva Certified Agronomic Health Card", title_style))
    elements.append(Paragraph(f"Official AI Field Diagnostic & Treatment Protocol • ID: {request_id}", subtitle_style))
    elements.append(Spacer(1, 8))

    # Overview Table
    summary_data = [
        ["Diagnosed Pathology:", diagnosis_condition, "Infection Severity:", severity],
        ["Model Confidence:", f"{confidence * 100:.1f}%", "Foliage Health Score:", f"{health_score} / 100"],
        ["Pathogen Class:", pathogen, "Soil Type Profile:", soil_type or "Evaluated Farm Topsoil"],
    ]
    t = Table(summary_data, colWidths=[130, 140, 130, 140])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#064e3b")),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#a7f3d0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 14))

    # Treatments
    elements.append(Paragraph("🌿 Integrated Pest Management (IPM) — Organic Regimen:", section_style))
    for i, item in enumerate(organic_rx, 1):
        elements.append(Paragraph(f"<b>{i}.</b> {item}", body_style))
        elements.append(Spacer(1, 3))

    elements.append(Spacer(1, 6))
    elements.append(Paragraph("🧪 Conventional Chemical Intervention (If progressive spread exceeds threshold):", section_style))
    for i, item in enumerate(chemical_rx, 1):
        elements.append(Paragraph(f"<b>{i}.</b> {item}", body_style))
        elements.append(Spacer(1, 3))

    elements.append(Spacer(1, 10))
    # Footer disclaimer
    disclaimer = Paragraph(
        "<i>Disclaimer: This digital diagnostic certificate is synthesized using Deep Convolutional Neural Networks and Integrated Pest Management (IPM) guidelines. Always consult a local Krishi Vigyan Kendra (KVK) officer for regional agrochemical validation.</i>",
        ParagraphStyle('Disc', parent=styles['Normal'], fontSize=7.5, textColor=colors.HexColor("#64748b"))
    )
    elements.append(disclaimer)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ==============================================================================
# 4. YOLOV8 EDGE LESION & FOLIAGE OBJECT DETECTOR
# ==============================================================================
_YOLO_MODEL = None

def run_yolov8_lesion_detection(image: Image.Image) -> Dict[str, Any]:
    """Run YOLOv8 inference to detect leaf regions, spots, and lesions."""
    global _YOLO_MODEL
    weights_path = BASE_DIR / "yolov8n.pt"

    if not weights_path.exists():
        return {
            "status": "not_found",
            "message": "yolov8n.pt not found in root directory",
            "detections": []
        }

    try:
        from ultralytics import YOLO
        if _YOLO_MODEL is None:
            _YOLO_MODEL = YOLO(str(weights_path))

        # Run inference
        results = _YOLO_MODEL(image, conf=0.25, verbose=False)
        result = results[0]

        # Convert plotted image to Base64 PNG
        plotted_img = result.plot()  # numpy array (BGR)
        # convert BGR to RGB
        plotted_rgb = plotted_img[..., ::-1]
        out_pil = Image.fromarray(plotted_rgb)

        buf = io.BytesIO()
        out_pil.save(buf, format="JPEG", quality=85)
        b64_str = base64.b64encode(buf.getvalue()).decode("ascii")

        boxes = []
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            name = result.names.get(cls_id, f"Class {cls_id}")
            conf = float(box.conf[0].item())
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            boxes.append({"label": name, "confidence": round(conf, 3), "bbox": xyxy})

        return {
            "status": "success",
            "total_objects": len(boxes),
            "detections": boxes,
            "annotated_image_b64": b64_str,
            "summary": f"Identified {len(boxes)} visual focal regions across the foliage canvas."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "detections": []
        }
