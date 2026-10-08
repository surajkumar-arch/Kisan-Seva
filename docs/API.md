# EDAPHNET API Specification Document

The EDAPHNET API is a production-ready RESTful gateway built using FastAPI.

**Base URL**: `http://localhost:8000`  
**Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)  
**ReDoc**: `http://localhost:8000/redoc`

---

## Headers & Observability

All API requests support the following headers:
- `X-Request-ID`: Client correlation UUID (if omitted, generated server-side).
- `X-Response-Time-Ms`: Latency in milliseconds returned in every response header.

---

## 1. System Health & Probes

### `GET /health`
Liveness probe for Kubernetes and Docker container monitoring.
- **Response**: `200 OK`
```json
{
  "status": "healthy",
  "service": "edaphnet-api",
  "timestamp": 1728400000.12
}
```

### `GET /ready`
Readiness probe verifying that registries and ML weight buffers are initialized.
- **Response**: `200 OK`
```json
{
  "status": "ready",
  "models_loaded": 5,
  "environment": "production"
}
```

### `GET /version`
Active deployment metadata.
- **Response**: `200 OK`
```json
{
  "version": "1.0.0",
  "api_name": "EDAPHNET",
  "framework": "PyTorch / FastAPI",
  "environment": "production"
}
```

---

## 2. Multi-Modal Inference

### `POST /api/v1/analyze`
Executes end-to-end multi-modal vision and tabular inference.

**Form Data / Multipart**:
- `soil_image`: (Binary JPEG/PNG) Soil surface photograph.
- `leaf_image`: (Binary JPEG/PNG) Foliage photograph.
- `n`: (Float) Nitrogen level in soil.
- `p`: (Float) Phosphorus level in soil.
- `k`: (Float) Potassium level in soil.
- `ph`: (Float) Soil pH value.
- `temperature`: (Float) Ambient temperature in °C.
- `humidity`: (Float) Relative humidity percentage.
- `rainfall`: (Float) Precipitation in mm.
- `include_explainability`: (Bool, default: true) Compute Grad-CAM and SHAP.
- `include_segmentation`: (Bool, default: true) Compute U-Net lesion mask.
- `include_localization`: (Bool, default: true) Compute YOLO bounding boxes.

**Response**: `200 OK` (Typed `InferenceResponse`)
```json
{
  "request_id": "c1f720aa-62b1-4d1a-824b-876a44bfec09",
  "timestamp": "2026-10-08T20:30:00Z",
  "soil": {
    "class": "Red Soil",
    "confidence": 0.87,
    "raw_confidence": 0.91,
    "calibrated": true,
    "probabilities": {
      "Red Soil": 0.87,
      "Alluvial Soil": 0.10
    },
    "model_version": "1.3.0"
  },
  "disease": {
    "class": "Early Blight",
    "confidence": 0.92,
    "raw_confidence": 0.94,
    "pathogen_type": "Fungal",
    "severity_level": "Moderate",
    "probabilities": {
      "Early Blight": 0.92
    },
    "model_version": "1.3.0"
  },
  "recommendations": [
    {
      "crop_name": "cotton",
      "rank": 1,
      "suitability_score": 92.5,
      "compatibility_grade": "A+",
      "projected_yield": "2.8 - 3.4 Tons/Hectare",
      "water_requirement": "500 - 700 mm",
      "growing_duration_days": "150 - 180 Days"
    }
  ],
  "reliability": {
    "level": "HIGH",
    "decision": "PROCEED",
    "ood_detected": false,
    "mahalanobis_distance": 3.82,
    "threshold_distance": 16.5,
    "message": "High reliability: all multi-modal checks, calibration and feature distributions verified."
  },
  "missing_measurements": [],
  "processing_time_ms": 142.5
}
```

---

## 3. Dedicated ML Component Routes

- `POST /api/v1/soil/predict`: Classify soil photograph.
- `POST /api/v1/disease/predict`: Classify plant foliage disease.
- `POST /api/v1/disease/localize`: Detect lesion bounding boxes via YOLO.
- `POST /api/v1/disease/segment`: Calculate infected area percentage via U-Net.
- `POST /api/v1/recommendations`: Calculate calibrated crop suitability rankings.
- `GET /api/v1/models`: List all certified models from registry with version and checksum.
- `GET /api/v1/requests/{request_id}`: Retrieve audit telemetry by request UUID.
