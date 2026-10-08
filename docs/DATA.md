# EDAPHNET Data Management & Lineage Specification

## 1. Zero Synthetic Data Policy

Per enterprise engineering standards and anti-hallucination rules:
- **No synthetic dummy dataset folders are committed** to the codebase repository.
- The `data/` directory structure maintains strict isolation with `.gitkeep` placeholders.
- Real agricultural field data is ingested through authenticated training scripts and validated against data contracts.

---

## 2. Supported Datasets & Schema Provenance

### 2.1 Soil Surface Image Dataset
- **Taxonomy**: 6 major agricultural soil types (`Alluvial`, `Black`, `Clay`, `Red`, `Sandy`, `Loamy`).
- **Input Specifications**: RGB JPEG/PNG, recommended minimum resolution $224 \times 224$ pixels.
- **Normalization**: ImageNet standard RGB channel centering ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).

### 2.2 Plant Foliar Pathology Dataset
- **Taxonomy**: Host foliage and common pathologies (`Healthy Foliage`, `Early Blight`, `Late Blight`, `Powdery Mildew`, `Leaf Rust`, `Bacterial Spot`, `Yellow Leaf Curl`).
- **Annotations**:
  - Classification: Multi-class label.
  - Localization: Bounding boxes `(x_min, y_min, x_max, y_max)`.
  - Segmentation: Binary necrosis pixel masks `(0: healthy/background, 1: lesion)`.

### 2.3 Agronomic Tabular Features
| Feature | Range | Unit | Description |
|:---|:---|:---|:---|
| `N` | $0 - 400$ | mg/kg or kg/ha | Available Nitrogen |
| `P` | $0 - 200$ | mg/kg or kg/ha | Available Phosphorus |
| `K` | $0 - 500$ | mg/kg or kg/ha | Available Potassium |
| `pH` | $0.0 - 14.0$ | pH unit | Potentiometric soil reaction |
| `Temperature` | $-10 - 60$ | °C | Mean growing temperature |
| `Humidity` | $0 - 100$ | % | Relative ambient humidity |
| `Rainfall` | $0 - 1000$ | mm | Annual / Seasonal precipitation |

---

## 3. Missing Data Strategy

When agricultural parameters are absent from field reports:
1. The measurement is explicitly recorded as `MISSING`.
2. A scientifically conservative regional median is imputed for feature fusion.
3. The measurement status is surfaced in the response contract: `status: MISSING`, `estimated_value: <value>`, `is_critical: true`.
4. The system penalty adjusts the overall reliability level to `MODERATE` or `LOW` and provides an actionable lab test prescription.
