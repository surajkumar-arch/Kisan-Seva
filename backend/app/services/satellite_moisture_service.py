"""Satellite Hydrological & Root-Zone Drought Risk Engine.

Computes synthetic NDWI (Normalized Difference Water Index), Root Zone Soil Moisture
(RZSM), Crop Evapotranspiration Deficit (ETc vs ET0), and dynamic irrigation scheduling
using physics-based agronomic hydrology.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Crop water coefficients (Kc values across growth stages)
CROP_KC_TABLE = {
    "Wheat": {"initial": 0.40, "mid": 1.15, "late": 0.35, "root_depth_m": 1.2, "depletion_fraction": 0.55},
    "Rice": {"initial": 1.10, "mid": 1.25, "late": 1.00, "root_depth_m": 0.6, "depletion_fraction": 0.20},
    "Cotton": {"initial": 0.45, "mid": 1.20, "late": 0.65, "root_depth_m": 1.4, "depletion_fraction": 0.65},
    "Maize": {"initial": 0.40, "mid": 1.20, "late": 0.50, "root_depth_m": 1.0, "depletion_fraction": 0.55},
    "Tomato": {"initial": 0.60, "mid": 1.15, "late": 0.80, "root_depth_m": 0.8, "depletion_fraction": 0.40},
    "Potato": {"initial": 0.50, "mid": 1.15, "late": 0.75, "root_depth_m": 0.6, "depletion_fraction": 0.35},
    "Sugarcane": {"initial": 0.40, "mid": 1.25, "late": 0.75, "root_depth_m": 1.6, "depletion_fraction": 0.65},
    "Mustard": {"initial": 0.35, "mid": 1.10, "late": 0.45, "root_depth_m": 1.0, "depletion_fraction": 0.60},
}

# Soil Available Water Capacity (AWC in mm/meter)
SOIL_AWC = {
    "Alluvial Soil": 160.0,
    "Black Soil": 200.0,
    "Clayey Soil": 180.0,
    "Red Soil": 120.0,
    "Sandy Soil": 70.0,
    "Loamy Soil": 150.0,
}


def calculate_penman_monteith_et0(temperature_c: float, humidity_pct: float, wind_kmh: float = 8.0, solar_rad_mj: float = 21.0) -> float:
    """Estimates Reference Evapotranspiration (ET0 in mm/day) via FAO-56 Penman-Monteith approximation."""
    t = temperature_c
    rh = max(10.0, min(99.0, humidity_pct))
    u2 = wind_kmh * (1000.0 / 3600.0)  # m/s
    
    # Saturation vapor pressure (es in kPa)
    es = 0.6108 * math.exp((17.27 * t) / (t + 237.3))
    ea = es * (rh / 100.0)
    vpd = max(0.05, es - ea)
    
    # Slope of saturation vapor pressure curve (Delta)
    delta = (4098.0 * es) / ((t + 237.3) ** 2)
    gamma = 0.067  # Psychrometric constant (kPa/C)
    
    rn = solar_rad_mj * 0.408 * 0.6  # Net radiation approx
    et0 = (0.408 * delta * rn + gamma * (900.0 / (t + 273.0)) * u2 * vpd) / (delta + gamma * (1.0 + 0.34 * u2))
    return round(max(1.5, min(12.0, et0)), 2)


def generate_synthetic_ndwi_heatmap(soil_type: str, moisture_pct: float) -> str:
    """Generates a high-resolution base64 encoded synthetic NDWI raster map."""
    grid_size = 48
    np.random.seed(42)
    
    # Base moisture pattern with natural spatial topography gradient
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Spatial terrain drainage simulation
    terrain_drainage = np.sin(xx * 1.5) * np.cos(yy * 1.5) * 0.15
    noise = np.random.normal(0, 0.05, (grid_size, grid_size))
    
    # Normalized moisture baseline (-0.3 to +0.6)
    base_val = (moisture_pct / 100.0) * 0.7 - 0.2
    ndwi_matrix = np.clip(base_val + terrain_drainage + noise, -0.4, 0.65)
    
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=100)
    cmap = matplotlib.colormaps["Blues"]
    im = ax.imshow(ndwi_matrix, cmap=cmap, vmin=-0.3, vmax=0.6, extent=[0, 100, 0, 100])
    ax.set_title(f"Synthetic SAR/NDWI Moisture Ingress ({soil_type})", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Field Longitude Transect (m)", fontsize=9)
    ax.set_ylabel("Field Latitude Transect (m)", fontsize=9)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NDWI Hydrological Index", fontsize=9)
    
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def evaluate_field_hydrology(
    crop: str = "Wheat",
    soil_type: str = "Alluvial Soil",
    temperature_c: float = 30.0,
    humidity_pct: float = 55.0,
    rainfall_14d_mm: float = 12.0,
    growth_stage: str = "mid",
    land_area_acres: float = 2.5,
) -> Dict[str, Any]:
    """Computes end-to-end hydrological risk, drought index, and irrigation prescription."""
    crop_info = CROP_KC_TABLE.get(crop, CROP_KC_TABLE["Wheat"])
    awc = SOIL_AWC.get(soil_type, 150.0)
    
    kc = crop_info.get(growth_stage, crop_info["mid"])
    root_depth = crop_info["root_depth_m"]
    p_depletion = crop_info["depletion_fraction"]
    
    # Total Available Water (TAW in mm)
    taw = awc * root_depth
    # Readily Available Water (RAW in mm)
    raw = taw * p_depletion
    
    # Evapotranspiration
    et0 = calculate_penman_monteith_et0(temperature_c, humidity_pct)
    etc = round(et0 * kc, 2)
    
    # 14-day cumulative water balance
    water_demand_14d = etc * 14.0
    effective_rain_14d = rainfall_14d_mm * 0.75  # 75% infiltration efficiency
    net_deficit_mm = max(0.0, water_demand_14d - effective_rain_14d)
    
    # Root Zone Soil Moisture (RZSM %)
    moisture_ratio = max(0.10, min(1.0, 1.0 - (net_deficit_mm / max(1.0, taw))))
    rzsm_pct = round(moisture_ratio * 100.0, 1)
    
    # Drought Risk Index (0 - 100)
    drought_risk_score = round(max(5.0, min(95.0, (1.0 - moisture_ratio) * 100.0)), 1)
    
    if drought_risk_score >= 65.0:
        drought_status = "CRITICAL_DEFICIT"
        drought_badge = "🚨 Severe Water Stress"
        action_urgency = "Immediate Irrigation Required within 24 Hours"
    elif drought_risk_score >= 40.0:
        drought_status = "MODERATE_DEFICIT"
        drought_badge = "⚠️ Approaching Wilting Point"
        action_urgency = "Schedule Drip Irrigation within 3 Days"
    else:
        drought_status = "OPTIMAL_HYDRATION"
        drought_badge = "✅ Adequate Moisture"
        action_urgency = "Maintain Normal Pulse Cycle"

    # Irrigation prescription (m3 per acre and liters total)
    # 1 mm over 1 acre = 4,046.86 liters = 4.047 m3
    water_needed_mm = round(min(net_deficit_mm, raw), 1)
    liters_per_acre = round(water_needed_mm * 4046.86)
    total_liters = round(liters_per_acre * land_area_acres)
    
    drip_hours_required = round(total_liters / max(100.0, (land_area_acres * 3500.0)), 1)  # approx 3.5k L/hr/acre drip capacity

    heatmap_base64 = generate_synthetic_ndwi_heatmap(soil_type, rzsm_pct)

    return {
        "status": "SUCCESS",
        "crop": crop,
        "soil_type": soil_type,
        "growth_stage": growth_stage,
        "land_area_acres": land_area_acres,
        "hydrology_metrics": {
            "reference_et0_mm_day": et0,
            "crop_evapotranspiration_etc_mm_day": etc,
            "crop_coefficient_kc": kc,
            "total_available_water_taw_mm": round(taw, 1),
            "readily_available_water_raw_mm": round(raw, 1),
            "water_deficit_14d_mm": round(net_deficit_mm, 1),
            "root_zone_soil_moisture_pct": rzsm_pct,
            "drought_risk_index_score": drought_risk_score,
            "drought_status": drought_status,
            "drought_badge": drought_badge,
            "action_urgency": action_urgency,
        },
        "irrigation_prescription": {
            "recommended_water_depth_mm": water_needed_mm,
            "water_volume_liters_per_acre": liters_per_acre,
            "total_field_volume_liters": total_liters,
            "recommended_drip_duration_hours": drip_hours_required,
            "recommended_timing": "Early Morning (05:00 - 08:00 AM) or Late Evening to minimize evaporative loss",
        },
        "heatmap_png_base64": heatmap_base64,
    }
