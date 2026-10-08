"""Live Weather & Environmental Air Quality Integration Service via WAQI API."""

import os
import logging
from typing import Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)

DEFAULT_WAQI_TOKEN = os.getenv("WAQI_API_TOKEN", "fed9145439cd8ab6221d3477f02875101d72efe3")
WAQI_BASE_URL = "https://api.waqi.info/feed"


def categorize_aqi(aqi: int) -> Dict[str, str]:
    """Categorize AQI and provide agronomic impact analysis for crops."""
    if aqi <= 50:
        return {
            "category": "Good",
            "color": "#10b981",
            "agronomic_risk": "Optimal",
            "foliar_impact": "Air quality is pristine. Optimal conditions for unobstructed stomatal gas exchange and light interception."
        }
    elif aqi <= 100:
        return {
            "category": "Moderate",
            "color": "#f59e0b",
            "agronomic_risk": "Low",
            "foliar_impact": "Acceptable air quality. Minor particulate deposition possible on older foliage cuticles."
        }
    elif aqi <= 150:
        return {
            "category": "Unhealthy for Sensitive Groups",
            "color": "#f97316",
            "agronomic_risk": "Moderate",
            "foliar_impact": "Elevated particulate matter (PM2.5) may cause mild surface deposition, reducing photosynthetic photon flux density (PPFD)."
        }
    elif aqi <= 200:
        return {
            "category": "Unhealthy",
            "color": "#ef4444",
            "agronomic_risk": "High",
            "foliar_impact": "Substantial particulate layer on leaf lamina. High PM2.5 and Ozone can induce stomatal closure and reduce transpiration efficiency."
        }
    elif aqi <= 300:
        return {
            "category": "Very Unhealthy",
            "color": "#8b5cf6",
            "agronomic_risk": "Severe",
            "foliar_impact": "Heavy atmospheric smog. Ozone oxidation stress risk and particulate clogging of stomatal apertures. Foliar wash or misting recommended."
        }
    else:
        return {
            "category": "Hazardous",
            "color": "#7f1d1d",
            "agronomic_risk": "Extreme",
            "foliar_impact": "Critical atmospheric pollution. Severe phytotoxic oxidative stress and solar radiation attenuation. Immediate foliar protective action advised."
        }


async def fetch_live_environmental_data(location: str = "here", token: Optional[str] = None) -> Dict[str, Any]:
    """Fetch live meteorological and air quality metrics from WAQI API.

    Args:
        location: 'here', city name (e.g. 'delhi', 'mumbai'), or 'geo:lat;lng'
        token: WAQI API authentication token
    """
    api_token = token or DEFAULT_WAQI_TOKEN
    clean_loc = location.strip() if location else "here"
    url = f"{WAQI_BASE_URL}/{clean_loc}/?token={api_token}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers={"User-Agent": "KisanSeva-Agriculture/8.0"})

        if response.status_code != 200:
            return {
                "status": "error",
                "message": f"WAQI API returned status HTTP {response.status_code}",
                "data": None
            }

        res_json = response.json()
        if res_json.get("status") != "ok" or "data" not in res_json:
            return {
                "status": "error",
                "message": res_json.get("data", "Failed to retrieve location from WAQI"),
                "data": None
            }

        raw = res_json["data"]
        iaqi = raw.get("iaqi", {})

        # Extract temperature, humidity, and atmospheric metrics
        temp_val = iaqi.get("t", {}).get("v")
        hum_val = iaqi.get("h", {}).get("v")
        pressure_val = iaqi.get("p", {}).get("v")
        wind_val = iaqi.get("w", {}).get("v")
        dew_val = iaqi.get("dew", {}).get("v")

        aqi_val = raw.get("aqi", 0)
        aqi_analysis = categorize_aqi(aqi_val)

        # Extract primary pollutants
        pollutants = {
            "pm25": iaqi.get("pm25", {}).get("v"),
            "pm10": iaqi.get("pm10", {}).get("v"),
            "no2": iaqi.get("no2", {}).get("v"),
            "so2": iaqi.get("so2", {}).get("v"),
            "o3": iaqi.get("o3", {}).get("v"),
            "co": iaqi.get("co", {}).get("v"),
        }

        city_info = raw.get("city", {})
        time_info = raw.get("time", {})

        return {
            "status": "success",
            "source": "World Air Quality Index (WAQI)",
            "station": city_info.get("name", "Current Location"),
            "geo": city_info.get("geo", []),
            "station_url": city_info.get("url", ""),
            "observation_time": time_info.get("s", ""),
            "timezone": time_info.get("tz", ""),
            "meteorology": {
                "temperature_celsius": round(temp_val, 1) if temp_val is not None else None,
                "humidity_percent": round(hum_val, 1) if hum_val is not None else None,
                "pressure_hpa": round(pressure_val, 1) if pressure_val is not None else None,
                "wind_speed_mps": round(wind_val, 1) if wind_val is not None else None,
                "dew_point_celsius": round(dew_val, 1) if dew_val is not None else None,
            },
            "air_quality": {
                "aqi": aqi_val,
                "category": aqi_analysis["category"],
                "color_code": aqi_analysis["color"],
                "agronomic_risk": aqi_analysis["agronomic_risk"],
                "foliar_impact_assessment": aqi_analysis["foliar_impact"],
                "dominant_pollutant": raw.get("dominentpol", "pm25"),
                "pollutants": pollutants,
            }
        }

    except Exception as exc:
        logger.error(f"Error querying WAQI API: {str(exc)}")
        return {
            "status": "error",
            "message": f"Exception connecting to WAQI API: {str(exc)}",
            "data": None
        }
