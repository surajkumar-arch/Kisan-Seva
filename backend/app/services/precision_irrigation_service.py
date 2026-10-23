"""
KisanSeva Precision Smart Irrigation & Solar Pump Digital Twin Service
Calculates soil moisture deficit, crop evapotranspiration (ETc), and optimal
solar PV pump operation windows with automated IoT MQTT telemetry schedules.
"""

from typing import Dict, Any, List
import math
from datetime import datetime, timedelta


class PrecisionIrrigationService:
    """
    IoT Precision Water Engineering & Solar Energy Optimization Engine.
    """

    # Available Water Capacity (mm of water per meter of soil depth)
    SOIL_HYDRAULIC_PROPERTIES = {
        "sandy_loam": {"name": "Sandy Loam", "field_capacity": 18.0, "wilting_point": 8.0, "awc_mm_m": 100.0, "max_inf_rate_mm_h": 25.0},
        "loam": {"name": "Alluvial Loam", "field_capacity": 27.0, "wilting_point": 12.0, "awc_mm_m": 150.0, "max_inf_rate_mm_h": 15.0},
        "clay_loam": {"name": "Clay Loam", "field_capacity": 32.0, "wilting_point": 15.0, "awc_mm_m": 170.0, "max_inf_rate_mm_h": 10.0},
        "black_cotton": {"name": "Deep Black Cotton (Vertisol)", "field_capacity": 38.0, "wilting_point": 19.0, "awc_mm_m": 190.0, "max_inf_rate_mm_h": 6.0},
        "red_sandy": {"name": "Red Sandy Soil (Alfisol)", "field_capacity": 20.0, "wilting_point": 9.0, "awc_mm_m": 110.0, "max_inf_rate_mm_h": 20.0}
    }

    # Crop Kc coefficients across stages
    CROP_KC_TABLE = {
        "wheat": {"initial": 0.35, "vegetative": 0.75, "flowering": 1.15, "grain_filling": 1.05, "maturation": 0.40, "root_depth_m": 0.9},
        "rice": {"initial": 1.05, "vegetative": 1.10, "flowering": 1.25, "grain_filling": 1.15, "maturation": 0.70, "root_depth_m": 0.6},
        "cotton": {"initial": 0.35, "vegetative": 0.70, "flowering": 1.20, "grain_filling": 0.95, "maturation": 0.50, "root_depth_m": 1.2},
        "maize": {"initial": 0.40, "vegetative": 0.80, "flowering": 1.20, "grain_filling": 1.05, "maturation": 0.55, "root_depth_m": 1.0},
        "sugarcane": {"initial": 0.40, "vegetative": 0.90, "flowering": 1.25, "grain_filling": 1.25, "maturation": 0.75, "root_depth_m": 1.5},
        "tomato": {"initial": 0.45, "vegetative": 0.75, "flowering": 1.15, "grain_filling": 1.05, "maturation": 0.80, "root_depth_m": 0.7},
        "potato": {"initial": 0.50, "vegetative": 0.75, "flowering": 1.15, "grain_filling": 0.90, "maturation": 0.65, "root_depth_m": 0.5}
    }

    @classmethod
    def calculate_irrigation_schedule(
        cls,
        farm_id: str,
        crop_name: str,
        growth_stage: str,
        soil_type: str,
        farm_area_acres: float,
        current_soil_moisture_pct: float,
        daily_et0_mm: float = 4.8,
        pump_rating_hp: float = 5.0,
        irrigation_method: str = "drip"
    ) -> Dict[str, Any]:
        """
        Calculate precise water deficit, duration, and pump control schedule.
        """
        crop_key = crop_name.lower().strip()
        crop_data = cls.CROP_KC_TABLE.get(crop_key, cls.CROP_KC_TABLE["wheat"])
        soil_key = soil_type.lower().strip()
        soil_data = cls.SOIL_HYDRAULIC_PROPERTIES.get(soil_key, cls.SOIL_HYDRAULIC_PROPERTIES["loam"])

        kc = crop_data.get(growth_stage.lower(), 1.0)
        root_depth = crop_data["root_depth_m"]

        # Crop Evapotranspiration ETc (mm/day)
        etc_mm = round(daily_et0_mm * kc, 2)

        # Total Available Water (TAW) in mm
        taw_mm = soil_data["awc_mm_m"] * root_depth

        # Readily Available Water (RAW) with MAD = 0.5
        raw_mm = round(taw_mm * 0.5, 1)

        # Current moisture deficit (mm)
        fc = soil_data["field_capacity"]
        pwp = soil_data["wilting_point"]
        available_vol = max(0.0, current_soil_moisture_pct - pwp)
        total_vol = max(1.0, fc - pwp)
        moisture_fraction = min(1.0, available_vol / total_vol)
        current_depletion_mm = round(taw_mm * (1.0 - moisture_fraction), 1)

        # Determine Irrigation Requirement (mm)
        irrigation_needed_mm = max(0.0, current_depletion_mm)
        irrigation_urgency = "URGENT" if current_depletion_mm >= raw_mm else ("SCHEDULED" if current_depletion_mm >= 0.5 * raw_mm else "ADEQUATE")

        # Irrigation efficiency
        efficiency = 0.90 if irrigation_method.lower() == "drip" else (0.75 if irrigation_method.lower() == "sprinkler" else 0.55)

        # Gross water volume needed:
        # 1 mm of water over 1 acre (4046.86 m2) = 4.047 m3 = 4,047 liters
        area_sqm = farm_area_acres * 4046.86
        net_liters = irrigation_needed_mm * area_sqm
        gross_liters = round(net_liters / efficiency, 0)
        gross_m3 = round(gross_liters / 1000.0, 2)

        # Pump delivery capacity: ~1 HP provides ~120 liters/min at standard 25m head
        liters_per_minute = pump_rating_hp * 125.0
        required_runtime_minutes = round(gross_liters / max(10.0, liters_per_minute), 0)

        # Solar PV window planning (10:00 AM to 03:30 PM peak irradiance)
        start_hour = 10
        start_min = 15
        start_time_obj = datetime.now().replace(hour=start_hour, minute=start_min, second=0, microsecond=0)
        end_time_obj = start_time_obj + timedelta(minutes=required_runtime_minutes)

        # MQTT Telemetry Payload
        mqtt_payload = {
            "topic": f"kisanseva/pumps/{farm_id}/control",
            "device_id": f"PUMP-SOLAR-{farm_id.upper()}",
            "command": "IRRIGATE_CYCLE",
            "parameters": {
                "runtime_minutes": int(required_runtime_minutes),
                "target_volume_liters": int(gross_liters),
                "pump_flow_lpm": liters_per_minute,
                "pulse_mode": True if irrigation_method.lower() == "drip" else False
            },
            "solar_sync": True,
            "scheduled_start": start_time_obj.strftime("%H:%M:%S"),
            "scheduled_stop": end_time_obj.strftime("%H:%M:%S")
        }

        return {
            "success": True,
            "farm_id": farm_id,
            "crop": crop_name.title(),
            "growth_stage": growth_stage.title(),
            "kc_coefficient": kc,
            "soil_type": soil_data["name"],
            "etc_crop_water_loss_mm_day": etc_mm,
            "total_available_water_mm": taw_mm,
            "readily_available_water_mm": raw_mm,
            "current_depletion_mm": current_depletion_mm,
            "irrigation_urgency": irrigation_urgency,
            "water_budget": {
                "gross_water_needed_liters": int(gross_liters),
                "gross_water_needed_m3": gross_m3,
                "application_efficiency_pct": int(efficiency * 100),
                "liters_saved_vs_flood": int(gross_liters * (0.85 - 0.55) / 0.55) if irrigation_method.lower() == "drip" else 0
            },
            "solar_pump_dispatch": {
                "pump_rating_hp": pump_rating_hp,
                "pump_flow_rate_lpm": liters_per_minute,
                "runtime_minutes": int(required_runtime_minutes),
                "runtime_hours": round(required_runtime_minutes / 60.0, 1),
                "solar_window": f"{start_time_obj.strftime('%I:%M %p')} - {end_time_obj.strftime('%I:%M %p')}",
                "solar_power_source": "100% PM-KUSUM Zero-Emission Solar PV (0 diesel consumed)"
            },
            "iot_mqtt_control_packet": mqtt_payload
        }


# Global instance
precision_irrigation_service = PrecisionIrrigationService()
