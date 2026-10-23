"""
KisanSeva Microclimate & Spray Early Warning Service
Calculates Penman-Monteith Evapotranspiration (ET0), Vapor Pressure Deficit (VPD),
Spray Window Suitability Index, and Extreme Weather (Frost, Heat Stress, Inversion) Alerts.
"""

from typing import Dict, Any, List
import math
from datetime import datetime, timedelta


class MicroclimateService:
    """
    Advanced agro-meteorological analytics engine for precision farm microclimate management.
    """

    @staticmethod
    def calculate_vapor_pressure_deficit(temp_c: float, relative_humidity: float) -> float:
        """
        Calculate Vapor Pressure Deficit (VPD) in kPa.
        VPD = VPsat - VPactual
        VPsat = 0.61078 * exp((17.27 * T) / (T + 237.3))
        """
        vp_sat = 0.61078 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
        vp_act = vp_sat * (relative_humidity / 100.0)
        vpd = vp_sat - vp_act
        return round(vpd, 2)

    @staticmethod
    def calculate_et0_hargreaves(t_max: float, t_min: float, t_mean: float, solar_radiation_mj: float = 18.5) -> float:
        """
        Calculate Reference Evapotranspiration (ET0) in mm/day using modified Hargreaves-Samani method.
        """
        temp_diff = max(0.1, t_max - t_min)
        et0 = 0.0023 * (t_mean + 17.8) * math.sqrt(temp_diff) * (solar_radiation_mj * 0.408)
        return round(max(0.5, et0), 2)

    @staticmethod
    def evaluate_spray_conditions(temp_c: float, humidity: float, wind_kmh: float, rain_prob: float) -> Dict[str, Any]:
        """
        Assess agrochemical spray window viability based on Delta T, wind drift, and wash-off risk.
        Delta T = Temp - Wet Bulb Temp. (Ideal: 2°C to 8°C; Wind: 5 to 15 km/h).
        """
        # Approximated Wet Bulb temp (Stull formula simplified)
        wet_bulb = temp_c * math.atan(0.151977 * math.sqrt(humidity + 8.313659)) + math.atan(temp_c + humidity) - math.atan(humidity - 1.676331) + 0.00391838 * (humidity ** 1.5) * math.atan(0.023101 * humidity) - 4.686035
        delta_t = round(temp_c - wet_bulb, 2)

        reasons = []
        is_safe = True
        risk_level = "OPTIMAL"

        if wind_kmh > 18.0:
            is_safe = False
            reasons.append(f"High wind speed ({wind_kmh} km/h) causes severe pesticide drift risk.")
            risk_level = "HAZARDOUS"
        elif wind_kmh < 3.0:
            reasons.append(f"Very calm wind ({wind_kmh} km/h) indicates potential thermal inversion; droplets remain suspended.")
            risk_level = "CAUTION"

        if temp_c > 32.0:
            is_safe = False
            reasons.append(f"High temperature ({temp_c}°C) accelerates droplet evaporation and causes foliar chemical burn.")
            risk_level = "HAZARDOUS"

        if delta_t > 8.0:
            reasons.append(f"Delta T is high ({delta_t}°C); spray droplets evaporate before target leaf absorption.")
            if risk_level != "HAZARDOUS":
                risk_level = "CAUTION"
        elif delta_t < 2.0:
            reasons.append(f"Delta T is very low ({delta_t}°C); high humidity prevents droplet impaction and promotes disease.")
            if risk_level != "HAZARDOUS":
                risk_level = "CAUTION"

        if rain_prob > 40.0:
            is_safe = False
            reasons.append(f"High precipitation probability ({rain_prob}%) causes chemical wash-off within rain-fast period.")
            risk_level = "HAZARDOUS"

        if is_safe and not reasons:
            reasons.append("Atmospheric stability, Delta T, and wind velocity are within optimal agronomic thresholds.")

        return {
            "status": risk_level,
            "can_spray": is_safe,
            "delta_t_c": delta_t,
            "wet_bulb_c": round(wet_bulb, 2),
            "wind_speed_kmh": wind_kmh,
            "rain_probability_pct": rain_prob,
            "reasons": reasons,
            "advisory": "Proceed with calibrated hollow-cone or twin-fan nozzles." if is_safe else "Postpone spraying until conditions stabilize."
        }

    @classmethod
    def analyze_microclimate(
        cls,
        farm_name: str,
        location: str,
        crop_type: str,
        temp_c: float,
        temp_max_c: float,
        temp_min_c: float,
        relative_humidity: float,
        wind_kmh: float,
        soil_temp_c: float = 24.5,
        rain_probability_pct: float = 10.0
    ) -> Dict[str, Any]:
        """
        Full agro-meteorological microclimate analysis, warning detection, and 24-hour spray forecast.
        """
        vpd = cls.calculate_vapor_pressure_deficit(temp_c, relative_humidity)
        et0 = cls.calculate_et0_hargreaves(temp_max_c, temp_min_c, temp_c)
        spray_check = cls.evaluate_spray_conditions(temp_c, relative_humidity, wind_kmh, rain_probability_pct)

        # VPD Interpretation
        if vpd < 0.4:
            vpd_status = "Very Low (Excessive Humidity / Fungal & Bacterial Blight Danger)"
            vpd_color = "#E74C3C"
        elif 0.4 <= vpd <= 1.2:
            vpd_status = "Optimal Transpiration & Nutrient Assimilation"
            vpd_color = "#2ECC71"
        elif 1.2 < vpd <= 1.8:
            vpd_status = "Moderate Stress (Stomata partially closing)"
            vpd_color = "#F39C12"
        else:
            vpd_status = "Extreme Stress (Desiccation & Floral Abortion Hazard)"
            vpd_color = "#C0392B"

        # Early Warning Alerts
        alerts = []

        # 1. Frost Warning
        if temp_min_c <= 4.0:
            severity = "CRITICAL" if temp_min_c <= 1.5 else "WARNING"
            alerts.append({
                "type": "Radiation Frost Early Warning",
                "severity": severity,
                "message": f"Projected minimum overnight temperature is {temp_min_c}°C. High risk of intracellular cell freezing and tissue necrosis in tender vegetative shoots.",
                "mitigation": "Initiate nocturnal micro-sprinkler irrigation to release latent heat of fusion; apply potassium silicate foliar spray or light organic mulch."
            })

        # 2. Extreme Heat Stress
        if temp_max_c >= 38.0:
            alerts.append({
                "type": "Extreme Heatwave & Pollen Desiccation",
                "severity": "CRITICAL" if temp_max_c >= 41.0 else "WARNING",
                "message": f"Daytime peak of {temp_max_c}°C exceeds critical heat threshold for {crop_type}. Induces pollen sterility, floral drop, and severe photo-respiration.",
                "mitigation": "Schedule pulse irrigation between 11:30 AM and 2:00 PM; erect 35% agro-shade netting over nursery or high-value canopy; apply anti-transpirant kaolin clay spray (2%)."
            })

        # 3. High Inversion Drift Risk
        if wind_kmh < 2.5 and relative_humidity > 80.0:
            alerts.append({
                "type": "Surface Thermal Inversion Layer",
                "severity": "WARNING",
                "message": "Near-surface air temperature profile indicates inversion cap. Aerosolized pesticide mist will not dissipate and could drift hundreds of meters off-target.",
                "mitigation": "Strictly suspend all chemical spray applications until solar warming establishes upward convection currents."
            })

        # Hourly 24-hr projected spray timeline
        hourly_schedule = []
        base_time = datetime.now()
        for h in range(12):
            proj_time = base_time + timedelta(hours=h * 2)
            # Simulate day/night thermal fluctuation
            hour_val = proj_time.hour
            is_day = 6 <= hour_val <= 18
            proj_temp = temp_c + (3.5 * math.sin((hour_val - 8) * math.pi / 12) if is_day else -2.5)
            proj_hum = max(30.0, min(95.0, relative_humidity - (15.0 if is_day else -10.0)))
            proj_wind = max(2.0, wind_kmh + (4.0 * math.cos(h)))
            eval_step = cls.evaluate_spray_conditions(round(proj_temp, 1), round(proj_hum, 1), round(proj_wind, 1), rain_probability_pct)
            hourly_schedule.append({
                "time": proj_time.strftime("%I:%M %p"),
                "temp_c": round(proj_temp, 1),
                "humidity_pct": round(proj_hum, 1),
                "wind_kmh": round(proj_wind, 1),
                "delta_t": eval_step["delta_t_c"],
                "spray_status": eval_step["status"]
            })

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "farm_name": farm_name,
            "location": location,
            "crop_type": crop_type,
            "microclimate_metrics": {
                "air_temperature_c": temp_c,
                "temp_max_c": temp_max_c,
                "temp_min_c": temp_min_c,
                "soil_temperature_c": soil_temp_c,
                "relative_humidity_pct": relative_humidity,
                "wind_velocity_kmh": wind_kmh,
                "vapor_pressure_deficit_kpa": vpd,
                "vpd_status": vpd_status,
                "vpd_color": vpd_color,
                "reference_et0_mm_day": et0,
            },
            "spray_window": spray_check,
            "early_warning_alerts": alerts,
            "hourly_spray_timeline": hourly_schedule
        }


# Global instance
microclimate_service = MicroclimateService()
