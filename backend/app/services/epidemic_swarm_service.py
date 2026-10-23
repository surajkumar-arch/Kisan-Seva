"""
KisanSeva Bio-Surveillance & Epidemic Swarm Early Warning Service
Monitors regional pathogen cluster density, computes downwind aerodynamic
spore/pest dispersal trajectories, and calculates epidemiological R0 reproduction rates.
"""

from typing import Dict, Any, List
import math
from datetime import datetime, timedelta


class EpidemicSwarmService:
    """
    Geospatial Epidemiology and Early Bio-Surveillance Defense Swarm Engine.
    """

    PATHOGEN_PROPERTIES = {
        "yellow_rust": {
            "name": "Stripe / Yellow Rust (Puccinia striiformis)",
            "crop": "Wheat",
            "type": "Airborne Fungal Spore",
            "base_r0": 2.8,
            "optimal_temp_c": (10.0, 18.0),
            "wind_dispersal_factor": 1.4,
            "latent_period_days": 10,
            "barrier_advisory": "Erect 1.5m high windbreaks; prophylactic spray of Tebuconazole 25.9% EC (1ml/L) or Propiconazole 25% EC (1ml/L) on all downwind field edges."
        },
        "fall_armyworm": {
            "name": "Fall Armyworm (Spodoptera frugiperda)",
            "crop": "Maize",
            "type": "Nocturnal Migratory Moth / Pest",
            "base_r0": 3.4,
            "optimal_temp_c": (22.0, 32.0),
            "wind_dispersal_factor": 2.1,
            "latent_period_days": 4,
            "barrier_advisory": "Install 15 pheromone traps/ha with Spodolure septa; intercrop with Desmodium; release egg parasitoid Trichogramma pretiosum @ 50,000/ha."
        },
        "pink_bollworm": {
            "name": "Pink Bollworm (Pectinophora gossypiella)",
            "crop": "Cotton",
            "type": "Lepidopteran Micro-Lepidoptera",
            "base_r0": 2.6,
            "optimal_temp_c": (25.0, 35.0),
            "wind_dispersal_factor": 1.2,
            "latent_period_days": 6,
            "barrier_advisory": "Mating disruption pheromone ropeladders (PB-Rope L); nocturnal light traps (1/acre); neem seed kernel extract (NSKE 5%) foliar spray."
        },
        "late_blight": {
            "name": "Late Blight (Phytophthora infestans)",
            "crop": "Potato / Tomato",
            "type": "Oomycete Zoosporangia",
            "base_r0": 3.8,
            "optimal_temp_c": (12.0, 20.0),
            "wind_dispersal_factor": 1.6,
            "latent_period_days": 5,
            "barrier_advisory": "Prophylactic application of Mancozeb 75% WP (2.5g/L); maintain raised beds to prevent stagnant water splash; rogue out infected focal plants immediately."
        }
    }

    @classmethod
    def simulate_epidemic_spread(
        cls,
        pathogen_key: str,
        origin_district: str,
        active_focal_cases: int,
        ambient_temp_c: float,
        ambient_humidity_pct: float,
        wind_speed_kmh: float,
        wind_azimuth_degrees: float = 45.0,  # 45 deg = North-East wind
        surrounding_crop_acreage: float = 3500.0
    ) -> Dict[str, Any]:
        """
        Simulate geospatial epidemiological transmission dynamics and calculate R0 reproduction factor.
        """
        path_info = cls.PATHOGEN_PROPERTIES.get(pathogen_key.lower(), cls.PATHOGEN_PROPERTIES["yellow_rust"])

        # Temperature suitability multiplier
        t_min, t_max = path_info["optimal_temp_c"]
        if t_min <= ambient_temp_c <= t_max:
            temp_mult = 1.35
        elif ambient_temp_c < t_min - 5.0 or ambient_temp_c > t_max + 8.0:
            temp_mult = 0.45
        else:
            temp_mult = 0.90

        # Humidity suitability multiplier (higher humidity favours fungal/oomycete spore germination)
        hum_mult = 1.30 if ambient_humidity_pct >= 75.0 else (1.0 if ambient_humidity_pct >= 55.0 else 0.60)

        # Wind velocity multiplier for aerial transmission
        wind_mult = min(2.5, 0.8 + (wind_speed_kmh / 15.0) * path_info["wind_dispersal_factor"])

        # Effective Reproduction Number Rt
        effective_r0 = round(path_info["base_r0"] * temp_mult * hum_mult * (wind_mult * 0.7), 2)

        # Dispersal speed in km/day
        dispersal_velocity_km_day = round(max(1.0, (wind_speed_kmh * 0.35) * path_info["wind_dispersal_factor"]), 1)

        # Downwind direction interpretation
        directions = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]
        dir_idx = int((wind_azimuth_degrees + 22.5) % 360 // 45)
        wind_direction_text = directions[dir_idx]

        # Threat classification
        if effective_r0 >= 2.5:
            threat_level = "RED ALERT - RAPID EPIDEMIC VECTOR"
            threat_color = "#E74C3C"
        elif effective_r0 >= 1.5:
            threat_level = "AMBER WARNING - REGIONAL TRANSMISSION DETECTED"
            threat_color = "#F39C12"
        else:
            threat_level = "YELLOW WATCH - LOCALIZED SPORADIC INFECTION"
            threat_color = "#27AE60"

        # Downwind farm cluster impact projection (24h, 48h, 72h)
        trajectory = []
        base_cases = active_focal_cases
        for day in [1, 2, 3]:
            dist_km = round(dispersal_velocity_km_day * day, 1)
            proj_infected_farms = int(base_cases * (effective_r0 ** (day * 0.7)))
            at_risk_acreage = round(min(surrounding_crop_acreage, proj_infected_farms * 3.2), 1)
            trajectory.append({
                "forecast_window": f"{day * 24} Hours",
                "corridor_reach_km": dist_km,
                "projected_active_foci": proj_infected_farms,
                "estimated_acreage_threatened": at_risk_acreage,
                "vector_status": "Containable with barrier spray" if day == 1 else "Active outbreak zone"
            })

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "origin_district": origin_district,
            "pathogen": path_info["name"],
            "target_crop": path_info["crop"],
            "pathogen_classification": path_info["type"],
            "epidemiological_metrics": {
                "base_reproduction_r0": path_info["base_r0"],
                "effective_reproduction_rt": effective_r0,
                "daily_dispersal_velocity_km": dispersal_velocity_km_day,
                "wind_corridor_heading": wind_direction_text,
                "wind_azimuth_deg": wind_azimuth_degrees,
                "wind_speed_kmh": wind_speed_kmh,
                "threat_level": threat_level,
                "threat_color": threat_color
            },
            "downwind_trajectory_forecast": trajectory,
            "prophylactic_barrier_protocol": {
                "advisory": path_info["barrier_advisory"],
                "quarantine_buffer_radius_km": round(dispersal_velocity_km_day * 1.5, 1),
                "community_action": "Broadcast automated push notification and SMS alert to all KVK nodes and registered KisanSeva farmers in the downwind sector."
            }
        }


# Global instance
epidemic_swarm_service = EpidemicSwarmService()
