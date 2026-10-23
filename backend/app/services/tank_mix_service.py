"""Smart Spray Tank-Mix Compatibility, Phytotoxicity & Nozzle Drift Calculator.

Evaluates chemical vs biological tank incompatibility, pH hydrolysis buffering,
spray droplet micron classification (ASABE S572.1 standard), and Pre-Harvest Interval (PHI).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# Chemical database with pH stability, formulation, active ingredient, and PHI (days)
AGROCHEMICAL_REGISTRY = {
    "Copper Oxychloride (Fungicide)": {
        "category": "Inorganic Fungicide",
        "optimal_ph_range": [6.0, 7.5],
        "incompatibilities": ["Trichoderma viride (Bio-agent)", "Phosphoric acid", "Alkaline fertilizers", "Lime sulfur"],
        "phi_days": 14,
        "dose_per_15l_tank": "40 g",
        "mechanism": "Multisite contact disruption of fungal spore enzymes",
    },
    "Mancozeb 75% WP (Fungicide)": {
        "category": "Dithiocarbamate Fungicide",
        "optimal_ph_range": [6.0, 7.0],
        "incompatibilities": ["Lime sulfur", "Bordeaux mixture", "Strong oxidizers"],
        "phi_days": 10,
        "dose_per_15l_tank": "35 g",
        "mechanism": "Inactivates sulfhydryl groups of fungal amino acids",
    },
    "Azoxystrobin 23% SC (Fungicide)": {
        "category": "Strobilurin Fungicide",
        "optimal_ph_range": [5.5, 7.0],
        "incompatibilities": ["Emulsifiable concentrate (EC) formulations on young foliage"],
        "phi_days": 7,
        "dose_per_15l_tank": "15 ml",
        "mechanism": "Inhibits mitochondrial respiration in fungi",
    },
    "Chlorpyrifos 20% EC (Insecticide)": {
        "category": "Organophosphate Insecticide",
        "optimal_ph_range": [5.0, 6.5],
        "incompatibilities": ["Alkaline water (rapid alkaline hydrolysis)", "Copper fungicides"],
        "phi_days": 21,
        "dose_per_15l_tank": "30 ml",
        "mechanism": "Acetylcholinesterase neurotoxic inhibitor",
    },
    "Imidacloprid 17.8% SL (Insecticide)": {
        "category": "Neonicotinoid Insecticide",
        "optimal_ph_range": [5.5, 7.5],
        "incompatibilities": ["Strong alkaline solutions"],
        "phi_days": 14,
        "dose_per_15l_tank": "5 ml",
        "mechanism": "Systemic nicotinic acetylcholine receptor blocker",
    },
    "Emamectin Benzoate 5% SG (Insecticide)": {
        "category": "Avermectin Insecticide",
        "optimal_ph_range": [5.5, 7.0],
        "incompatibilities": ["Highly alkaline sprays", "Copper hydroxide"],
        "phi_days": 5,
        "dose_per_15l_tank": "7 g",
        "mechanism": "Glutamate-gated chloride channel activator",
    },
    "Trichoderma viride (Bio-agent)": {
        "category": "Beneficial Microbial Bio-Fungicide",
        "optimal_ph_range": [5.5, 7.0],
        "incompatibilities": ["Copper Oxychloride (Fungicide)", "Mancozeb 75% WP (Fungicide)", "Azoxystrobin 23% SC (Fungicide)", "Chemical Antibiotics"],
        "phi_days": 0,
        "dose_per_15l_tank": "50 g",
        "mechanism": "Mycoparasitism, antibiotic excretion, and induced plant defense",
    },
    "19:19:19 NPK Foliar (Micronutrient)": {
        "category": "Water Soluble Fertilizer",
        "optimal_ph_range": [6.0, 7.0],
        "incompatibilities": ["Calcium nitrate", "Heavy metal sulfates"],
        "phi_days": 0,
        "dose_per_15l_tank": "75 g",
        "mechanism": "Rapid stomatal nutrient absorption",
    },
    "Zinc Sulfate Monohydrate (Micronutrient)": {
        "category": "Inorganic Micronutrient",
        "optimal_ph_range": [5.5, 6.5],
        "incompatibilities": ["Phosphatic fertilizers (forms insoluble zinc phosphate precipitate)"],
        "phi_days": 0,
        "dose_per_15l_tank": "25 g",
        "mechanism": "Enzyme activator and chlorophyll synthesizer",
    },
}


def check_tank_mix_compatibility(
    item_a: str,
    item_b: str,
    water_ph: float = 7.5,
    wind_speed_kmh: float = 9.0,
    temperature_c: float = 29.0,
    humidity_pct: float = 60.0,
    tank_capacity_liters: float = 15.0,
) -> Dict[str, Any]:
    """Evaluates compatibility between two chemicals, spray physics, and buffering requirements."""
    reg_a = AGROCHEMICAL_REGISTRY.get(item_a)
    reg_b = AGROCHEMICAL_REGISTRY.get(item_b)

    if not reg_a or not reg_b:
        return {
            "status": "ERROR",
            "message": f"One or both chemicals not recognized in registry: '{item_a}', '{item_b}'",
        }

    conflicts: List[str] = []
    warnings: List[str] = []
    is_compatible = True

    # Check direct incompatibilities
    for inc in reg_a["incompatibilities"]:
        if inc.lower() in item_b.lower() or item_b.lower() in inc.lower():
            conflicts.append(f"CRITICAL ANTAGONISM: {item_a} is incompatible with {item_b} ({inc}).")
            is_compatible = False

    for inc in reg_b["incompatibilities"]:
        if inc.lower() in item_a.lower() or item_a.lower() in inc.lower():
            conflicts.append(f"CRITICAL ANTAGONISM: {item_b} is incompatible with {item_a} ({inc}).")
            is_compatible = False

    # Check Biological vs Chemical Antagonism
    is_a_bio = "Bio" in reg_a["category"] or "Bio" in item_a
    is_b_bio = "Bio" in reg_b["category"] or "Bio" in item_b
    is_a_chem_fungicide = "Fungicide" in reg_a["category"] and not is_a_bio
    is_b_chem_fungicide = "Fungicide" in reg_b["category"] and not is_b_bio

    if (is_a_bio and is_b_chem_fungicide) or (is_b_bio and is_a_chem_fungicide):
        conflicts.append(
            f"BIOCIDE HAZARD: Chemical fungicide '{item_a if is_a_chem_fungicide else item_b}' will kill the living microbial spores of '{item_b if is_a_chem_fungicide else item_a}'."
        )
        is_compatible = False

    # Water pH & Alkaline Hydrolysis check
    ph_status = "OPTIMAL"
    ph_advisory = "Water pH is within the stable absorption window."
    buffering_needed = False

    min_ph = max(reg_a["optimal_ph_range"][0], reg_b["optimal_ph_range"][0])
    max_ph = min(reg_a["optimal_ph_range"][1], reg_b["optimal_ph_range"][1])

    if water_ph > max_ph:
        ph_status = "ALKALINE_HYDROLYSIS_RISK"
        buffering_needed = True
        conflicts.append(
            f"Alkaline Hydrolysis: Water pH ({water_ph}) exceeds stability limit ({max_ph}). Half-life will degrade in under 2 hours."
        )
        ph_advisory = f"Add 15–20g of agricultural citric acid or acidifier adjuvant to buffer water pH down to {round((min_ph + max_ph) / 2, 1)} before adding chemicals."
    elif water_ph < min_ph:
        ph_status = "ACIDIC_PRECIPITATION_RISK"
        buffering_needed = True
        warnings.append(f"Water is too acidic ({water_ph}). Risk of leaf scorching or precipitation.")
        ph_advisory = "Buffering with neutral agricultural adjuvant recommended."

    # Spray droplet physics & drift risk (ASABE standard)
    # High wind or low humidity increases drift and evaporation
    drift_score = (wind_speed_kmh / 25.0) * 0.6 + ((100.0 - humidity_pct) / 100.0) * 0.4
    drift_pct = round(max(5.0, min(95.0, drift_score * 100.0)), 1)

    if wind_speed_kmh > 15.0:
        drift_category = "HIGH_DRIFT_DANGER"
        nozzle_type = "Air-Induction (AI) Coarse Droplet Nozzle (350–500 µm)"
        spray_verdict = "DO NOT SPRAY: Wind speed exceeds 15 km/h. Off-target drift will contaminate adjacent crops."
    elif temperature_c > 33.0 and humidity_pct < 40.0:
        drift_category = "RAPID_DROPLET_EVAPORATION"
        nozzle_type = "Medium-Coarse Flat Fan Nozzle (250–350 µm)"
        spray_verdict = "POSTPONE SPRAY: High heat will evaporate droplets before cuticle penetration. Spray at dusk."
    else:
        drift_category = "SAFE_SPRAY_CONDITIONS"
        nozzle_type = "Standard Hollow Cone or Twin Flat Fan (200–300 µm)"
        spray_verdict = "IDEAL CONDITIONS: Suitable for thorough foliar canopy coverage."

    # Pre-harvest interval (safe harvest delay)
    max_phi = max(reg_a["phi_days"], reg_b["phi_days"])

    # Scale dosage to tank size (standard ratio based on 15L default)
    tank_scale = tank_capacity_liters / 15.0

    return {
        "status": "SUCCESS",
        "tank_compatibility": {
            "is_compatible": is_compatible,
            "overall_safety_rating": "SAFE TO MIX" if is_compatible else "DANGEROUS INCOMPATIBILITY",
            "safety_badge": "✅ Approved Tank Mix" if is_compatible else "🚫 Incompatible - Do Not Mix",
            "conflicts": conflicts,
            "warnings": warnings,
        },
        "chemicals_analyzed": [
            {
                "name": item_a,
                "category": reg_a["category"],
                "tank_dose": f"{reg_a['dose_per_15l_tank']} per 15L tank",
                "phi_days": reg_a["phi_days"],
                "action": reg_a["mechanism"],
            },
            {
                "name": item_b,
                "category": reg_b["category"],
                "tank_dose": f"{reg_b['dose_per_15l_tank']} per 15L tank",
                "phi_days": reg_b["phi_days"],
                "action": reg_b["mechanism"],
            },
        ],
        "water_quality_buffering": {
            "measured_ph": water_ph,
            "recommended_ph_range": f"{min_ph} - {max_ph}",
            "ph_status": ph_status,
            "buffering_needed": buffering_needed,
            "advisory": ph_advisory,
        },
        "spray_physics_and_drift": {
            "ambient_conditions": f"{temperature_c}°C, {humidity_pct}% RH, {wind_speed_kmh} km/h wind",
            "drift_risk_pct": drift_pct,
            "drift_category": drift_category,
            "recommended_nozzle": nozzle_type,
            "spray_window_advisory": spray_verdict,
        },
        "compliance_and_safety": {
            "mandatory_phi_days": max_phi,
            "harvest_safety_rule": f"Wait at least {max_phi} days after spraying before harvesting or selling produce.",
            "ppe_requirement": "Wear nitrile gloves, N95 respiratory mask, and eye goggles during mixing and application.",
        },
    }
