"""Multi-agent agronomy advisor.

Four cooperating specialist agents each produce findings plus a reasoning trace,
then a planner agent merges them into a prioritised, time-boxed action plan.
Rule-based and fully offline: no external LLM or API key required.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.models.plant_net import get_disease_metadata


@dataclass
class AgentReport:
    agent: str
    findings: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)  # {"priority": 1-3, "when": str, "action": str}
    trace: List[str] = field(default_factory=list)
    risk: float = 0.0  # 0..1


def soil_agent(soil_type: Optional[str], ph: Optional[float], n: Optional[float],
               p: Optional[float], k: Optional[float]) -> AgentReport:
    r = AgentReport("SoilAgent")
    if soil_type:
        r.trace.append(f"Soil classified as {soil_type}.")
        if soil_type in ("Clay Soil", "Black Soil"):
            r.findings.append("Heavy soil: slow drainage raises root-rot and fungal risk after rain.")
            r.actions.append({"priority": 2, "when": "Before next irrigation",
                              "action": "Use raised beds or furrows and avoid irrigating when topsoil is still wet."})
            r.risk += 0.2
        elif soil_type in ("Sandy Soil", "Red Soil"):
            r.findings.append("Light soil: fast leaching of nutrients and water.")
            r.actions.append({"priority": 2, "when": "This week",
                              "action": "Switch to split fertiliser doses and add mulch/compost to hold moisture."})
            r.risk += 0.15
    if ph is not None:
        r.trace.append(f"pH={ph}.")
        if ph < 5.5:
            r.findings.append(f"Soil is acidic (pH {ph}).")
            r.actions.append({"priority": 1, "when": "Before sowing",
                              "action": "Apply agricultural lime (about 2-4 q/ha, adjust to a soil test)."})
            r.risk += 0.25
        elif ph > 8.0:
            r.findings.append(f"Soil is alkaline (pH {ph}).")
            r.actions.append({"priority": 1, "when": "Before sowing",
                              "action": "Apply gypsum or elemental sulphur and add organic matter."})
            r.risk += 0.25
    for name, val, low, fix in (("Nitrogen", n, 50, "urea or compost"),
                                ("Phosphorus", p, 20, "single super phosphate (SSP) or DAP"),
                                ("Potassium", k, 30, "muriate of potash (MOP)")):
        if val is not None and val < low:
            r.findings.append(f"{name} is low ({val}).")
            r.actions.append({"priority": 2, "when": "Next fertiliser round",
                              "action": f"Apply {fix} to correct {name.lower()} deficiency."})
            r.risk += 0.1
    r.risk = min(1.0, r.risk)
    return r


def disease_agent(condition: Optional[str], confidence: Optional[float]) -> AgentReport:
    r = AgentReport("DiseaseAgent")
    if not condition:
        r.trace.append("No plant diagnosis supplied.")
        return r
    conf = confidence or 0.0
    meta = get_disease_metadata(condition)
    healthy = "healthy" in condition.lower()
    r.trace.append(f"Diagnosis={condition} (confidence {conf:.0%}).")
    if healthy:
        r.findings.append("Foliage appears healthy.")
        r.actions.append({"priority": 3, "when": "Weekly",
                          "action": "Continue scouting and keep preventive practices."})
        return r
    r.findings.append(f"Likely {condition} ({meta['pathogen_type']}), severity {meta['severity_level']}.")
    if conf < 0.6:
        r.findings.append("Diagnosis confidence is low: confirm with a better photo or a local extension officer.")
        r.actions.append({"priority": 1, "when": "Today", "action": "Retake a close, well-lit photo of a single affected leaf."})
    urgent = 1 if meta["severity_level"] == "Severe" else 2
    for step in meta["organic_treatment"][:2]:
        r.actions.append({"priority": urgent, "when": "Within 48 hours", "action": step})
    if conf >= 0.6 and meta["chemical_treatment"]:
        r.actions.append({"priority": urgent, "when": "If spread continues after 3-5 days",
                          "action": meta["chemical_treatment"][0]})
    r.risk = (0.8 if meta["severity_level"] == "Severe" else 0.5) * max(conf, 0.3)
    return r


def weather_agent(temp_c: Optional[float], humidity: Optional[float],
                  rainfall_mm: Optional[float], condition: Optional[str]) -> AgentReport:
    r = AgentReport("WeatherRiskAgent")
    fungal = bool(condition) and "healthy" not in (condition or "").lower()
    if humidity is not None and temp_c is not None:
        r.trace.append(f"T={temp_c}C RH={humidity}% rain={rainfall_mm}mm.")
        if humidity >= 80 and 15 <= temp_c <= 30:
            r.findings.append("Warm and humid weather strongly favours fungal and bacterial spread.")
            r.actions.append({"priority": 1 if fungal else 2, "when": "Next 3 days",
                              "action": "Spray protectively in the early morning and avoid overhead irrigation."})
            r.risk += 0.5
        if temp_c >= 38:
            r.findings.append("Heat stress risk.")
            r.actions.append({"priority": 2, "when": "Afternoons",
                              "action": "Irrigate in evening/early morning and consider shade net or mulching."})
            r.risk += 0.3
    if rainfall_mm is not None and rainfall_mm >= 50:
        r.findings.append("Heavy rain expected or recent: waterlogging and spray wash-off risk.")
        r.actions.append({"priority": 2, "when": "Before rain",
                          "action": "Clear drainage channels and re-apply sprays after rain stops."})
        r.risk += 0.3
    r.risk = min(1.0, r.risk)
    return r


def planner_agent(reports: List[AgentReport]) -> Dict[str, Any]:
    actions: List[Dict[str, Any]] = []
    seen = set()
    for rep in reports:
        for a in rep.actions:
            if a["action"] in seen:
                continue
            seen.add(a["action"])
            actions.append({**a, "source": rep.agent})
    actions.sort(key=lambda a: a["priority"])
    risk = max([rep.risk for rep in reports] + [0.0])
    level = "High" if risk >= 0.6 else "Moderate" if risk >= 0.3 else "Low"
    return {
        "overall_risk_score": round(risk, 2),
        "overall_risk_level": level,
        "action_plan": actions,
    }


def run_agents(
    soil_type: Optional[str] = None,
    ph: Optional[float] = None,
    nitrogen: Optional[float] = None,
    phosphorus: Optional[float] = None,
    potassium: Optional[float] = None,
    condition: Optional[str] = None,
    confidence: Optional[float] = None,
    temperature_c: Optional[float] = None,
    humidity: Optional[float] = None,
    rainfall_mm: Optional[float] = None,
) -> Dict[str, Any]:
    reports = [
        soil_agent(soil_type, ph, nitrogen, phosphorus, potassium),
        disease_agent(condition, confidence),
        weather_agent(temperature_c, humidity, rainfall_mm, condition),
    ]
    plan = planner_agent(reports)
    return {
        **plan,
        "agents": [
            {"agent": r.agent, "findings": r.findings, "reasoning_trace": r.trace, "risk": round(r.risk, 2)}
            for r in reports
        ],
    }
