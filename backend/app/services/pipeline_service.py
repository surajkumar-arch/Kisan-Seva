"""Unified Pipeline Orchestration & Farm Diagnostics Service for Phase 6."""

import io
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

from backend.app.schemas.input_schema import (
    SoilParameters,
    EnvironmentalParameters,
    ImageMetadata,
    AnalysisRecord,
)
from backend.app.schemas.soil_schema import SoilPredictionResponse
from backend.app.schemas.plant_schema import PlantPredictionResponse
from backend.app.schemas.fusion_schema import FusionResponse
from backend.app.schemas.crop_schema import CropRecommendationResponse
from backend.app.schemas.pipeline_schema import (
    FarmAlert,
    ActionPlanItem,
    FarmDiagnosticsSummary,
    PipelineAnalyzeResponse,
    PipelineHistoryItem,
    PipelineHistoryResponse,
    RegionalSummary,
    BatchAnalyzeResponse,
    PipelineStatsResponse,
)
from backend.app.services.validation import (
    save_analysis_record,
    get_analysis_record_by_request_id,
    generate_request_id,
)
from backend.app.utils.image_utils import save_uploaded_image
from backend.app.services.soil_health_service import (
    predict_soil_health,
    init_phase2_db,
)
from backend.app.services.plant_health_service import (
    predict_plant_health,
    init_phase3_db,
)
from backend.app.services.fusion_service import (
    execute_feature_fusion,
    init_phase4_db,
)
from backend.app.services.crop_service import (
    generate_crop_recommendations,
    init_phase5_db,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DB_PATH = PROJECT_ROOT / "data" / "smart_agriculture.db"
UPLOADS_DIR = BACKEND_DIR / "uploads"


def resolve_image_path(rel_path: str) -> Optional[Path]:
    """Resolve an image relative path robustly across backend and project root."""
    if not rel_path:
        return None
    p1 = BACKEND_DIR / rel_path
    if p1.exists():
        return p1
    p2 = PROJECT_ROOT / rel_path
    if p2.exists():
        return p2
    p3 = Path(rel_path)
    if p3.exists():
        return p3
    return None

# Fast in-memory cache for recent pipeline analyses
_PIPELINE_CACHE: Dict[str, PipelineAnalyzeResponse] = {}


def init_phase6_db(db_path: Optional[Path] = None) -> None:
    """Ensure pipeline_runs table exists in SQLite database."""
    target_path = db_path or DB_PATH
    os.makedirs(target_path.parent, exist_ok=True)

    with sqlite3.connect(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                farm_health_score INTEGER NOT NULL,
                farm_vigor_index INTEGER NOT NULL,
                soil_type TEXT NOT NULL,
                plant_condition TEXT NOT NULL,
                primary_crop TEXT NOT NULL,
                primary_suitability REAL NOT NULL,
                projected_yield TEXT NOT NULL,
                critical_alerts_count INTEGER NOT NULL,
                summary_json TEXT NOT NULL,
                FOREIGN KEY (request_id) REFERENCES analysis_records(request_id)
            );
            """
        )
        conn.commit()


def synthesize_farm_diagnostics(
    soil_resp: SoilPredictionResponse,
    plant_resp: PlantPredictionResponse,
    fusion_resp: FusionResponse,
    crop_resp: CropRecommendationResponse,
    soil_params: Optional[SoilParameters],
    env_params: Optional[EnvironmentalParameters],
    latency_breakdown: Dict[str, float],
) -> FarmDiagnosticsSummary:
    """Synthesize findings from all 5 multi-modal phases into an executive diagnostic report."""
    # 1. Overall Farm Health Score (0 - 100)
    # Soil fertility (35%), Plant Foliage Health (35%), Multi-modal Vigor (30%)
    s_score = soil_resp.soil_health_score
    p_score = plant_resp.plant_health_score
    fvi = fusion_resp.farm_vigor_index

    farm_health = int(round((s_score * 0.35) + (p_score * 0.35) + (fvi * 0.30)))
    farm_health = max(0, min(100, farm_health))

    if farm_health >= 80:
        status_str = "Thriving & Resilient"
    elif farm_health >= 65:
        status_str = "Good Condition (Minor Stress)"
    elif farm_health >= 50:
        status_str = "Fair (Needs Agronomic Attention)"
    else:
        status_str = "Degraded / High Risk (Urgent Action Required)"

    # 2. Alerts Generation
    alerts: List[FarmAlert] = []

    # Disease alert
    if not plant_resp.is_healthy:
        level = "CRITICAL" if plant_resp.severity_level in ["High", "Severe", "Critical"] else "WARNING"
        action = plant_resp.chemical_treatment[0] if plant_resp.chemical_treatment else "Apply recommended systemic fungicide."
        alerts.append(
            FarmAlert(
                level=level,
                category="DISEASE",
                title=f"Pathogen Outbreak: {plant_resp.condition}",
                message=f"Plant foliage displays active {plant_resp.condition} with {plant_resp.severity_level} severity (confidence: {int(plant_resp.confidence * 100)}%).",
                recommended_action=f"Immediate Foliar Action: {action}",
            )
        )

    # Soil Chemistry alerts
    if soil_params:
        if soil_params.ph is not None:
            if soil_params.ph < 5.5:
                alerts.append(
                    FarmAlert(
                        level="WARNING",
                        category="SOIL",
                        title="Strong Soil Acidity Detected",
                        message=f"Soil pH is {soil_params.ph} (Acidic). Reduces availability of P, Ca, and Mg.",
                        recommended_action="Apply agricultural lime (calcium carbonate) or dolomite at 2.0 - 2.5 t/ha.",
                    )
                )
            elif soil_params.ph > 8.0:
                alerts.append(
                    FarmAlert(
                        level="WARNING",
                        category="SOIL",
                        title="High Soil Alkalinity Detected",
                        message=f"Soil pH is {soil_params.ph} (Alkaline). Restricts micronutrient availability (Fe, Zn).",
                        recommended_action="Apply agricultural gypsum and incorporate acidifying organic mulch.",
                    )
                )

        if soil_params.nitrogen is not None and soil_params.nitrogen < 120:
            alerts.append(
                FarmAlert(
                    level="WARNING",
                    category="NUTRIENT",
                    title="Low Nitrogen Level",
                    message=f"Available Nitrogen is low ({soil_params.nitrogen} kg/ha). Stunts vegetative growth.",
                    recommended_action="Top-dress with Urea or apply well-rotted farmyard manure.",
                )
            )

        if soil_params.phosphorus is not None and soil_params.phosphorus < 25:
            alerts.append(
                FarmAlert(
                    level="WARNING",
                    category="NUTRIENT",
                    title="Low Phosphorus Level",
                    message=f"Available Phosphorus is {soil_params.phosphorus} kg/ha. Inhibits root development.",
                    recommended_action="Apply Single Super Phosphate (SSP) or Diammonium Phosphate (DAP) at sowing.",
                )
            )

        if soil_params.potassium is not None and soil_params.potassium < 120:
            alerts.append(
                FarmAlert(
                    level="INFO",
                    category="NUTRIENT",
                    title="Sub-optimal Potassium Level",
                    message=f"Available Potassium is {soil_params.potassium} kg/ha. Reduces disease and drought resistance.",
                    recommended_action="Apply Muriate of Potash (MOP) at 50 kg/ha.",
                )
            )

    # Environmental stress alerts
    if env_params:
        if env_params.temperature is not None and env_params.temperature > 38.0:
            alerts.append(
                FarmAlert(
                    level="WARNING",
                    category="CLIMATE",
                    title="Severe Thermal Heat Stress",
                    message=f"Ambient temperature is {env_params.temperature}°C, accelerating transpiration and moisture loss.",
                    recommended_action="Initiate protective drip irrigation during early mornings and apply straw mulching.",
                )
            )
        if env_params.rainfall is not None and env_params.rainfall < 40.0:
            alerts.append(
                FarmAlert(
                    level="WARNING",
                    category="CLIMATE",
                    title="Drought & Low Moisture Deficit",
                    message=f"Rainfall is critically low ({env_params.rainfall} mm). Water deficit imminent.",
                    recommended_action="Schedule supplementary irrigation and select drought-tolerant crop varieties.",
                )
            )

    # 3. Action Plan Synthesis
    action_plan: List[ActionPlanItem] = []
    priority_idx = 1

    # Step 1: Immediate Disease Spraying (if diseased)
    if not plant_resp.is_healthy:
        spray_text = plant_resp.organic_treatment[0] if plant_resp.organic_treatment else "Prune and treat infected foliage."
        action_plan.append(
            ActionPlanItem(
                priority=priority_idx,
                category="Plant Protection",
                action=f"Treat {plant_resp.condition}",
                timeframe="Immediate (0-48 hours)",
                details=f"Apply {spray_text}. Follow with preventive sanitation.",
            )
        )
        priority_idx += 1

    # Step 2: Soil Conditioning
    if soil_resp.recommendations:
        action_plan.append(
            ActionPlanItem(
                priority=priority_idx,
                category="Soil Amendment",
                action=f"Prepare {soil_resp.soil_type} for Planting",
                timeframe="Pre-Planting (1-2 Weeks)",
                details=soil_resp.recommendations[0],
            )
        )
        priority_idx += 1

    # Step 3: Crop Sowing & Variety Selection
    top_crop = crop_resp.primary_crop
    action_plan.append(
        ActionPlanItem(
            priority=priority_idx,
            category="Crop Sowing",
            action=f"Sow {top_crop} (Top Recommended)",
            timeframe="Optimal Sowing Window",
            details=f"Projected yield: {crop_resp.recommendations[0].projected_yield}. Water requirement: {crop_resp.recommendations[0].water_requirement}.",
        )
    )
    priority_idx += 1

    # Step 4: Water Management
    action_plan.append(
        ActionPlanItem(
            priority=priority_idx,
            category="Irrigation & Field Management",
            action="Calibrate Water Supply",
            timeframe="Throughout Growing Season",
            details=f"Maintain optimum root-zone moisture for {crop_resp.recommendations[0].water_requirement} demand under current rainfall conditions.",
        )
    )

    return FarmDiagnosticsSummary(
        farm_health_score=farm_health,
        farm_status=status_str,
        soil_health_score=s_score,
        plant_health_score=p_score,
        farm_vigor_index=fvi,
        soil_type=soil_resp.soil_type,
        plant_condition=plant_resp.condition,
        primary_crop=crop_resp.primary_crop,
        primary_suitability=crop_resp.primary_suitability,
        projected_yield=crop_resp.recommendations[0].projected_yield if crop_resp.recommendations else "N/A",
        critical_alerts=alerts,
        action_plan=action_plan,
        execution_time_ms=latency_breakdown,
    )


def orchestrate_pipeline(
    request_id: Optional[str] = None,
    soil_image: Optional[Image.Image] = None,
    plant_image: Optional[Image.Image] = None,
    soil_params: Optional[SoilParameters] = None,
    env_params: Optional[EnvironmentalParameters] = None,
    force_refresh: bool = False,
    db_path: Optional[Path] = None,
) -> PipelineAnalyzeResponse:
    """Execute end-to-end multi-modal analysis (Phase 1 through Phase 5) with full orchestration."""
    target_db = db_path or DB_PATH
    t0_total = time.perf_counter()
    latency: Dict[str, float] = {}

    init_phase6_db(target_db)

    # 1. Check in-memory cache first if request_id provided
    if request_id and not force_refresh and request_id in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[request_id]

    # 2. Phase 1: Input Acquisition or Record Retrieval
    t0_p1 = time.perf_counter()
    if request_id:
        record = get_analysis_record_by_request_id(request_id, db_path=target_db)
        if not record:
            raise ValueError(f"Request ID '{request_id}' not found in database.")
        
        # Load images if not provided
        if soil_image is None and record.get("soil_image_path"):
            full_soil_path = resolve_image_path(record["soil_image_path"])
            if full_soil_path and full_soil_path.exists():
                soil_image = Image.open(full_soil_path).convert("RGB")

        if plant_image is None and record.get("plant_image_path"):
            full_plant_path = resolve_image_path(record["plant_image_path"])
            if full_plant_path and full_plant_path.exists():
                plant_image = Image.open(full_plant_path).convert("RGB")

        # Extract parameters if not passed directly
        if soil_params is None:
            soil_params = SoilParameters(
                ph=record.get("ph"),
                nitrogen=record.get("nitrogen"),
                phosphorus=record.get("phosphorus"),
                potassium=record.get("potassium"),
            )
        if env_params is None:
            env_params = EnvironmentalParameters(
                temperature=record.get("temperature"),
                humidity=record.get("humidity"),
                rainfall=record.get("rainfall"),
            )
        active_req_id = request_id
    else:
        # Require images for brand new acquisition
        if soil_image is None or plant_image is None:
            raise ValueError("Both soil_image and plant_image are required when request_id is not specified.")
        
        active_req_id = generate_request_id()
        
        # Save processed images
        soil_buf = io.BytesIO()
        soil_image.save(soil_buf, format="JPEG")
        soil_bytes = soil_buf.getvalue()

        plant_buf = io.BytesIO()
        plant_image.save(plant_buf, format="JPEG")
        plant_bytes = plant_buf.getvalue()

        soil_fname, _ = save_uploaded_image(
            soil_bytes,
            f"soil_{active_req_id}.jpg",
            str(UPLOADS_DIR / "soil"),
        )
        plant_fname, _ = save_uploaded_image(
            plant_bytes,
            f"plant_{active_req_id}.jpg",
            str(UPLOADS_DIR / "plant"),
        )

        soil_meta = ImageMetadata(
            filename=soil_fname,
            path=str(Path("uploads/soil") / soil_fname),
            content_type="image/jpeg",
            size_bytes=len(soil_bytes),
            dimensions=soil_image.size,
        )
        plant_meta = ImageMetadata(
            filename=plant_fname,
            path=str(Path("uploads/plant") / plant_fname),
            content_type="image/jpeg",
            size_bytes=len(plant_bytes),
            dimensions=plant_image.size,
        )

        # Insert Phase 1 record
        rec = AnalysisRecord(
            request_id=active_req_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            soil_image=soil_meta,
            plant_image=plant_meta,
            soil_parameters=soil_params or SoilParameters(),
            environment=env_params or EnvironmentalParameters(),
        )
        save_analysis_record(rec, db_path=target_db)
        record = get_analysis_record_by_request_id(active_req_id, db_path=target_db)

    latency["phase1_acquisition_ms"] = round((time.perf_counter() - t0_p1) * 1000, 2)

    # 3. Phase 2: Soil Health Detection
    t0_p2 = time.perf_counter()
    if soil_image is None:
        raise ValueError(f"Soil image unavailable for request_id '{active_req_id}'.")
    soil_resp = predict_soil_health(
        soil_image=soil_image,
        request_id=active_req_id,
        soil_params=soil_params,
        db_path=target_db,
    )
    latency["phase2_soil_ms"] = round((time.perf_counter() - t0_p2) * 1000, 2)

    # 4. Phase 3: Plant Health Detection
    t0_p3 = time.perf_counter()
    if plant_image is None:
        raise ValueError(f"Plant image unavailable for request_id '{active_req_id}'.")
    plant_resp = predict_plant_health(
        plant_image=plant_image,
        request_id=active_req_id,
        db_path=target_db,
    )
    latency["phase3_plant_ms"] = round((time.perf_counter() - t0_p3) * 1000, 2)

    # 5. Phase 4: Feature Fusion
    t0_p4 = time.perf_counter()
    fusion_resp = execute_feature_fusion(
        request_id=active_req_id,
        db_path=target_db,
    )
    latency["phase4_fusion_ms"] = round((time.perf_counter() - t0_p4) * 1000, 2)

    # 6. Phase 5: Crop Recommendation
    t0_p5 = time.perf_counter()
    crop_resp = generate_crop_recommendations(
        request_id=active_req_id,
        db_path=target_db,
    )
    latency["phase5_crop_ms"] = round((time.perf_counter() - t0_p5) * 1000, 2)

    # Total Latency
    latency["total_pipeline_ms"] = round((time.perf_counter() - t0_total) * 1000, 2)

    # 7. Synthesize Diagnostics
    summary = synthesize_farm_diagnostics(
        soil_resp=soil_resp,
        plant_resp=plant_resp,
        fusion_resp=fusion_resp,
        crop_resp=crop_resp,
        soil_params=soil_params,
        env_params=env_params,
        latency_breakdown=latency,
    )

    # 8. Record in pipeline_runs table
    with sqlite3.connect(target_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO pipeline_runs (
                request_id,
                created_at,
                farm_health_score,
                farm_vigor_index,
                soil_type,
                plant_condition,
                primary_crop,
                primary_suitability,
                projected_yield,
                critical_alerts_count,
                summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                active_req_id,
                datetime.now(timezone.utc).isoformat(),
                summary.farm_health_score,
                summary.farm_vigor_index,
                summary.soil_type,
                summary.plant_condition,
                summary.primary_crop,
                summary.primary_suitability,
                summary.projected_yield,
                len(summary.critical_alerts),
                summary.model_dump_json(),
            ),
        )
        conn.commit()

    # Build final response
    response = PipelineAnalyzeResponse(
        status="success",
        message="End-to-end agricultural intelligence pipeline executed successfully",
        request_id=active_req_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        summary=summary,
        phase1_input=record or {},
        phase2_soil=soil_resp,
        phase3_plant=plant_resp,
        phase4_fusion=fusion_resp,
        phase5_crop=crop_resp,
    )

    # Store in memory cache
    _PIPELINE_CACHE[active_req_id] = response
    return response


def get_pipeline_history_records(
    limit: int = 20,
    offset: int = 0,
    min_health_score: Optional[int] = None,
    soil_type: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> PipelineHistoryResponse:
    """Retrieve historical pipeline executions with optional filtering."""
    target_db = db_path or DB_PATH
    init_phase6_db(target_db)

    query = "SELECT * FROM pipeline_runs WHERE 1=1"
    params: List[Any] = []

    if min_health_score is not None:
        query += " AND farm_health_score >= ?"
        params.append(min_health_score)

    if soil_type:
        query += " AND soil_type = ?"
        params.append(soil_type)

    # Count total
    count_query = query.replace("SELECT *", "SELECT COUNT(*)")
    with sqlite3.connect(target_db) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(count_query, params)
        total_records = cur.fetchone()[0]

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cur.execute(query, params)
        rows = cur.fetchall()

    records = [
        PipelineHistoryItem(
            request_id=row["request_id"],
            created_at=row["created_at"],
            farm_health_score=row["farm_health_score"],
            farm_vigor_index=row["farm_vigor_index"],
            soil_type=row["soil_type"],
            plant_condition=row["plant_condition"],
            primary_crop=row["primary_crop"],
            primary_suitability=row["primary_suitability"],
            projected_yield=row["projected_yield"],
            critical_alerts_count=row["critical_alerts_count"],
        )
        for row in rows
    ]

    return PipelineHistoryResponse(
        status="success",
        total_records=total_records,
        limit=limit,
        offset=offset,
        records=records,
    )


def get_pipeline_statistics(db_path: Optional[Path] = None) -> PipelineStatsResponse:
    """Aggregate system-level agronomic intelligence metrics."""
    target_db = db_path or DB_PATH
    init_phase6_db(target_db)

    with sqlite3.connect(target_db) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM pipeline_runs")
        rows = cur.fetchall()

    if not rows:
        return PipelineStatsResponse(
            status="success",
            total_farms_analyzed=0,
            avg_farm_health_score=0.0,
            avg_farm_vigor_index=0.0,
            soil_distribution={},
            disease_distribution={},
            crop_recommendation_distribution={},
            health_grade_distribution={},
        )

    total = len(rows)
    avg_health = sum(r["farm_health_score"] for r in rows) / total
    avg_vigor = sum(r["farm_vigor_index"] for r in rows) / total

    soil_dist: Dict[str, int] = {}
    disease_dist: Dict[str, int] = {}
    crop_dist: Dict[str, int] = {}
    health_grade: Dict[str, int] = {"Thriving (80-100)": 0, "Good (65-79)": 0, "Fair (50-64)": 0, "Degraded (<50)": 0}

    for r in rows:
        s_type = r["soil_type"]
        soil_dist[s_type] = soil_dist.get(s_type, 0) + 1

        p_cond = r["plant_condition"]
        disease_dist[p_cond] = disease_dist.get(p_cond, 0) + 1

        p_crop = r["primary_crop"]
        crop_dist[p_crop] = crop_dist.get(p_crop, 0) + 1

        hs = r["farm_health_score"]
        if hs >= 80:
            health_grade["Thriving (80-100)"] += 1
        elif hs >= 65:
            health_grade["Good (65-79)"] += 1
        elif hs >= 50:
            health_grade["Fair (50-64)"] += 1
        else:
            health_grade["Degraded (<50)"] += 1

    return PipelineStatsResponse(
        status="success",
        total_farms_analyzed=total,
        avg_farm_health_score=round(avg_health, 1),
        avg_farm_vigor_index=round(avg_vigor, 1),
        soil_distribution=soil_dist,
        disease_distribution=disease_dist,
        crop_recommendation_distribution=crop_dist,
        health_grade_distribution=health_grade,
    )


def generate_markdown_report(result: PipelineAnalyzeResponse) -> str:
    """Generate a clean GFM markdown agronomic diagnostic report."""
    s = result.summary
    p2 = result.phase2_soil
    p3 = result.phase3_plant
    p5 = result.phase5_crop

    alerts_md = ""
    if s.critical_alerts:
        alerts_md = "\n### ⚠️ Active Farm Hazard Alerts\n"
        for a in s.critical_alerts:
            alerts_md += f"- **[{a.level}] {a.title}**: {a.message}\n  *Action*: {a.recommended_action}\n"
    else:
        alerts_md = "\n> [!NOTE]\n> No critical hazards detected. Field parameters are within safe agronomic thresholds.\n"

    actions_md = "\n### 📋 Prioritized Agronomic Action Plan\n"
    for item in s.action_plan:
        actions_md += f"{item.priority}. **{item.action}** ({item.timeframe})\n   - *Category*: {item.category}\n   - *Details*: {item.details}\n"

    crops_md = "\n### 🌾 Top Recommended Crops & Yield Potential\n| Rank | Crop | Suitability | Compatibility Grade | Projected Yield | Water Need |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for c in p5.recommendations:
        crops_md += f"| #{c.rank} | **{c.crop_name}** | {c.suitability_score}% | {c.compatibility_grade} | {c.projected_yield} | {c.water_requirement} |\n"

    report = f"""# 🌱 KisanSeva: Farm Diagnostic & Agronomic Prescription Report

**Request ID**: `{result.request_id}`  
**Generated On**: `{result.timestamp}`  
**Status**: `{s.farm_status}`  

---

## 📊 Executive Farm Health Metrics
| Metric | Score | Assessment |
| :--- | :--- | :--- |
| **Overall Farm Health Score** | **{s.farm_health_score}/100** | {s.farm_status} |
| **Soil Health & Fertility** | **{s.soil_health_score}/100** | {p2.fertility_grade} ({p2.soil_type}) |
| **Plant Foliage Health** | **{s.plant_health_score}/100** | {p3.condition} ({p3.severity_level}) |
| **Multi-Modal Farm Vigor (FVI)** | **{s.farm_vigor_index}/100** | {result.phase4_fusion.vigor_status} |

---
{alerts_md}
---
{actions_md}
---
{crops_md}

---
### 🔬 Technical Soil & Climate Context
- **Soil Classification**: {p2.soil_type} (Confidence: {int(p2.confidence * 100)}%)
- **Soil Chemistry**: pH: {p2.chemical_assessment.ph_status or 'N/A'}, N: {p2.chemical_assessment.nitrogen_status or 'N/A'}, P: {p2.chemical_assessment.phosphorus_status or 'N/A'}, K: {p2.chemical_assessment.potassium_status or 'N/A'}
- **Plant Pathology**: Condition: {p3.condition}, Pathogen: {p3.pathogen_type}, Severity: {p3.severity_level}
- **Primary Organic Treatment**: {p3.organic_treatment[0] if p3.organic_treatment else 'None'}
- **Primary Chemical Treatment**: {p3.chemical_treatment[0] if p3.chemical_treatment else 'None'}
- **Pipeline Execution Latency**: Total: {s.execution_time_ms.get('total_pipeline_ms', 'N/A')} ms
"""
    return report


def generate_html_report(result: PipelineAnalyzeResponse) -> str:
    """Generate a printable HTML agronomic report with CSS styling."""
    s = result.summary
    p2 = result.phase2_soil
    p3 = result.phase3_plant
    p5 = result.phase5_crop

    # Status color
    color = "#10b981" if s.farm_health_score >= 80 else ("#f59e0b" if s.farm_health_score >= 60 else "#ef4444")

    alerts_html = ""
    for a in s.critical_alerts:
        border_col = "#ef4444" if a.level == "CRITICAL" else "#f59e0b"
        bg_col = "#fef2f2" if a.level == "CRITICAL" else "#fffbeb"
        alerts_html += f"""
        <div style="border-left: 4px solid {border_col}; background: {bg_col}; padding: 12px; margin-bottom: 10px; border-radius: 4px;">
            <strong style="color: {border_col};">[{a.level}] {a.title}</strong>
            <p style="margin: 4px 0; color: #374151;">{a.message}</p>
            <small style="color: #4b5563;"><b>Action:</b> {a.recommended_action}</small>
        </div>
        """

    actions_html = ""
    for item in s.action_plan:
        actions_html += f"""
        <li style="margin-bottom: 10px;">
            <strong>{item.action}</strong> <span style="background: #e0f2fe; color: #0284c7; padding: 2px 8px; border-radius: 12px; font-size: 11px;">{item.timeframe}</span><br/>
            <span style="color: #6b7280; font-size: 13px;">Category: {item.category} — {item.details}</span>
        </li>
        """

    crops_rows = ""
    for c in p5.recommendations:
        crops_rows += f"""
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 10px; font-weight: bold;">#{c.rank}</td>
            <td style="padding: 10px; font-weight: bold; color: #059669;">{c.crop_name}</td>
            <td style="padding: 10px;">{c.suitability_score}%</td>
            <td style="padding: 10px;"><span style="background: #ecfdf5; color: #059669; padding: 2px 8px; border-radius: 10px;">{c.compatibility_grade}</span></td>
            <td style="padding: 10px;">{c.projected_yield}</td>
            <td style="padding: 10px;">{c.water_requirement}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Farm Diagnostic Report - {result.request_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f9fafb; color: #111827; margin: 0; padding: 30px; }}
        .card {{ background: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); padding: 30px; max-width: 900px; margin: 0 auto; }}
        .header {{ border-bottom: 2px solid #f3f4f6; padding-bottom: 20px; margin-bottom: 25px; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px; }}
        .metric-box {{ background: #f3f4f6; border-radius: 8px; padding: 15px; text-align: center; }}
        .metric-val {{ font-size: 26px; font-weight: 800; color: #111827; }}
        .metric-lbl {{ font-size: 12px; color: #6b7280; text-transform: uppercase; font-weight: 600; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th {{ background: #f9fafb; text-align: left; padding: 10px; font-size: 13px; color: #4b5563; border-bottom: 2px solid #e5e7eb; }}
        @media print {{ body {{ background: #fff; padding: 0; }} .card {{ box-shadow: none; padding: 0; }} .no-print {{ display: none; }} }}
    </style>
</head>
<body>
    <div class="card">
        <div class="no-print" style="margin-bottom: 20px; text-align: right;">
            <button onclick="window.print()" style="background: #059669; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600;">🖨️ Print / Save as PDF</button>
        </div>
        <div class="header">
            <h1 style="color: #065f46; margin: 0 0 8px 0;">🌱 KisanSeva: Farm Diagnostic & Agronomic Prescription</h1>
            <p style="margin: 0; color: #6b7280; font-size: 14px;">Request ID: <b>{result.request_id}</b> | Generated: {result.timestamp}</p>
        </div>

        <div class="grid">
            <div class="metric-box" style="border-top: 4px solid {color};">
                <div class="metric-val" style="color: {color};">{s.farm_health_score}/100</div>
                <div class="metric-lbl">Farm Health Score</div>
            </div>
            <div class="metric-box" style="border-top: 4px solid #10b981;">
                <div class="metric-val">{s.soil_health_score}/100</div>
                <div class="metric-lbl">Soil Fertility ({p2.soil_type})</div>
            </div>
            <div class="metric-box" style="border-top: 4px solid #3b82f6;">
                <div class="metric-val">{s.plant_health_score}/100</div>
                <div class="metric-lbl">Plant Foliage ({p3.condition})</div>
            </div>
            <div class="metric-box" style="border-top: 4px solid #8b5cf6;">
                <div class="metric-val">{s.farm_vigor_index}/100</div>
                <div class="metric-lbl">Farm Vigor (FVI)</div>
            </div>
        </div>

        <h3 style="color: #374151; margin-top: 25px;">Active Alerts</h3>
        {alerts_html if s.critical_alerts else '<p style="color: #059669;">No active hazard alerts. Parameters are in safe range.</p>'}

        <h3 style="color: #374151; margin-top: 25px;">Prioritized Action Plan</h3>
        <ol style="padding-left: 20px;">
            {actions_html}
        </ol>

        <h3 style="color: #374151; margin-top: 25px;">Recommended Crop Varieties</h3>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Crop</th>
                    <th>Suitability</th>
                    <th>Grade</th>
                    <th>Projected Yield</th>
                    <th>Water Need</th>
                </tr>
            </thead>
            <tbody>
                {crops_rows}
            </tbody>
        </table>
    </div>
</body>
</html>"""
    return html


def generate_agronomic_prescription(analysis: PipelineAnalyzeResponse) -> Dict[str, Any]:
    """Generate structured medical-grade agricultural prescription for extension officers."""
    summary = analysis.summary
    p2 = analysis.phase2_soil
    p3 = analysis.phase3_plant
    p5 = analysis.phase5_crop

    return {
        "prescription_id": f"RX-{analysis.request_id}",
        "date_issued": analysis.timestamp,
        "farm_id": analysis.request_id,
        "vital_signs": {
            "overall_health_score": f"{summary.farm_health_score}/100 ({summary.farm_status})",
            "soil_fertility": f"{summary.soil_health_score}/100 ({p2.fertility_grade})",
            "plant_health": f"{summary.plant_health_score}/100 ({p3.severity_level})",
            "vigor_index": f"{summary.farm_vigor_index}/100 ({analysis.phase4_fusion.vigor_status})",
        },
        "pathology_diagnosis": {
            "foliage_condition": p3.condition,
            "pathogen_type": p3.pathogen_type,
            "severity_level": p3.severity_level,
            "symptoms_observed": p3.symptoms,
        },
        "soil_diagnosis": {
            "soil_type": p2.soil_type,
            "ph_status": p2.chemical_assessment.ph_status,
            "nitrogen_status": p2.chemical_assessment.nitrogen_status,
            "phosphorus_status": p2.chemical_assessment.phosphorus_status,
            "potassium_status": p2.chemical_assessment.potassium_status,
        },
        "prescribed_treatments": {
            "urgent_protection": [
                {
                    "protocol": "Conventional Chemical",
                    "prescription": p3.chemical_treatment[0] if p3.chemical_treatment else "None required",
                    "application": "Apply with calibrated hydraulic sprayer during morning hours. Re-entry interval: 24h.",
                },
                {
                    "protocol": "Organic Bio-Control",
                    "prescription": p3.organic_treatment[0] if p3.organic_treatment else "None required",
                    "application": "Apply at 10-day intervals as a preventative biological barrier.",
                },
            ] if not p3.is_healthy else [{"protocol": "Preventive Care", "prescription": "Maintain standard field scouting", "application": "Inspect weekly"}],
            "soil_amendments": p2.recommendations,
            "crop_prescription": {
                "primary_candidate": summary.primary_crop,
                "suitability_score": f"{summary.primary_suitability}%",
                "projected_yield": summary.projected_yield,
                "agronomic_guidelines": p5.recommendations[0].cultivation_tips if p5.recommendations else [],
            },
        },
        "calendar_schedule": [
            {"day": "Day 1-2", "task": "Foliar Protection Spray", "action": p3.chemical_treatment[0] if not p3.is_healthy and p3.chemical_treatment else "Inspect foliage undersides"},
            {"day": "Day 5-7", "task": "Soil Conditioning", "action": p2.recommendations[0] if p2.recommendations else "Incorporate organic matter"},
            {"day": "Day 10-14", "task": "Basal Fertilizer Application", "action": f"Apply balanced basal NPK for {summary.primary_crop}"},
            {"day": "Day 15+", "task": "Crop Sowing / Planting", "action": f"Sow certified seeds of {summary.primary_crop}"},
        ],
        "extension_officer_notes": f"Field classified as {summary.farm_status}. Observe strict safety gear protocols when applying agrochemicals.",
    }
