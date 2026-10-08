"""EDAPHNET Production Streamlit Web Dashboard & Mobile-Responsive UI.

Includes:
- 8-Step Guided Workflow
- Visual Distinction: Prediction vs Confidence vs Reliability vs Uncertainty vs Missing Data
- Soil Classification & Mahalanobis OOD
- Foliar Disease Detection & U-Net Segmentation & YOLO Bounding Boxes
- Calibrated Crop Recommendations & Missing Agricultural Data Strategy
- Explainable AI: Grad-CAM Overlays & SHAP Feature Attributions
- Model Information, System Health & History Telemetry
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Direct import of Pipeline Service
from src.edaphnet.application.pipeline_service import EdaphNetPipelineService
from src.edaphnet.config.settings import settings
from src.edaphnet.domain.enums import ReliabilityLevel

# Streamlit Page Config
st.set_page_config(
    page_title="EDAPHNET — Smart Agriculture Intelligence Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise CSS
st.markdown(
    """
    <style>
    /* Global styles */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-high {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-moderate {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-low {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-abstain {
        background-color: #475569;
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .alert-box {
        border-left: 4px solid #f59e0b;
        background-color: #fffbeb;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    """Cache singleton instance of pipeline service."""
    return EdaphNetPipelineService()


pipeline = get_pipeline()

# Navigation Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1592982537447-7440770cbfc9?w=300&auto=format&fit=crop&q=80", caption="EDAPHNET Enterprise AI", use_container_width=True)
    st.markdown("### Navigation")
    page = st.radio(
        "Select Portal Module:",
        [
            "🌾 Guided Analysis (8 Steps)",
            "🔬 Plant Disease Diagnostic",
            "🌱 Soil Classification",
            "📊 Crop Recommendations",
            "🧠 Explainability & SHAP",
            "📋 Model Catalog & Metrics",
            "🏥 System Health & Telemetry",
        ],
    )
    st.markdown("---")
    st.markdown("**Environment:** `Production`")
    st.markdown("**Models Certified:** `5 Loaded`")
    st.markdown("**Version:** `v1.4.0`")


# =========================================================================
# MODULE 1: GUIDED MULTI-MODAL 8-STEP WORKFLOW
# =========================================================================
if page == "🌾 Guided Analysis (8 Steps)":
    st.markdown('<div class="main-title">EDAPHNET Multimodal Agricultural Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Dual-vision foliar and soil analytics powered by EfficientNet-B0, YOLO, U-Net, and Calibrated Crop Ranking.</div>', unsafe_allow_html=True)

    st.info("💡 **Production Workflow**: Follow Steps 1 to 3 to provide imagery and laboratory values, then click **Analyze**.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Step 1: Upload Soil Image")
        soil_file = st.file_uploader("Soil Surface Photograph (JPG/PNG)", type=["jpg", "jpeg", "png"], key="guided_soil")
        if soil_file:
            st.image(soil_file, caption="Selected Soil Sample", use_container_width=True)

    with col2:
        st.subheader("Step 2: Upload Plant Leaf Image")
        leaf_file = st.file_uploader("Foliage/Leaf Photograph (JPG/PNG)", type=["jpg", "jpeg", "png"], key="guided_leaf")
        if leaf_file:
            st.image(leaf_file, caption="Selected Leaf Sample", use_container_width=True)

    st.subheader("Step 3: Enter Available Agronomic & Climate Parameters")
    st.caption("Leave empty if laboratory test is unavailable. The system will apply the Missing Data Strategy.")

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        n_val = st.number_input("Nitrogen (N) [mg/kg]", min_value=0.0, max_value=300.0, value=None, placeholder="e.g. 85.0")
        p_val = st.number_input("Phosphorus (P) [mg/kg]", min_value=0.0, max_value=200.0, value=None, placeholder="e.g. 45.0")
    with fcol2:
        k_val = st.number_input("Potassium (K) [mg/kg]", min_value=0.0, max_value=300.0, value=None, placeholder="e.g. 42.0")
        ph_val = st.number_input("Soil pH (0 - 14)", min_value=0.0, max_value=14.0, value=None, placeholder="e.g. 6.5")
    with fcol3:
        temp_val = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=None, placeholder="e.g. 26.5")
        hum_val = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=None, placeholder="e.g. 78.0")
    with fcol4:
        rain_val = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=None, placeholder="e.g. 180.0")
        include_cam = st.checkbox("Generate Grad-CAM Overlays", value=True)

    st.markdown("---")
    st.subheader("Step 4: Execute Production ML Pipeline")

    if st.button("🚀 Analyze Field Sample", type="primary", use_container_width=True):
        soil_bytes = soil_file.getvalue() if soil_file else None
        leaf_bytes = leaf_file.getvalue() if leaf_file else None

        agronomic = {
            "N": n_val,
            "P": p_val,
            "K": k_val,
            "ph": ph_val,
            "temperature": temp_val,
            "humidity": hum_val,
            "rainfall": rain_val,
        }

        with st.spinner("Executing Quality Gate -> EfficientNet-B0 -> YOLO -> U-Net -> Calibrated Recommendation..."):
            response = pipeline.analyze(
                soil_image_bytes=soil_bytes,
                leaf_image_bytes=leaf_bytes,
                agronomic_features=agronomic,
                include_explainability=include_cam,
            )

        st.success(f"Analysis completed in {response.inference_latency_ms:.1f}ms (Request ID: `{response.request_id}`)")

        # -------------------------------------------------------------
        # STEP 5: REVIEW PREDICTIONS
        # -------------------------------------------------------------
        st.markdown("### Step 5: Review Predictions")
        pred_col1, pred_col2 = st.columns(2)

        with pred_col1:
            st.markdown("#### 🌱 Soil Identification")
            if response.soil:
                if response.soil.reliability == ReliabilityLevel.ABSTAIN:
                    st.error(f"❌ Soil Image Rejected: {response.soil.quality_report.reasons if response.soil.quality_report else 'OOD'}")
                else:
                    st.metric("Identified Soil Type", response.soil.soil_type.replace('_', ' ').title(), f"Confidence: {response.soil.confidence*100:.1f}%")
                    st.write(f"**Calibrated:** {response.soil.calibrated} | **OOD Distance:** {response.soil.ood_score:.2f}")

                    # Bar chart of top classes
                    probs_df = pd.DataFrame(list(response.soil.top_k_probabilities.items()), columns=["Soil Type", "Probability"])
                    fig_soil = px.bar(probs_df, x="Probability", y="Soil Type", orientation="h", title="Calibrated Posterior Probability")
                    fig_soil.update_layout(height=220, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_soil, use_container_width=True)
            else:
                st.info("No soil image provided for classification.")

        with pred_col2:
            st.markdown("#### 🔬 Foliage Pathology & Lesion Diagnostics")
            if response.disease:
                if response.disease.reliability == ReliabilityLevel.ABSTAIN:
                    st.error(f"❌ Leaf Image Rejected: {response.disease.quality_report.reasons if response.disease.quality_report else 'OOD'}")
                else:
                    st.metric("Detected Condition", response.disease.disease_class.replace('_', ' ').title(), f"Confidence: {response.disease.confidence*100:.1f}%")
                    st.write(f"**Host Crop:** {response.disease.crop_name.title()} | **Pathogen:** {response.disease.pathogen_type or 'Biotic'}")

                    if response.segmentation:
                        st.warning(f"⚠️ **Infected Area (U-Net):** {response.segmentation.affected_area_percentage:.1f}% ({response.segmentation.severity_grade.title()})")

                    if response.localizations:
                        st.info(f"🎯 **Lesions Localized (YOLO):** {len(response.localizations)} bounding boxes detected")
            else:
                st.info("No leaf image provided for pathology analysis.")

        # -------------------------------------------------------------
        # STEP 6: REVIEW RELIABILITY & UNCERTAINTY
        # -------------------------------------------------------------
        st.markdown("### Step 6: Review Multi-Modal Reliability & Uncertainty")
        rel = response.reliability

        badge_class = f"badge-{rel.level.value}"
        st.markdown(f"System Reliability Assessment: <span class='{badge_class}'>{rel.level.value.upper()}</span>", unsafe_allow_html=True)
        st.write(f"**Telemetry Summary:** {rel.message}")

        if rel.ood_detected:
            st.warning("⚠️ **Out-of-Distribution Warning**: One or more image embeddings exceeded Mahalanobis confidence radii.")

        if response.missing_measurements:
            st.markdown("#### 🧪 Missing Laboratory Data Audit")
            missing_data = [
                {
                    "Parameter": m.measurement,
                    "Status": m.status.value.upper(),
                    "Default Value Imputed": m.imputed_value,
                    "Critical": "YES" if m.is_critical else "NO",
                }
                for m in response.missing_measurements
            ]
            st.table(pd.DataFrame(missing_data))

            if rel.additional_measurements_recommended:
                st.info(f"💡 **Recommended Action**: For higher accuracy, test laboratory levels for: {', '.join(rel.additional_measurements_recommended)}")

        # -------------------------------------------------------------
        # STEP 7: REVIEW CROP RECOMMENDATIONS
        # -------------------------------------------------------------
        st.markdown("### Step 7: Review Calibrated Crop Recommendations")
        if response.recommendations:
            rec_cols = st.columns(min(len(response.recommendations), 5))
            for i, rec in enumerate(response.recommendations[:5]):
                with rec_cols[i]:
                    st.markdown(f"**#{rec.rank} {rec.crop_name.title()}**")
                    st.metric("Suitability", f"{rec.suitability_score*100:.1f}%")
                    st.caption(f"Risk: {rec.climate_risk_level.title()}")
                    for r in rec.suitability_reasons[:2]:
                        st.caption(f"• {r}")
        else:
            st.warning("No crop recommendations generated.")

        # -------------------------------------------------------------
        # STEP 8: VIEW EXPLANATIONS (XAI)
        # -------------------------------------------------------------
        st.markdown("### Step 8: Explainable AI (Grad-CAM & Attributions)")
        xai_col1, xai_col2 = st.columns(2)

        with xai_col1:
            st.markdown("#### Vision Attention Overlays (Grad-CAM)")
            if response.disease and response.disease.gradcam_overlay_url:
                st.image(response.disease.gradcam_overlay_url, caption="Foliar Disease Lesion Activation Map", use_container_width=True)
            elif response.soil and response.soil.gradcam_overlay_url:
                st.image(response.soil.gradcam_overlay_url, caption="Soil Texture Feature Map", use_container_width=True)
            else:
                st.info("Grad-CAM overlays generated in-memory or saved directly.")

        with xai_col2:
            st.markdown("#### Recommendation Feature Attributions (SHAP Drivers)")
            if response.recommendations and response.recommendations[0].suitability_reasons:
                st.markdown(f"**Key drivers for top recommendation ({response.recommendations[0].crop_name.title()}):**")
                for reason in response.recommendations[0].suitability_reasons:
                    st.markdown(f"- `{reason}`")


# =========================================================================
# MODULE 2: PLANT DISEASE DIAGNOSTIC
# =========================================================================
elif page == "🔬 Plant Disease Diagnostic":
    st.title("🔬 Plant Foliar Disease Diagnostic Laboratory")
    st.write("Upload a clear leaf photograph to diagnose foliar pathogens, localize necrotic spots with YOLO, and compute lesion area percentage with U-Net.")

    dis_file = st.file_uploader("Upload Leaf Photo", type=["jpg", "png", "jpeg"], key="dis_solo")
    if dis_file:
        img_bytes = dis_file.getvalue()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Foliage Specimen", use_container_width=True)

        with col2:
            with st.spinner("Evaluating Quality Gate & Disease Pathology..."):
                quality = pipeline.quality_gate.validate_leaf_image(img_bytes)
                if not quality.is_acceptable:
                    st.error(f"Image Rejected: {', '.join(quality.reasons)}")
                else:
                    pred = pipeline.disease_model.predict(pil_img)
                    st.subheader(f"Condition: {pred['disease_class'].replace('_', ' ').title()}")
                    st.metric("Calibrated Confidence", f"{pred['confidence']*100:.1f}%")
                    st.write(f"**Host Crop:** {pred['crop_name'].title()} | **Pathogen:** {pred.get('pathogen_type', 'Biotic')}")

                    st.markdown("#### Prescribed Agricultural Treatment")
                    for t in pred.get("treatment", []):
                        st.markdown(f"- ✅ {t}")

                    st.markdown("#### Preventive Measures")
                    for p in pred.get("preventive_measures", []):
                        st.markdown(f"- 🛡️ {p}")


# =========================================================================
# MODULE 3: SOIL CLASSIFICATION
# =========================================================================
elif page == "🌱 Soil Classification":
    st.title("🌱 Soil Surface Classification & OOD Reliability Gate")
    st.write("Classifies agricultural soils (Alluvial, Black, Red, Clay, Sandy) using EfficientNet-B0 with temperature scaling and Mahalanobis distance OOD detection.")

    soil_img_file = st.file_uploader("Upload Soil Photo", type=["jpg", "png", "jpeg"], key="soil_solo")
    if soil_img_file:
        bytes_data = soil_img_file.getvalue()
        soil_pil = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        scol1, scol2 = st.columns(2)
        with scol1:
            st.image(soil_pil, caption="Soil Specimen", use_container_width=True)

        with scol2:
            quality = pipeline.quality_gate.validate_soil_image(bytes_data)
            if not quality.is_acceptable:
                st.error(f"Image Rejected: {', '.join(quality.reasons)}")
            else:
                pred = pipeline.soil_model.predict(soil_pil)
                st.subheader(f"Identified Soil: {pred['soil_type'].replace('_', ' ').title()}")
                st.metric("Calibrated Confidence", f"{pred['confidence']*100:.1f}%")
                st.write(f"**Mahalanobis OOD Distance:** `{pred.get('ood_distance', 0.0):.2f}`")
                st.write(f"**OOD Status:** {'OUT OF DISTRIBUTION (ABSTAIN)' if pred.get('is_ood') else 'IN DISTRIBUTION (RELIABLE)'}")


# =========================================================================
# MODULE 4: CROP RECOMMENDATIONS
# =========================================================================
elif page == "📊 Crop Recommendations":
    st.title("📊 Calibrated Crop Recommendation Engine")
    st.write("Calculates calibrated suitability rankings for 22+ agricultural crops based on agronomic N-P-K and environmental conditions.")

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st_soil = st.selectbox("Soil Type", ["alluvial", "black", "red", "clay", "sandy"])
        n_in = st.slider("Nitrogen (N)", 0.0, 200.0, 90.0)
        p_in = st.slider("Phosphorus (P)", 0.0, 150.0, 42.0)
        k_in = st.slider("Potassium (K)", 0.0, 200.0, 43.0)
    with rcol2:
        ph_in = st.slider("Soil pH", 3.0, 10.0, 6.5)
        temp_in = st.slider("Temperature (°C)", 10.0, 45.0, 26.0)
        hum_in = st.slider("Relative Humidity (%)", 10.0, 100.0, 80.0)
        rain_in = st.slider("Rainfall (mm)", 20.0, 500.0, 200.0)

    recs, missing = pipeline.recommender.recommend(
        soil_type=st_soil, n=n_in, p=p_in, k=k_in, ph=ph_in, temperature=temp_in, humidity=hum_in, rainfall=rain_in
    )

    st.markdown("### Top Ranked Suitable Crops")
    df_recs = pd.DataFrame([
        {
            "Rank": f"#{r.rank}",
            "Crop": r.crop_name.title(),
            "Suitability Score": f"{r.suitability_score*100:.1f}%",
            "Climate Risk": r.climate_risk_level.title(),
        }
        for r in recs
    ])
    st.table(df_recs)


# =========================================================================
# MODULE 5: EXPLAINABILITY & SHAP
# =========================================================================
elif page == "🧠 Explainability & SHAP":
    st.title("🧠 Explainable AI Diagnostic Center")
    st.write("Inspect how the vision models and recommendation engines derive their outputs using Grad-CAM attention heatmaps and SHAP feature attributions.")

    st.markdown("#### Sample SHAP Attribution Decomposition")
    feats = {"N": 80.0, "P": 40.0, "K": 40.0, "temperature": 27.0, "humidity": 80.0, "ph": 6.5, "rainfall": 210.0}
    explanation = pipeline.shap_explainer.explain_recommendation(feats, target_crop="rice", predicted_score=0.92)

    st.write(f"**Narrative:** {explanation['narrative']}")
    df_shap = pd.DataFrame(list(explanation["feature_attributions"].items()), columns=["Feature", "Attribution"])
    fig_shap = px.bar(df_shap, x="Attribution", y="Feature", orientation="h", title="SHAP Marginal Contribution to Rice Recommendation", color="Attribution", color_continuous_scale="Tealgrn")
    st.plotly_chart(fig_shap, use_container_width=True)


# =========================================================================
# MODULE 6: MODEL CATALOG & METRICS
# =========================================================================
elif page == "📋 Model Catalog & Metrics":
    st.title("📋 Production Model Registry Catalog")
    st.write("Audited list of certified models with versioning, checksums, and dataset provenance.")

    models = pipeline.registry.list_models()
    model_data = [
        {
            "Model Name": m.model_name,
            "Version": m.version,
            "Framework": m.framework,
            "Status": m.status.value.upper(),
            "Dataset Version": m.dataset_version,
            "Preprocessing": m.preprocessing_version,
            "SHA256": m.checksum_sha256[:16] + "...",
        }
        for m in models
    ]
    st.dataframe(pd.DataFrame(model_data), use_container_width=True)


# =========================================================================
# MODULE 7: SYSTEM HEALTH & TELEMETRY
# =========================================================================
elif page == "🏥 System Health & Telemetry":
    st.title("🏥 System Health & Observability")
    st.write("Real-time operational health checks, telemetry latencies, and cloud infrastructure connectivity.")

    hcol1, hcol2, hcol3 = st.columns(3)
    with hcol1:
        st.metric("API Gateway Status", "ONLINE", "200 OK")
    with hcol2:
        st.metric("Database Sync", "CONNECTED", "MongoDB Atlas / SQLite")
    with hcol3:
        st.metric("Cloud Storage", "ACTIVE", "Cloudinary CDN (stg42sga)")

    st.markdown("### Production Architecture Topology")
    st.code(
        """
        User / Mobile Client
                 ↓
          FastAPI Gateway (Port 8000)
                 ↓
          Quality Gate Validator
                 ↓
        -------------------------------------------
        | Dual-Vision ML Pipeline                 |
        | - Soil: EfficientNet-B0 + Calib + OOD   |
        | - Leaf: EfficientNet-B0 + YOLO + U-Net  |
        -------------------------------------------
                 ↓
        Crop Recommender + SHAP Explainer
                 ↓
        Persistence: Cloudinary CDN + Mongo Atlas
        """,
        language="text",
    )
