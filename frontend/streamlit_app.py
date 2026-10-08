"""Streamlit Interactive Agricultural Intelligence Dashboard (Phase 7).

Features:
- Multi-Modal Visual Analytics: Gauges, Crop Compatibility Radar, Attention Donut, Spatial Farm Heatmap.
- Step-by-Step Interactive Farmer Guidance Studio with 1-Click Field Presets.
- Extension Officer Medical-Grade Agronomic Prescription & Dual-Treatment Protocols.
- Platform Historical Analytics & PDF/HTML/Markdown Report Generation.
"""

import io
import json
import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from frontend.utils.charts import (
    create_health_gauge,
    create_crop_radar_chart,
    create_modality_weights_donut,
    create_stress_scores_bar,
    create_spatial_farm_heatmap,
    create_disease_probabilities_chart,
)
from frontend.utils.presets import FARM_PRESETS, generate_synthetic_field_image

def render_plant_disease_result_card(p_res: dict):
    """Render comprehensive diagnostic report for leaf foliage disease."""
    is_healthy = p_res.get("is_healthy", False)
    cond = p_res.get("condition", "Unknown")
    pathogen = p_res.get("pathogen_type", "None")
    severity = p_res.get("severity_level", "None")
    score = p_res.get("plant_health_score", 0)
    conf = p_res.get("confidence", 0.0) * 100
    symptoms = p_res.get("symptoms", "")
    organic = p_res.get("organic_treatment", [])
    chemical = p_res.get("chemical_treatment", [])
    preventive = p_res.get("preventive_measures", [])
    probs = p_res.get("probabilities", {})

    status_color = "#10b981" if is_healthy else ("#f59e0b" if severity in ["Low", "Moderate"] else "#ef4444")
    badge_text = "HEALTHY FOLIAGE" if is_healthy else f"PATHOLOGY DETECTED: {cond.upper()}"

    st.markdown(
        f"""
        <div style="background: white; border: 2px solid {status_color}; border-radius: 12px; padding: 20px; margin-top: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.06);">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f3f4f6; padding-bottom: 12px; margin-bottom: 15px;">
                <div>
                    <span style="background: {status_color}; color: white; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.05em;">{badge_text}</span>
                    <h2 style="margin: 8px 0 0 0; color: #111827; font-size: 1.6rem; font-weight: 800;">{cond}</h2>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.85rem; color: #6b7280;">Request ID: <b>{p_res.get('request_id', 'N/A')}</b></span><br/>
                    <span style="font-size: 1.1rem; font-weight: 700; color: {status_color};">Confidence: {conf:.1f}%</span>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Foliage Health Score", f"{score}/100")
    with c2:
        st.metric("Pathogen Type", pathogen)
    with c3:
        st.metric("Infection Severity", severity)
    with c4:
        st.metric("Model Confidence", f"{conf:.1f}%")

    st.markdown("---")
    st.markdown(f"**🔍 Observed Visual Pathology Symptoms:** *{symptoms}*")

    st.markdown("#### 🔬 Neural Probability Distribution Across 7 Foliage Classes")
    if probs:
        p_fig = create_disease_probabilities_chart(probs)
        st.plotly_chart(p_fig, use_container_width=True)

    st.markdown("#### 💊 Integrated Pest & Disease Management (IPM) Prescriptions")
    tab_org, tab_chem, tab_prev = st.tabs(["🌿 Organic Bio-Control", "🧪 Conventional Chemical", "🛡️ Cultural Preventive Measures"])

    with tab_org:
        for idx, item in enumerate(organic, 1):
            st.markdown(f"**{idx}.** {item}")

    with tab_chem:
        for idx, item in enumerate(chemical, 1):
            st.markdown(f"**{idx}.** {item}")

    with tab_prev:
        for idx, item in enumerate(preventive, 1):
            st.markdown(f"**{idx}.** {item}")

    st.markdown("</div>", unsafe_allow_html=True)


# Backend Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Page Configuration
st.set_page_config(
    page_title="KisanSeva - Agricultural Intelligence Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System & CSS Tokens
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #064e3b 0%, #047857 50%, #059669 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(6, 78, 59, 0.2);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #a7f3d0;
        margin-top: 6px;
        font-weight: 500;
    }
    .phase-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #ffffff;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-top: 12px;
    }
    .rx-card {
        background: #ffffff;
        border: 2px solid #059669;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .rx-header {
        border-bottom: 2px dashed #059669;
        padding-bottom: 12px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .crop-pill {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .alert-critical {
        background: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .alert-warning {
        background: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .alert-info {
        background: #f0fdf4;
        border-left: 5px solid #10b981;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation & Backend Probe
with st.sidebar:
    st.markdown("### 🌾 KisanSeva Intelligence Hub")
    
    # Live Backend Health Probe
    backend_online = False
    try:
        h_resp = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if h_resp.status_code == 200:
            st.success("🟢 API Engine: Online (v6.0.0)")
            backend_online = True
        else:
            st.warning("🟡 API Engine: Degraded")
    except Exception:
        st.error("🔴 API Engine: Offline (Run FastAPI on :8000)")

    st.markdown("---")
    st.markdown("### 🧭 Navigation & Studio Views")
    active_view = st.radio(
        "Select Dashboard Module:",
        [
            "🌿 Plant Foliage Disease Doctor",
            "🌾 Precision Crop Recommendation Advisor",
            "🚀 1-Click Multi-Modal Farm Studio",
            "🩺 Agronomic Rx & Treatment Pad",
            "📊 Multi-Modal Visual Analytics",
            "📜 Scan History & Export Center",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🏗️ Architecture Status")
    st.markdown(
        """
        - **Phase 1: Acquisition** ✅
        - **Phase 2: Soil Health** ✅
        - **Phase 3: Foliage Disease** ✅
        - **Phase 4: Multi-Modal Fusion** ✅
        - **Phase 5: Crop Recommendation** ✅
        - **Phase 6: Unified Pipeline API** ✅
        - **Phase 7: Frontend & Dashboard** ✅
        - **Phase 8: Testing & Deployment** 🟢 *(Production v8.0.0)*
        """
    )
    st.caption("KisanSeva Agricultural Decision Platform v8.0.0")

# Hero Banner
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">🌱 KisanSeva: Smart Agriculture System</h1>
        <div class="hero-subtitle">Multi-Modal Diagnostic Intelligence & Precision Agronomic Decision Dashboard</div>
        <span class="phase-pill">✨ Phase 8: Production Ready & Multi-Modal Architecture</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize Session State for Analysis Persistence
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None
if "current_prescription" not in st.session_state:
    st.session_state.current_prescription = None
if "plant_only_result" not in st.session_state:
    st.session_state.plant_only_result = None
if "live_env_data" not in st.session_state:
    st.session_state.live_env_data = None



# =========================================================================
# MODULE 1: 1-CLICK MULTI-MODAL FARM STUDIO
# =========================================================================
if active_view == "🚀 1-Click Multi-Modal Farm Studio":
    st.markdown("## 🚀 Field Diagnostic Studio")
    st.markdown("Provide farm visual evidence and soil readings, or select a pre-calibrated scenario to run instant multi-modal analysis.")

    # Preset Loader Bar
    st.markdown("#### ⚡ Quick Preset Scenarios")
    preset_choice = st.selectbox(
        "Choose an agricultural preset scenario to auto-populate evidence:",
        ["None (Custom Upload)"] + list(FARM_PRESETS.keys())
    )

    active_preset = FARM_PRESETS.get(preset_choice) if preset_choice != "None (Custom Upload)" else None
    if active_preset:
        st.info(f"💡 **Scenario Context**: {active_preset['description']}")

    # Form Columns
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.markdown("### 🟤 1. Topsoil Visual Evidence")
        if active_preset:
            st.caption(f"Preset Soil Type: **{active_preset['soil_type']}**")
            synth_soil = generate_synthetic_field_image("soil", active_preset["soil_type"])
            st.image(synth_soil, caption=f"Preset Evidence: {active_preset['soil_type']}", use_container_width=True)
            soil_bytes = synth_soil.getvalue()
            soil_fname = f"preset_{active_preset['soil_type'].lower().replace(' ', '_')}.jpg"
        else:
            soil_uploader = st.file_uploader(
                "Upload field soil photograph",
                type=["jpg", "jpeg", "png"],
                key="studio_soil"
            )
            soil_bytes = soil_uploader.getvalue() if soil_uploader else None
            soil_fname = soil_uploader.name if soil_uploader else None
            if soil_uploader:
                st.image(soil_uploader, caption="Uploaded Soil", use_container_width=True)

    with col_img2:
        st.markdown("### 🌿 2. Crop Foliage Evidence")
        if active_preset:
            st.caption(f"Preset Condition: **{active_preset['plant_condition']}**")
            synth_plant = generate_synthetic_field_image("plant", active_preset["plant_condition"])
            st.image(synth_plant, caption=f"Preset Foliage: {active_preset['plant_condition']}", use_container_width=True)
            plant_bytes = synth_plant.getvalue()
            plant_fname = f"preset_{active_preset['plant_condition'].lower().replace(' ', '_')}.jpg"
        else:
            plant_uploader = st.file_uploader(
                "Upload plant leaf photograph",
                type=["jpg", "jpeg", "png"],
                key="studio_plant"
            )
            plant_bytes = plant_uploader.getvalue() if plant_uploader else None
            plant_fname = plant_uploader.name if plant_uploader else None
            if plant_uploader:
                st.image(plant_uploader, caption="Uploaded Foliage", use_container_width=True)

        if plant_bytes:
            if st.button("🔬 Detect Disease from This Leaf Now", key="quick_disease_btn", use_container_width=True):
                with st.spinner("🔬 Diagnosing plant foliage disease with MobileNetV3-Large..."):
                    try:
                        files = {"plant_image": (plant_fname or "plant.jpg", plant_bytes, "image/jpeg")}
                        resp = requests.post(f"{BACKEND_URL}/api/plant/predict", files=files, timeout=20)
                        if resp.status_code == 200:
                            st.session_state.plant_only_result = resp.json()
                            st.success(f"✅ Diagnosed: **{st.session_state.plant_only_result['condition']}** ({st.session_state.plant_only_result['pathogen_type']})")
                        else:
                            st.error(f"❌ Diagnosis failed ({resp.status_code}): {resp.text}")
                    except Exception as ex:
                        st.error(f"❌ Connection error: {str(ex)}")

    st.markdown("---")
    st.markdown("### 🧪 3. Soil Nutrients & Meteorological Readings")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        init_ph = float(active_preset["ph"]) if active_preset else 6.5
        val_ph = st.number_input("Soil pH (0-14)", min_value=0.0, max_value=14.0, value=init_ph, step=0.1)
        init_temp = float(active_preset["temperature"]) if active_preset else 26.0
        val_temp = st.number_input("Ambient Temp (°C)", min_value=-10.0, max_value=60.0, value=init_temp, step=0.5)

    with col_c2:
        init_n = float(active_preset["nitrogen"]) if active_preset else 140.0
        val_n = st.number_input("Nitrogen - N (kg/ha)", min_value=0.0, value=init_n, step=5.0)
        init_hum = float(active_preset["humidity"]) if active_preset else 70.0
        val_hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=init_hum, step=1.0)

    with col_c3:
        init_p = float(active_preset["phosphorus"]) if active_preset else 35.0
        val_p = st.number_input("Phosphorus - P (kg/ha)", min_value=0.0, value=init_p, step=2.0)
        init_rain = float(active_preset["rainfall"]) if active_preset else 120.0
        val_rain = st.number_input("Rainfall (mm)", min_value=0.0, value=init_rain, step=5.0)

    with col_c4:
        init_k = float(active_preset["potassium"]) if active_preset else 150.0
        val_k = st.number_input("Potassium - K (kg/ha)", min_value=0.0, value=init_k, step=5.0)
        st.markdown("<br>", unsafe_allow_html=True)
        force_refresh = st.checkbox("Bypass Cache (Force ML Re-run)", value=False)

    st.markdown("<br>", unsafe_allow_html=True)
    col_act1, col_act2 = st.columns([1, 1])
    with col_act1:
        disease_only_btn = st.button("🌿 Detect Foliage Disease Only", use_container_width=True)
    with col_act2:
        run_btn = st.button("🚀 Execute Multi-Modal AI Diagnosis", type="primary", use_container_width=True)

    if disease_only_btn:
        if not plant_bytes:
            st.warning("⚠️ Please upload a plant leaf photograph first under '2. Crop Foliage Evidence'.")
        else:
            with st.spinner("🔬 Diagnosing plant disease with MobileNetV3-Large..."):
                try:
                    files = {"plant_image": (plant_fname or "plant.jpg", plant_bytes, "image/jpeg")}
                    resp = requests.post(f"{BACKEND_URL}/api/plant/predict", files=files, timeout=20)
                    if resp.status_code == 200:
                        st.session_state.plant_only_result = resp.json()
                        st.success(f"✅ Foliage Disease Diagnosis Completed! ({st.session_state.plant_only_result['condition']})")
                    else:
                        st.error(f"❌ Disease diagnosis failed ({resp.status_code}): {resp.text}")
                except Exception as ex:
                    st.error(f"❌ Connection error: {str(ex)}")

    if run_btn:
        if plant_bytes and not soil_bytes:
            st.info("ℹ️ Topsoil photo not provided for full farm fusion. Running standalone Foliage Disease Detection on your uploaded leaf...")
            with st.spinner("🔬 Diagnosing plant foliage disease with MobileNetV3-Large..."):
                try:
                    files = {"plant_image": (plant_fname or "plant.jpg", plant_bytes, "image/jpeg")}
                    resp = requests.post(f"{BACKEND_URL}/api/plant/predict", files=files, timeout=20)
                    if resp.status_code == 200:
                        st.session_state.plant_only_result = resp.json()
                        st.success(f"✅ Foliage Disease Diagnosis Completed! ({st.session_state.plant_only_result['condition']})")
                    else:
                        st.error(f"❌ Disease diagnosis failed ({resp.status_code}): {resp.text}")
                except Exception as ex:
                    st.error(f"❌ Connection error: {str(ex)}")
        elif not soil_bytes and not plant_bytes:
            st.error("❌ Please upload at least a plant leaf photograph or select an agricultural preset scenario above.")
        elif soil_bytes and not plant_bytes:
            st.warning("⚠️ Crop foliage photo is missing. Please upload a leaf photo to diagnose disease, or select a preset scenario.")
        else:
            with st.spinner("⚡ Running Gated Neural Multi-Modal Pipeline..."):
                try:
                    files = {
                        "soil_image": (soil_fname or "soil.jpg", soil_bytes, "image/jpeg"),
                        "plant_image": (plant_fname or "plant.jpg", plant_bytes, "image/jpeg"),
                    }
                    data = {
                        "ph": val_ph,
                        "nitrogen": val_n,
                        "phosphorus": val_p,
                        "potassium": val_k,
                        "temperature": val_temp,
                        "humidity": val_hum,
                        "rainfall": val_rain,
                        "force_refresh": force_refresh,
                    }
                    resp = requests.post(f"{BACKEND_URL}/api/pipeline/analyze-all", files=files, data=data, timeout=30)
                    if resp.status_code == 200:
                        analysis_res = resp.json()
                        st.session_state.current_analysis = analysis_res

                        # Fetch prescription
                        req_id = analysis_res["request_id"]
                        rx_resp = requests.get(f"{BACKEND_URL}/api/pipeline/prescription/{req_id}", timeout=5)
                        if rx_resp.status_code == 200:
                            st.session_state.current_prescription = rx_resp.json()

                        st.success(f"✅ Diagnostic Completed! Generated Request ID: `{req_id}`")
                    else:
                        st.error(f"❌ Analysis failed ({resp.status_code}): {resp.text}")
                except Exception as exc:
                    st.error(f"❌ Communication Error: {str(exc)}")

    # Show Standalone Plant Disease Report if generated
    if st.session_state.plant_only_result:
        st.markdown("---")
        st.markdown("## 🔬 Plant Foliage Disease Diagnosis Report")
        render_plant_disease_result_card(st.session_state.plant_only_result)


    # Show Diagnostic Summary if Available
    if st.session_state.current_analysis:
        a = st.session_state.current_analysis
        summary = a["summary"]
        p2 = a["phase2_soil"]
        p3 = a["phase3_plant"]
        p4 = a["phase4_fusion"]
        p5 = a["phase5_crop"]

        st.markdown("---")
        st.markdown("## 📈 Live Diagnostic Scorecards")

        # 4 Sleek Interactive Gauges
        g_c1, g_c2, g_c3, g_c4 = st.columns(4)
        with g_c1:
            fig1 = create_health_gauge(summary["farm_health_score"], "Farm Health Score", summary["farm_status"])
            st.plotly_chart(fig1, use_container_width=True)
        with g_c2:
            fig2 = create_health_gauge(summary["soil_health_score"], "Soil Fertility", p2["soil_type"])
            st.plotly_chart(fig2, use_container_width=True)
        with g_c3:
            fig3 = create_health_gauge(summary["plant_health_score"], "Foliage Health", p3["condition"])
            st.plotly_chart(fig3, use_container_width=True)
        with g_c4:
            fig4 = create_health_gauge(summary["farm_vigor_index"], "Farm Vigor Index", p4["vigor_status"])
            st.plotly_chart(fig4, use_container_width=True)

        # Critical Alerts Section
        st.markdown("### ⚠️ Active Hazard & Stress Alerts")
        if summary["critical_alerts"]:
            for alert in summary["critical_alerts"]:
                a_class = "alert-critical" if alert["level"] == "CRITICAL" else ("alert-warning" if alert["level"] == "WARNING" else "alert-info")
                st.markdown(
                    f"""
                    <div class="{a_class}">
                        <strong style="font-size: 1.05rem;">[{alert['level']}] {alert['title']}</strong><br/>
                        <span style="color: #374151;">{alert['message']}</span><br/>
                        <span style="color: #1e3a8a; font-weight: 600; font-size: 0.9rem;">🛠️ Prescribed Action: {alert['recommended_action']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("🟢 No critical hazards detected. Field visual and chemical indicators are balanced.")

        # Top 3 Recommended Crops
        st.markdown("### 🌾 Precision Crop Recommendations")
        rec_cols = st.columns(3)
        for i, crop in enumerate(p5["recommendations"][:3]):
            with rec_cols[i]:
                st.markdown(
                    f"""
                    <div class="crop-pill">
                        <div style="font-size: 1.25rem; font-weight: 800; color: #065f46;">#{crop['rank']} {crop['crop_name']}</div>
                        <div style="font-size: 0.9rem; font-weight: 700; color: #059669; margin: 4px 0;">{crop['suitability_score']:.1f}% Match ({crop['compatibility_grade']})</div>
                        <div style="font-size: 0.85rem; color: #4b5563;"><b>Projected Yield:</b> {crop['projected_yield']}</div>
                        <div style="font-size: 0.85rem; color: #4b5563;"><b>Water Demand:</b> {crop['water_requirement']}</div>
                        <div style="font-size: 0.85rem; color: #4b5563;"><b>Cycle:</b> {crop['growing_duration_days']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =========================================================================
# MODULE: STANDALONE PLANT FOLIAGE DISEASE DIAGNOSTIC LAB
# =========================================================================
elif active_view == "🌿 Plant Foliage Disease Doctor":
    st.markdown("## 🌿 Plant Foliage Disease Diagnostic Doctor")
    st.markdown("Upload any crop leaf photograph to detect disease pathology, identify pathogens, evaluate infection severity, and get immediate agronomic treatment protocols.")

    col_l1, col_l2 = st.columns([1, 1])

    with col_l1:
        st.markdown("### 📸 1. Leaf Photograph Input")
        disease_preset = st.selectbox(
            "Test with an authentic disease sample or upload your own:",
            [
                "Upload Your Own Leaf Photo",
                "Tomato Early Blight (Alternaria solani)",
                "Potato Late Blight (Phytophthora infestans)",
                "Leaf Rust (Puccinia spp.)",
                "Powdery Mildew (Erysiphe spp.)",
                "Bacterial Spot (Xanthomonas spp.)",
                "Yellow Leaf Curl Virus",
                "Healthy Leaf Foliage",
            ]
        )

        lab_plant_bytes = None
        lab_plant_fname = "leaf.jpg"

        if disease_preset != "Upload Your Own Leaf Photo":
            # Map selection to condition keyword
            cond_map = {
                "Tomato Early Blight (Alternaria solani)": "Early Blight",
                "Potato Late Blight (Phytophthora infestans)": "Late Blight",
                "Leaf Rust (Puccinia spp.)": "Leaf Rust",
                "Powdery Mildew (Erysiphe spp.)": "Powdery Mildew",
                "Bacterial Spot (Xanthomonas spp.)": "Bacterial Spot",
                "Yellow Leaf Curl Virus": "Yellow Leaf Curl",
                "Healthy Leaf Foliage": "Healthy Foliage",
            }
            sample_cond = cond_map.get(disease_preset, "Early Blight")
            synth_leaf = generate_synthetic_field_image("plant", sample_cond)
            st.image(synth_leaf, caption=f"Sample Leaf: {disease_preset}", use_container_width=True)
            lab_plant_bytes = synth_leaf.getvalue()
            lab_plant_fname = f"sample_{sample_cond.lower().replace(' ', '_')}.jpg"
        else:
            lab_uploader = st.file_uploader(
                "Upload plant leaf image (JPG, PNG)",
                type=["jpg", "jpeg", "png"],
                key="lab_leaf_uploader"
            )
            if lab_uploader:
                lab_plant_bytes = lab_uploader.getvalue()
                lab_plant_fname = lab_uploader.name
                st.image(lab_uploader, caption="Uploaded Leaf Photograph", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_lab_btn = st.button("🔬 Diagnose Leaf Disease Now", type="primary", use_container_width=True)

    with col_l2:
        st.markdown("### 📋 2. Supported Pathologies & Classes")
        st.markdown(
            """
            Our **MobileNetV3-Large** neural classifier evaluates crop foliage against 7 diagnostic classes:
            - **Healthy Foliage**: Optimal cuticle integrity, vibrant lamina.
            - **Early Blight** (*Alternaria solani*): Fungal necrotic bullseye lesions.
            - **Late Blight** (*Phytophthora infestans*): Oomycete rapidly spreading water-soaked lesions.
            - **Powdery Mildew** (*Erysiphe spp.*): Talcum-like white superficial fungal growth.
            - **Leaf Rust** (*Puccinia spp.*): Raised orange/brown powdery pustules.
            - **Bacterial Spot** (*Xanthomonas spp.*): Water-soaked dark angular spots with chlorotic halos.
            - **Yellow Leaf Curl** (*Begomovirus*): Upward cupping, chlorosis, and stunting.
            """
        )

    if run_lab_btn:
        if not lab_plant_bytes:
            st.error("❌ Please upload a leaf photo or pick a sample from the dropdown first.")
        else:
            with st.spinner("🔬 Running MobileNetV3-Large neural disease inference..."):
                try:
                    files = {"plant_image": (lab_plant_fname, lab_plant_bytes, "image/jpeg")}
                    resp = requests.post(f"{BACKEND_URL}/api/plant/predict", files=files, timeout=20)
                    if resp.status_code == 200:
                        st.session_state.plant_only_result = resp.json()
                        st.success(f"✅ Diagnosis Completed: **{st.session_state.plant_only_result['condition']}**")
                    else:
                        st.error(f"❌ Diagnosis failed ({resp.status_code}): {resp.text}")
                except Exception as ex:
                    st.error(f"❌ Connection error: {str(ex)}")

    if st.session_state.plant_only_result:
        st.markdown("---")
        st.markdown("## 🔬 Certified Foliage Diagnostic Report")
        render_plant_disease_result_card(st.session_state.plant_only_result)


# =========================================================================
# MODULE 2: MULTI-MODAL VISUAL ANALYTICS
# =========================================================================
elif active_view == "📊 Multi-Modal Visual Analytics":
    st.markdown("## 📊 Multi-Modal Visual Analytics")
    st.markdown("Deep-dive into cross-domain feature interactions, neural attention gating weights, and spatial field vigor patterns.")

    if not st.session_state.current_analysis:
        st.info("ℹ️ No active diagnostic loaded. Run a diagnosis in the 'Diagnostic Studio' first or select a preset to populate analytics.")
    else:
        a = st.session_state.current_analysis
        summary = a["summary"]
        p4 = a["phase4_fusion"]
        p5 = a["phase5_crop"]
        p1 = a["phase1_input"]

        row1_c1, row1_c2 = st.columns(2)
        with row1_c1:
            st.markdown("### 🕸️ Crop Growth Envelope Radar")
            st.caption("Current Field Parameters vs Recommended Crop Optimal Envelope")
            primary_crop_name = summary["primary_crop"]
            # Look up crop profile from catalog
            try:
                cat_resp = requests.get(f"{BACKEND_URL}/api/crop/catalog", timeout=3)
                crop_cat = cat_resp.json().get("catalog", {})
                target_profile = crop_cat.get(primary_crop_name, {})
                radar_fig = create_crop_radar_chart(p1, target_profile)
                st.plotly_chart(radar_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render radar: {str(e)}")

        with row1_c2:
            st.markdown("### 🎯 Phase 4 Attention Gating Distribution")
            st.caption("Learned Domain Modality Weights (Softmax)")
            donut_fig = create_modality_weights_donut(p4["modality_weights"])
            st.plotly_chart(donut_fig, use_container_width=True)

        st.markdown("---")
        row2_c1, row2_c2 = st.columns(2)

        with row2_c1:
            st.markdown("### ⚡ Multi-Modal Domain Stress Indicators")
            st.caption("Normalized Stress & Hazard Factors (0 - 100%)")
            stress_fig = create_stress_scores_bar(p4["stress_scores"])
            st.plotly_chart(stress_fig, use_container_width=True)

        with row2_c2:
            st.markdown("### 🗺️ Spatial Plot Vigor Heatmap (5-Ha Plot)")
            st.caption("Simulated Zone-by-Zone Multi-Sensor Health")
            heat_fig = create_spatial_farm_heatmap(summary["farm_vigor_index"], summary["plant_condition"])
            st.plotly_chart(heat_fig, use_container_width=True)


# =========================================================================
# MODULE 3: AGRONOMIC RX & TREATMENT PAD
# =========================================================================
elif active_view == "🩺 Agronomic Rx & Treatment Pad":
    st.markdown("## 🩺 Agronomic Prescription & Extension Consultation")
    st.markdown("Medical-grade field treatment prescription with dual protocols (Conventional Chemical vs Organic Bio-Control) and calendar schedules.")

    if not st.session_state.current_analysis:
        st.info("ℹ️ No farm scan loaded. Please execute a scan in 'Diagnostic Studio' to generate a prescription.")
    else:
        rx = st.session_state.current_prescription
        a = st.session_state.current_analysis

        if not rx:
            try:
                rx_resp = requests.get(f"{BACKEND_URL}/api/pipeline/prescription/{a['request_id']}", timeout=5)
                if rx_resp.status_code == 200:
                    rx = rx_resp.json()
                    st.session_state.current_prescription = rx
            except Exception:
                pass

        if rx:
            st.markdown(
                f"""
                <div class="rx-card">
                    <div class="rx-header">
                        <div>
                            <span style="font-size: 1.6rem; font-weight: 900; color: #065f46;">℞ KISANSEVA PRESCRIPTION</span><br/>
                            <span style="color: #6b7280; font-size: 0.85rem;">Prescription ID: <b>{rx['prescription_id']}</b> | Issued: {rx['date_issued']}</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="background: #059669; color: white; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem;">CERTIFIED AGRONOMIC RX</span>
                        </div>
                    </div>
                """,
                unsafe_allow_html=True
            )

            # Vital Signs
            v_c1, v_c2, v_c3, v_c4 = st.columns(4)
            with v_c1:
                st.metric("Farm Health Status", rx["vital_signs"]["overall_health_score"])
            with v_c2:
                st.metric("Soil Status", rx["vital_signs"]["soil_fertility"])
            with v_c3:
                st.metric("Foliage Pathogen", rx["vital_signs"]["plant_health"])
            with v_c4:
                st.metric("Farm Vigor FVI", rx["vital_signs"]["vigor_index"])

            st.markdown("---")
            st.markdown("### 🔬 Pathological & Soil Diagnosis")
            col_d_p, col_d_s = st.columns(2)
            with col_d_p:
                st.markdown(f"**Condition Diagnosed:** `{rx['pathology_diagnosis']['foliage_condition']}`")
                st.markdown(f"**Pathogen Type:** `{rx['pathology_diagnosis']['pathogen_type']}` (Severity: `{rx['pathology_diagnosis']['severity_level']}`)")
                st.markdown(f"**Visual Symptoms:** *{rx['pathology_diagnosis']['symptoms_observed']}*")

            with col_d_s:
                st.markdown(f"**Soil Classification:** `{rx['soil_diagnosis']['soil_type']}`")
                st.markdown(f"**pH Status:** `{rx['soil_diagnosis']['ph_status']}`")
                st.markdown(f"**NPK Balance:** N: `{rx['soil_diagnosis']['nitrogen_status']}`, P: `{rx['soil_diagnosis']['phosphorus_status']}`, K: `{rx['soil_diagnosis']['potassium_status']}`")

            st.markdown("---")
            st.markdown("### 💊 Prescribed Treatment Protocols")
            protocol_mode = st.radio("Toggle Treatment Philosophy:", ["Both Protocols (Integrated Pest Management)", "Conventional Chemical Only", "Organic Bio-Control Only"], horizontal=True)

            treatments = rx["prescribed_treatments"]["urgent_protection"]
            for treat in treatments:
                if "Chemical" in protocol_mode and "Organic" in treat["protocol"]:
                    continue
                if "Organic" in protocol_mode and "Chemical" in treat["protocol"]:
                    continue

                badge_col = "#dc2626" if "Chemical" in treat["protocol"] else "#059669"
                st.markdown(
                    f"""
                    <div style="background: #f9fafb; border-left: 4px solid {badge_col}; padding: 12px; margin-bottom: 8px; border-radius: 4px;">
                        <span style="color: {badge_col}; font-weight: 700; font-size: 0.85rem;">[{treat['protocol'].upper()}]</span>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #111827; margin: 4px 0;">{treat['prescription']}</div>
                        <small style="color: #4b5563;"><b>Application Directive:</b> {treat['application']}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.markdown("### 📅 Farm Management Intervention Calendar")
            cal_df = pd.DataFrame(rx["calendar_schedule"])
            st.table(cal_df)

            st.caption(f"**Agronomist Advisory Note:** {rx['extension_officer_notes']}")
            st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# MODULE 4: PRECISION CROP RECOMMENDATION ADVISOR
# =========================================================================
elif active_view == "🌾 Precision Crop Recommendation Advisor":
    st.markdown("## 🌾 Precision Crop Recommendation Advisor")
    st.markdown("Evaluate soil chemical nutrients (N, P, K, pH) and micro-meteorological variables (Temperature, Humidity, Rainfall) against agronomic envelopes to recommend the highest-yielding, most sustainable crops.")

    # 1. Preset Scenarios
    st.markdown("#### ⚡ 1-Click Regional Agro-Climatic Presets")
    crop_preset_options = {
        "Custom Farm Input": None,
        "🌾 Indo-Gangetic Alluvial Belt (Rice / Wheat / Sugarcane)": {
            "ph": 6.8, "n": 160.0, "p": 50.0, "k": 140.0, "temp": 24.0, "hum": 65.0, "rain": 110.0, "soil": "Alluvial Soil"
        },
        "🌸 Deccan Plateau Black Soil (Cotton / Maize / Pulses)": {
            "ph": 7.5, "n": 120.0, "p": 40.0, "k": 100.0, "temp": 30.0, "hum": 55.0, "rain": 85.0, "soil": "Black Soil"
        },
        "🥜 Semi-Arid Sandy Tracts (Groundnut / Moth Beans / Chickpea)": {
            "ph": 6.6, "n": 60.0, "p": 45.0, "k": 70.0, "temp": 32.0, "hum": 45.0, "rain": 45.0, "soil": "Sandy Soil"
        },
        "🍃 Subtropical Hill Slope (Tea / Coffee / Kidney Beans)": {
            "ph": 5.2, "n": 110.0, "p": 35.0, "k": 100.0, "temp": 20.0, "hum": 85.0, "rain": 220.0, "soil": "Red Soil"
        },
    }

    selected_crop_preset = st.selectbox("Choose a regional preset or input custom field readings:", list(crop_preset_options.keys()))
    active_crop_data = crop_preset_options.get(selected_crop_preset)

    st.markdown("---")
    col_inp1, col_inp2 = st.columns(2)

    with col_inp1:
        st.markdown("### 🧪 Soil Chemistry & Fertility")
        c_ph = st.slider("Soil pH Level:", 0.0, 14.0, float(active_crop_data["ph"] if active_crop_data else 6.5), 0.1)
        c_n = st.slider("Nitrogen (N) [kg/ha]:", 0.0, 300.0, float(active_crop_data["n"] if active_crop_data else 140.0), 5.0)
        c_p = st.slider("Phosphorus (P) [kg/ha]:", 0.0, 150.0, float(active_crop_data["p"] if active_crop_data else 35.0), 2.0)
        c_k = st.slider("Potassium (K) [kg/ha]:", 0.0, 350.0, float(active_crop_data["k"] if active_crop_data else 150.0), 5.0)
        c_soil = st.selectbox(
            "Primary Soil Classification:",
            ["Alluvial Soil", "Black Soil", "Clay Soil", "Red Soil", "Sandy Soil", "Loamy Soil"],
            index=["Alluvial Soil", "Black Soil", "Clay Soil", "Red Soil", "Sandy Soil", "Loamy Soil"].index(active_crop_data["soil"]) if active_crop_data and active_crop_data["soil"] in ["Alluvial Soil", "Black Soil", "Clay Soil", "Red Soil", "Sandy Soil", "Loamy Soil"] else 5
        )

    with col_inp2:
        st.markdown("### 🌤️ Weather & Climate Parameters")
        c_temp = st.slider("Average Ambient Temperature (°C):", -5.0, 55.0, float(active_crop_data["temp"] if active_crop_data else 26.0), 0.5)
        c_hum = st.slider("Relative Atmospheric Humidity (%):", 10.0, 100.0, float(active_crop_data["hum"] if active_crop_data else 65.0), 1.0)
        c_rain = st.slider("Annual / Seasonal Rainfall (mm):", 0.0, 500.0, float(active_crop_data["rain"] if active_crop_data else 120.0), 5.0)

        st.markdown("<br>", unsafe_allow_html=True)
        run_rec_btn = st.button("🌾 Predict Optimal Crops Now", type="primary", use_container_width=True)

    if run_rec_btn:
        with st.spinner("🌾 Running precision neural & agronomic crop suitability analysis..."):
            try:
                payload = {
                    "ph": c_ph,
                    "nitrogen": c_n,
                    "phosphorus": c_p,
                    "potassium": c_k,
                    "temperature": c_temp,
                    "humidity": c_hum,
                    "rainfall": c_rain,
                    "soil_type": c_soil,
                }
                resp = requests.post(f"{BACKEND_URL}/api/crop/predict-direct", json=payload, timeout=10)
                if resp.status_code == 200:
                    st.session_state.crop_recommendation_result = resp.json()
                    st.success(f"✅ Recommendation Completed: Top Crop **{st.session_state.crop_recommendation_result['primary_crop']}** ({st.session_state.crop_recommendation_result['primary_suitability']}% Match)")
                else:
                    st.error(f"❌ Recommendation failed: {resp.text}")
            except Exception as ex:
                st.error(f"❌ API connection error: {str(ex)}")

    if "crop_recommendation_result" in st.session_state and st.session_state.crop_recommendation_result:
        rec_data = st.session_state.crop_recommendation_result
        top_crop = rec_data["primary_crop"]
        top_score = rec_data["primary_suitability"]
        recs = rec_data["recommendations"]

        st.markdown("---")
        st.markdown("## 🥇 Top Recommended Crop Match")
        
        # Primary Crop Hero Banner
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #064e3b 0%, #059669 100%); color: white; border-radius: 14px; padding: 24px 30px; box-shadow: 0 10px 25px -5px rgba(5, 150, 105, 0.3); margin-bottom: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="background: rgba(255, 255, 255, 0.2); padding: 4px 14px; border-radius: 9999px; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.05em;">#1 HIGHEST SUITABILITY</span>
                        <h1 style="margin: 8px 0 0 0; font-size: 2.4rem; color: #ffffff; font-weight: 900;">{top_crop}</h1>
                        <p style="margin: 6px 0 0 0; color: #a7f3d0; font-size: 1.05rem;">Projected Yield: <b>{recs[0]['projected_yield']}</b> | Cycle: <b>{recs[0]['growing_duration_days']}</b></p>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 3rem; font-weight: 900; color: #ffffff;">{top_score:.1f}%</span><br/>
                        <span style="background: #10b981; color: white; padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">{recs[0]['compatibility_grade']}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Ranked Crop Grid
        st.markdown("### 📋 Ranked Commercial Crop Suitability (Top 5)")
        grid_cols = st.columns(len(recs))
        for i, crop_item in enumerate(recs):
            with grid_cols[i]:
                st.markdown(
                    f"""
                    <div class="crop-pill" style="min-height: 220px;">
                        <div style="font-size: 1.2rem; font-weight: 800; color: #065f46;">#{crop_item['rank']} {crop_item['crop_name']}</div>
                        <div style="font-size: 0.95rem; font-weight: 800; color: #059669; margin: 4px 0;">{crop_item['suitability_score']:.1f}% Match</div>
                        <div style="font-size: 0.8rem; color: #374151; margin-top: 6px;"><b>Yield:</b> {crop_item['projected_yield']}</div>
                        <div style="font-size: 0.8rem; color: #374151;"><b>Water:</b> {crop_item['water_requirement']}</div>
                        <div style="font-size: 0.8rem; color: #374151;"><b>Cycle:</b> {crop_item['growing_duration_days']}</div>
                        <div style="font-size: 0.75rem; color: #6b7280; margin-top: 6px;"><i>{crop_item['cultivation_tips'][0] if crop_item['cultivation_tips'] else ''}</i></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Radar Envelope Comparison
        st.markdown("---")
        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            st.markdown("### 🕸️ Agronomic Fit Radar Chart")
            st.caption(f"Comparing Farm Inputs against Optimal Envelope for **{top_crop}**")
            try:
                cat_resp = requests.get(f"{BACKEND_URL}/api/crop/catalog", timeout=3)
                catalog_dict = cat_resp.json().get("catalog", {})
                target_profile = catalog_dict.get(top_crop, {})
                farm_dict = rec_data.get("input_profile", {})
                radar_fig = create_crop_radar_chart(farm_dict, target_profile)
                st.plotly_chart(radar_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render radar: {str(e)}")

        with col_r2:
            st.markdown(f"### 💡 Cultivation Guidelines for {top_crop}")
            for tip in recs[0].get("cultivation_tips", []):
                st.markdown(f"- 🌿 {tip}")
            st.markdown("#### Optimal Growth Boundaries:")
            ranges = recs[0].get("optimal_ranges", {})
            st.markdown(f"- **pH Window:** {ranges.get('ph', [6.0, 7.5])[0]} &ndash; {ranges.get('ph', [6.0, 7.5])[1]}")
            st.markdown(f"- **Nitrogen:** {ranges.get('nitrogen', [100, 160])[0]} &ndash; {ranges.get('nitrogen', [100, 160])[1]} kg/ha")
            st.markdown(f"- **Phosphorus:** {ranges.get('phosphorus', [30, 60])[0]} &ndash; {ranges.get('phosphorus', [30, 60])[1]} kg/ha")
            st.markdown(f"- **Potassium:** {ranges.get('potassium', [100, 200])[0]} &ndash; {ranges.get('potassium', [100, 200])[1]} kg/ha")
            st.markdown(f"- **Temperature:** {ranges.get('temperature', [20, 30])[0]}°C &ndash; {ranges.get('temperature', [20, 30])[1]}°C")
            st.markdown(f"- **Rainfall:** {ranges.get('rainfall', [100, 200])[0]} &ndash; {ranges.get('rainfall', [100, 200])[1]} mm")


# =========================================================================
# MODULE 5: SCAN HISTORY & EXPORT CENTER
# =========================================================================
elif active_view == "📜 Scan History & Export Center":
    st.markdown("## 📜 Historical Field Scans & Export Center")
    st.markdown("Inspect historical pipeline executions, filter by soil type or health grade, and download formal diagnostic reports.")

    # Platform Stats KPI Header
    try:
        stats_resp = requests.get(f"{BACKEND_URL}/api/pipeline/stats", timeout=5)
        if stats_resp.status_code == 200:
            s_data = stats_resp.json()
            kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
            with kpi_c1:
                st.metric("Total Plots Analyzed", s_data["total_farms_analyzed"])
            with kpi_c2:
                st.metric("Avg Farm Health Score", f"{s_data['avg_farm_health_score']} / 100")
            with kpi_c3:
                st.metric("Avg Farm Vigor Index", f"{s_data['avg_farm_vigor_index']} / 100")
    except Exception:
        pass

    st.markdown("---")
    st.markdown("### 📋 Historical Farm Diagnostic Records")

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        min_score = st.slider("Filter by Minimum Farm Health Score:", 0, 100, 0)
    with f_col2:
        soil_filter = st.selectbox("Filter by Soil Type:", ["All Types", "Alluvial Soil", "Black Soil", "Clay Soil", "Red Soil", "Sandy Soil", "Loamy Soil"])

    url = f"{BACKEND_URL}/api/pipeline/history?limit=30&min_health_score={min_score}"
    if soil_filter != "All Types":
        url += f"&soil_type={soil_filter}"

    try:
        hist_resp = requests.get(url, timeout=5)
        if hist_resp.status_code == 200:
            records = hist_resp.json()["records"]
            if records:
                df = pd.DataFrame(records)
                st.dataframe(df, use_container_width=True)

                st.markdown("### 📥 Instant Report Download")
                target_req_id = st.selectbox("Select Request ID to Export:", [r["request_id"] for r in records])
                if target_req_id:
                    d_c1, d_c2 = st.columns(2)
                    with d_c1:
                        html_export_url = f"{BACKEND_URL}/api/pipeline/report/{target_req_id}?format=html"
                        st.link_button("🖨️ Open Printable HTML Report (Save as PDF)", html_export_url, use_container_width=True)
                    with d_c2:
                        md_r = requests.get(f"{BACKEND_URL}/api/pipeline/report/{target_req_id}?format=markdown", timeout=5)
                        if md_r.status_code == 200:
                            st.download_button(
                                "📥 Download GFM Markdown Report",
                                data=md_r.text,
                                file_name=f"kisanseva_report_{target_req_id}.md",
                                mime="text/markdown",
                                use_container_width=True
                            )
            else:
                st.info("No matching records found. Run a new diagnosis in the Diagnostic Studio.")
    except Exception as exc:
        st.error(f"Error fetching historical records: {str(exc)}")
