"""Streamlit UI for Outbreak Radar, Action Advisor and voice questions.

All data comes from the shared FastAPI backend, so the web dashboard and the
mobile app always show the same community reports and prices.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from frontend.utils.voice import LANG_CODES, read_aloud_button

LEVEL_COLORS = {"ALERT": "#dc2626", "WATCH": "#f59e0b", "NORMAL": "#16a34a"}
LEVEL_ICON = {"ALERT": "🔴", "WATCH": "🟠", "NORMAL": "🟢"}


def _get(url: str, **kw) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, timeout=kw.pop("timeout", 8), **kw)
        if r.status_code == 200:
            return r.json()
        st.error(f"Request failed ({r.status_code}): {r.text[:200]}")
    except Exception as ex:
        st.error(f"Cannot reach the API: {ex}")
    return None


def _post(url: str, **kw) -> Optional[requests.Response]:
    try:
        return requests.post(url, timeout=kw.pop("timeout", 15), **kw)
    except Exception as ex:
        st.error(f"Cannot reach the API: {ex}")
        return None


def _regions(base: str) -> List[Dict[str, Any]]:
    if "radar_regions" not in st.session_state:
        data = _get(f"{base}/api/community/regions")
        st.session_state.radar_regions = data["regions"] if data else []
    return st.session_state.radar_regions


# ----------------------------------------------------------------------------
# Share a diagnosis with the community
# ----------------------------------------------------------------------------
def render_share_to_radar(base: str, p_res: Dict[str, Any], key: str) -> None:
    """Offer to share a confident disease diagnosis anonymously."""
    if p_res.get("is_healthy"):
        return
    regions = _regions(base)
    if not regions:
        return
    with st.expander("🛰️ Warn nearby farmers: share this diagnosis (anonymous)"):
        st.caption("Only the district, crop and disease are shared. No photo, name or exact location.")
        names = [r["district"] for r in regions]
        district = st.selectbox("Your district", names, key=f"share_district_{key}")
        if st.button("Share with Outbreak Radar", key=f"share_btn_{key}"):
            resp = _post(
                f"{base}/api/community/report",
                json={
                    "condition": p_res.get("specific_condition") or p_res.get("condition"),
                    "confidence": float(p_res.get("confidence", 0.0)),
                    "district": district,
                    "crop": p_res.get("plant_name"),
                    "severity": p_res.get("severity_level"),
                },
            )
            if resp is not None and resp.status_code == 201:
                st.success(f"Thank you. Your report for {district} was added to the radar.")
            elif resp is not None:
                st.warning(resp.json().get("detail", resp.text))


# ----------------------------------------------------------------------------
# Outbreak Radar view
# ----------------------------------------------------------------------------
def render_radar_view(base: str) -> None:
    st.markdown("## 🛰️ Community Outbreak Radar")
    st.markdown(
        "Anonymous, district-level disease reports from farmers become an **early-warning map**. "
        "Only confident, non-healthy diagnoses are accepted."
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    days = c1.slider("Look-back window (days)", 3, 60, 14)
    include_demo = c2.toggle("Include demo data", value=True,
                             help="Demo reports are flagged and can be removed at any time.")
    with c3:
        b1, b2 = st.columns(2)
        if b1.button("➕ Load demo scenario", use_container_width=True):
            r = _post(f"{base}/api/community/demo/seed")
            if r is not None and r.status_code == 200:
                st.toast(f"Added {r.json()['reports_added']} demo reports")
        if b2.button("🧹 Remove demo data", use_container_width=True):
            try:
                r = requests.delete(f"{base}/api/community/demo", timeout=10)
                if r.status_code == 200:
                    st.toast(f"Removed {r.json()['reports_removed']} demo reports")
            except Exception as ex:
                st.error(f"Cannot reach the API: {ex}")

    data = _get(f"{base}/api/community/outbreaks", params={"days": days, "include_demo": include_demo})
    if not data:
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Cases in window", data["total_cases"])
    m2.metric("🔴 Alerts", data["alerts"])
    m3.metric("🟠 Watch", data["watch"])

    hotspots = data["hotspots"]
    if not hotspots:
        st.info("No reports in this window yet. Share a diagnosis from the Plant Doctor, "
                "or load the demo scenario to see the radar in action.")
    else:
        df = pd.DataFrame(hotspots)
        if df["includes_demo_data"].any():
            st.warning("Some points include **DEMO data**. Remove it before a real deployment.")
        fig = px.scatter_map(
            df, lat="lat", lon="lon", size="cases", color="level",
            color_discrete_map=LEVEL_COLORS, size_max=38, zoom=3.6,
            center={"lat": 22.5, "lon": 79.0},
            hover_name="district",
            hover_data={"condition": True, "cases": True, "trend": True, "lat": False, "lon": False, "level": False},
            map_style="open-street-map", height=520,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="Level")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Hotspots")
        show = df[["level", "district", "state", "condition", "cases", "prior_cases", "trend", "top_crop"]].copy()
        show["level"] = show["level"].map(lambda x: f"{LEVEL_ICON[x]} {x}")
        st.dataframe(show, use_container_width=True, hide_index=True)

    th = data["thresholds"]
    st.caption(
        f"WATCH ≥ {th['watch_min_cases']} cases · ALERT ≥ {th['alert_min_cases']} cases and "
        f"≥ {th['alert_growth_factor']}× the previous window · min report confidence "
        f"{th['min_report_confidence']:.0%}"
    )

    st.markdown("---")
    st.markdown("### 📍 Check my area")
    regions = _regions(base)
    if regions:
        names = [r["district"] for r in regions]
        cc1, cc2 = st.columns([2, 1])
        mine = cc1.selectbox("My district", names, index=names.index("Ludhiana") if "Ludhiana" in names else 0)
        radius = cc2.slider("Radius (km)", 25, 500, 150, step=25)
        loc = next(r for r in regions if r["district"] == mine)
        near = _get(f"{base}/api/community/nearby",
                    params={"lat": loc["lat"], "lon": loc["lon"], "radius_km": radius,
                            "days": days, "include_demo": include_demo})
        if near:
            level = near["highest_level"]
            if near["hotspots"]:
                top = near["hotspots"][0]
                msg = (f"{LEVEL_ICON[level]} **{level}** near {mine}: {top['condition']} in "
                       f"{top['district']} ({top['distance_km']} km away, {top['cases']} cases, {top['trend']}).")
                (st.error if level == "ALERT" else st.warning if level == "WATCH" else st.info)(msg)
                read_aloud_button(msg.replace("**", ""), "English")
                st.dataframe(pd.DataFrame(near["hotspots"])[
                    ["distance_km", "district", "condition", "cases", "trend", "level"]],
                    use_container_width=True, hide_index=True)
            else:
                st.success(f"No reported outbreaks within {radius} km of {mine} in the last {days} days.")

    st.markdown("---")
    st.markdown("### 📝 Report a case")
    last = st.session_state.get("plant_only_result") or {}
    with st.form("manual_report"):
        f1, f2 = st.columns(2)
        district = f1.selectbox("District", [r["district"] for r in regions] or ["-"], key="rep_dist")
        crop = f2.text_input("Crop", value=last.get("plant_name") or "")
        f3, f4 = st.columns(2)
        condition = f3.text_input("Disease", value=last.get("specific_condition") or last.get("condition") or "")
        conf = f4.slider("Model confidence", 0.0, 1.0, float(last.get("confidence", 0.8)))
        if st.form_submit_button("Submit anonymous report"):
            resp = _post(f"{base}/api/community/report",
                         json={"condition": condition, "confidence": conf, "district": district, "crop": crop or None})
            if resp is not None and resp.status_code == 201:
                st.success("Report accepted. Thank you!")
            elif resp is not None:
                st.warning(resp.json().get("detail", resp.text))


# ----------------------------------------------------------------------------
# Action Advisor view
# ----------------------------------------------------------------------------
def _inr(x: Optional[float]) -> str:
    return "n/a" if x is None else f"₹{x:,.0f}"


def render_advisor_view(base: str) -> None:
    st.markdown("## 💰 Action Advisor: Cost & Profit Estimate")
    st.markdown("Turn soil values into a **fertilizer plan, cost and indicative revenue** for your field.")

    crops_data = _get(f"{base}/api/advisor/crops")
    if not crops_data:
        return
    crop_names = [c["crop"] for c in crops_data["crops"]]
    ctx = st.session_state.get("farm_ctx", {})

    c1, c2 = st.columns(2)
    crop = c1.selectbox("Crop", crop_names)
    area = c2.number_input("Field area (acres)", min_value=0.1, max_value=10000.0, value=2.0, step=0.5)

    st.caption("Leave a nutrient blank if you do not know it; the advisor will not assume a deficit.")
    n1, n2, n3 = st.columns(3)

    def _opt(label: str, default: Optional[float], col) -> Optional[float]:
        txt = col.text_input(label, value="" if default is None else str(default))
        try:
            return float(txt) if txt.strip() else None
        except ValueError:
            col.error("Enter a number")
            return None

    nit = _opt("Nitrogen N (kg/ha)", ctx.get("nitrogen"), n1)
    pho = _opt("Phosphorus P (kg/ha)", ctx.get("phosphorus"), n2)
    pot = _opt("Potassium K (kg/ha)", ctx.get("potassium"), n3)

    if st.button("Calculate plan", type="primary"):
        resp = _post(f"{base}/api/advisor/economics",
                     json={"crop": crop, "area_acres": area, "nitrogen": nit, "phosphorus": pho, "potassium": pot})
        if resp is None:
            return
        if resp.status_code != 200:
            st.error(resp.json().get("detail", resp.text))
            return
        st.session_state.advisor_result = resp.json()

    e = st.session_state.get("advisor_result")
    if not e or e.get("crop") != crop:
        return

    fert = e["fertilizer"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected yield", f"{e['expected_yield_tons']} t")
    m2.metric("Fertilizer cost", _inr(fert["total_fertilizer_cost_inr"]))
    m3.metric("Indicative revenue", _inr(e["expected_gross_revenue_inr"]))
    m4.metric("Margin after fertilizer", _inr(e["margin_after_fertilizer_inr"]))

    st.info(f"**Price basis:** {e['price_basis']}"
            + (f" · ₹{e['price_inr_per_quintal']:,.0f}/quintal" if e["price_inr_per_quintal"] else ""))
    st.caption(f"{e['yield_basis']}. {e['margin_note']}")

    st.markdown("#### 🧪 Fertilizer plan")
    kg, cost = fert["fertilizer_kg"], fert["fertilizer_cost_inr"]
    st.dataframe(pd.DataFrame([
        {"Fertilizer": "Urea (46% N)", "Quantity (kg)": kg["urea"], "Cost (₹)": cost["urea"]},
        {"Fertilizer": "DAP (18-46-0)", "Quantity (kg)": kg["dap"], "Cost (₹)": cost["dap"]},
        {"Fertilizer": "MOP (60% K₂O)", "Quantity (kg)": kg["mop"], "Cost (₹)": cost["mop"]},
    ]), use_container_width=True, hide_index=True)
    if fert["unknown_nutrients"]:
        st.warning("Not provided, so no deficit was assumed for: " + ", ".join(fert["unknown_nutrients"]))
    st.caption(f"Target (kg/ha): {fert['targets_kg_per_ha']} · Deficit: {fert['deficit_kg_per_ha']} · "
               f"Prices (₹/kg, reference defaults): {fert['prices_inr_per_kg']}")

    if e["cultivation_tips"]:
        with st.expander("Cultivation tips"):
            for t in e["cultivation_tips"]:
                st.markdown(f"- {t}")

    summary = (f"For {area} acres of {crop}: expected yield {e['expected_yield_tons']} tons. "
               f"Fertilizer needed: {kg['urea']} kg urea, {kg['dap']} kg DAP, {kg['mop']} kg MOP, "
               f"costing about {_inr(fert['total_fertilizer_cost_inr'])}.")
    read_aloud_button(summary, "English")

    with st.expander("📊 Compare crops for this field"):
        picks = st.multiselect("Crops", crop_names, default=crop_names[:4])
        if picks and st.button("Compare"):
            r = _post(f"{base}/api/advisor/compare",
                      json={"crops": picks, "area_acres": area, "nitrogen": nit, "phosphorus": pho, "potassium": pot})
            if r is not None and r.status_code == 200:
                st.dataframe(pd.DataFrame(r.json()["ranking"]).rename(columns={
                    "expected_gross_revenue_inr": "Revenue ₹", "fertilizer_cost_inr": "Fertilizer ₹",
                    "margin_after_fertilizer_inr": "Margin ₹", "price_basis": "Price basis"}),
                    use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# Voice question
# ----------------------------------------------------------------------------
def render_voice_question(base: str, key: str, language: str, context: Dict[str, Any]) -> None:
    """Record a spoken question and get a spoken-ready answer (needs GEMINI_API_KEY)."""
    status = _get(f"{base}/api/ai/voice/status", timeout=4) or {}
    audio = st.audio_input("🎤 Ask by voice", key=f"voice_in_{key}")
    if not audio:
        if not status.get("voice_question_enabled", False):
            st.caption("Voice questions need `GEMINI_API_KEY` on the server. You can still type your question.")
        return
    if st.button("Send voice question", key=f"voice_send_{key}"):
        import json
        resp = _post(
            f"{base}/api/ai/voice-chat",
            files={"audio": ("question.wav", audio.getvalue(), "audio/wav")},
            data={"language": language, "context": json.dumps(context, default=str)},
            timeout=40,
        )
        if resp is None:
            return
        if resp.status_code == 200:
            j = resp.json()
            st.markdown(f"**You said:** {j.get('transcript') or '(not captured)'}")
            st.markdown(j["reply"])
            read_aloud_button(j["reply"], language)
        else:
            st.warning(resp.json().get("detail", resp.text))
