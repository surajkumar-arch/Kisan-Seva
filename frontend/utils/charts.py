"""Interactive Plotly Data Visualizations for Phase 7 Dashboard."""

from typing import Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def create_health_gauge(value: int, title: str, subtitle: str = "") -> go.Figure:
    """Create a sleek, modern gauge chart for health scores and indices (0-100)."""
    if value >= 80:
        bar_color = "#10b981"  # Emerald
    elif value >= 60:
        bar_color = "#f59e0b"  # Amber
    else:
        bar_color = "#ef4444"  # Red

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": f"<b>{title}</b><br><span style='font-size:12px;color:#6b7280'>{subtitle}</span>", "font": {"size": 15, "color": "#1f2937"}},
            number={"font": {"size": 34, "color": bar_color, "family": "Inter, sans-serif"}, "suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af", "tickfont": {"size": 10}},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#f3f4f6",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(239, 68, 68, 0.12)"},
                    {"range": [50, 75], "color": "rgba(245, 158, 11, 0.12)"},
                    {"range": [75, 100], "color": "rgba(16, 185, 129, 0.12)"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 4},
                    "thickness": 0.8,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        height=190,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif"),
    )
    return fig


def create_crop_radar_chart(farm_params: Dict[str, Any], crop_profile: Dict[str, Any]) -> go.Figure:
    """Create a multi-dimensional radar chart comparing current farm levels vs optimal crop envelope."""
    categories = ["Soil pH", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature", "Rainfall"]

    # Normalize farm values to [0, 100] scale for visualization
    ph_val = float(farm_params.get("ph") or 6.5)
    n_val = float(farm_params.get("nitrogen") or 140.0)
    p_val = float(farm_params.get("phosphorus") or 35.0)
    k_val = float(farm_params.get("potassium") or 150.0)
    temp_val = float(farm_params.get("temperature") or 25.0)
    rain_val = float(farm_params.get("rainfall") or 120.0)

    # Scale functions to 0-100
    norm_farm = [
        min(100, max(0, (ph_val / 14.0) * 100)),
        min(100, max(0, (n_val / 250.0) * 100)),
        min(100, max(0, (p_val / 100.0) * 100)),
        min(100, max(0, (k_val / 300.0) * 100)),
        min(100, max(0, ((temp_val + 5) / 50.0) * 100)),
        min(100, max(0, (rain_val / 300.0) * 100)),
    ]

    # Crop optimal midpoints
    opt_ph = np.mean(crop_profile.get("optimal_ph", [6.0, 7.5]))
    opt_n = np.mean(crop_profile.get("optimal_n", [100, 180]))
    opt_p = np.mean(crop_profile.get("optimal_p", [30, 60]))
    opt_k = np.mean(crop_profile.get("optimal_k", [100, 200]))
    opt_t = np.mean(crop_profile.get("optimal_temp", [20, 30]))
    opt_r = np.mean(crop_profile.get("optimal_rainfall", [100, 200]))

    norm_crop = [
        min(100, max(0, (opt_ph / 14.0) * 100)),
        min(100, max(0, (opt_n / 250.0) * 100)),
        min(100, max(0, (opt_p / 100.0) * 100)),
        min(100, max(0, (opt_k / 300.0) * 100)),
        min(100, max(0, ((opt_t + 5) / 50.0) * 100)),
        min(100, max(0, (opt_r / 300.0) * 100)),
    ]

    # Close the polygon loop
    categories_loop = categories + [categories[0]]
    norm_farm_loop = norm_farm + [norm_farm[0]]
    norm_crop_loop = norm_crop + [norm_crop[0]]

    fig = go.Figure()

    # Optimal crop polygon
    fig.add_trace(
        go.Scatterpolar(
            r=norm_crop_loop,
            theta=categories_loop,
            fill="toself",
            name=f"Optimal {crop_profile.get('crop_name', 'Crop')}",
            line=dict(color="#10b981", width=2),
            fillcolor="rgba(16, 185, 129, 0.2)",
        )
    )

    # Current farm polygon
    fig.add_trace(
        go.Scatterpolar(
            r=norm_farm_loop,
            theta=categories_loop,
            fill="toself",
            name="Current Farm State",
            line=dict(color="#3b82f6", width=2, dash="dash"),
            fillcolor="rgba(59, 130, 246, 0.25)",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
            bgcolor="rgba(249, 250, 251, 0.6)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=320,
        margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_modality_weights_donut(modality_weights: Dict[str, float]) -> go.Figure:
    """Donut chart illustrating Phase 4 learned attention gating distribution."""
    labels = ["Soil Domain", "Plant Foliage", "Climate & Weather"]
    vals = [
        modality_weights.get("soil_weight", 0.35),
        modality_weights.get("plant_weight", 0.35),
        modality_weights.get("climate_weight", 0.30),
    ]
    colors = ["#8b5cf6", "#10b981", "#0284c7"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=vals,
                hole=0.6,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hoverinfo="label+value",
            )
        ]
    )
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text="<b>Attention</b><br>Gating",
                x=0.5,
                y=0.5,
                font_size=12,
                showarrow=False,
                font_color="#4b5563",
            )
        ],
    )
    return fig


def create_stress_scores_bar(stress_scores: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart indicating multi-modal domain risks and stress."""
    factors = ["Soil Stress", "Pathogen Risk", "Climate Stress"]
    values = [
        stress_scores.get("soil_stress", 0.2),
        stress_scores.get("plant_disease_risk", 0.1),
        stress_scores.get("environmental_stress", 0.15),
    ]
    colors = ["#ef4444" if v > 0.6 else ("#f59e0b" if v > 0.3 else "#10b981") for v in values]

    fig = go.Figure(
        go.Bar(
            x=[v * 100 for v in values],
            y=factors,
            orientation="h",
            marker=dict(color=colors, cornerradius=4),
            text=[f"{int(v * 100)}%" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Stress Factor (0-100%)", showgrid=True),
        yaxis=dict(autorange="reversed"),
        height=180,
        margin=dict(l=10, r=20, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_spatial_farm_heatmap(vigor_index: int, condition: str) -> go.Figure:
    """Generate interactive 5x5 zone spatial field vigor & pathogen dispersion map."""
    np.random.seed(42)
    base_val = vigor_index
    # 5x5 spatial plot grid (Hectares A1 to E5)
    noise = np.random.normal(loc=0, scale=8, size=(5, 5))
    grid = np.clip(base_val + noise, 10, 99)

    # If disease present, simulate a localized cluster
    if condition not in ["Healthy Foliage", "Healthy"]:
        grid[1:3, 1:3] -= 25.0
        grid = np.clip(grid, 5, 99)

    x_labels = [f"Col {i+1}" for i in range(5)]
    y_labels = [f"Row {chr(65+i)}" for i in range(5)]

    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            x=x_labels,
            y=y_labels,
            colorscale=[
                [0.0, "#ef4444"],    # Degraded / diseased
                [0.5, "#f59e0b"],    # Moderate
                [1.0, "#10b981"],    # Optimal vigor
            ],
            zmin=0,
            zmax=100,
            text=[[f"{val:.0f} FVI" for val in row] for row in grid],
            texttemplate="%{text}",
            hoverongaps=False,
            colorbar=dict(title="Vigor (FVI)", len=0.8),
        )
    )
    fig.update_layout(
        title=dict(text="<b>Field Zone Spatial Vigor Distribution (5-Hectare Plot)</b>", font=dict(size=13)),
        height=260,
        margin=dict(l=30, r=30, t=35, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_disease_probabilities_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart showing neural network probabilities across disease classes."""
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=False)
    classes = [k for k, _ in sorted_items]
    probs = [v * 100 for _, v in sorted_items]

    colors = []
    for c in classes:
        if c == "Healthy Foliage":
            colors.append("#10b981")
        else:
            colors.append("#ef4444")

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=classes,
            orientation="h",
            marker=dict(color=colors, cornerradius=4),
            text=[f"{p:.1f}%" for p in probs],
            textposition="auto",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 105], title="Confidence (%)", showgrid=True),
        height=240,
        margin=dict(l=10, r=20, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

