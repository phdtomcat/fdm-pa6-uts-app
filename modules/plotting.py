"""
Charts: predicted vs measured scatter, error histogram.
Uses Plotly so Streamlit can render interactively.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def predicted_vs_measured_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(
            title="Predicted vs Measured (no data)",
            xaxis_title="Measured UTS (MPa)",
            yaxis_title="Predicted UTS (MPa)",
        )
        return fig

    measured = df["measured_uts"].astype(float)
    predicted = df["predicted_uts"].astype(float)

    fig.add_trace(
        go.Scatter(
            x=measured,
            y=predicted,
            mode="markers",
            name="Measurements",
            marker=dict(size=10, color=df.get("pattern", "blue").astype("category").cat.codes
                       if "pattern" in df.columns else "blue",
                       showscale=False),
            text=df.apply(
                lambda r: f"{r['pattern']} | ρ={r['density_pct']}% | T={r['temp_c']}°C",
                axis=1,
            ),
            hovertemplate="%{text}<br>Predicted: %{y:.2f}<br>Measured: %{x:.2f}<extra></extra>",
        )
    )

    lo = float(min(measured.min(), predicted.min())) - 1.0
    hi = float(max(measured.max(), predicted.max())) + 1.0
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="y = x (perfect fit)",
            line=dict(dash="dash", color="gray"),
        )
    )
    fig.update_layout(
        title="Predicted vs Measured UTS",
        xaxis_title="Measured UTS (MPa)",
        yaxis_title="Predicted UTS (MPa)",
        height=450,
    )
    return fig


def error_histogram(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="Error distribution (no data)")
        return fig
    residuals = (df["predicted_uts"].astype(float) - df["measured_uts"].astype(float))
    fig.add_trace(go.Histogram(x=residuals, nbinsx=20, name="Predicted − Measured"))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Error distribution (Predicted − Measured)",
        xaxis_title="Error (MPa)",
        yaxis_title="Number of measurements",
        height=350,
    )
    return fig
