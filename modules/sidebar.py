"""
Sidebar: model info and quick validation status overview.
"""

from __future__ import annotations

import streamlit as st

from .metrics import compute_summary
from .predictor import PlaceholderUTSModel
from .validation_log import load_log


def render(model: PlaceholderUTSModel) -> None:
    st.sidebar.title("Dog Bone UTS Predictor")
    st.sidebar.markdown(
        f"**Model:** `{model.name}`\n\n"
        f"**Material:** PA6 (FDM)\n\n"
        f"**Target:** Ultimate Tensile Strength (MPa)"
    )

    if model.is_placeholder:
        st.sidebar.warning(
            "Placeholder model is active. To get real predictions, load the "
            "trained LightGBM model (see `modules/predictor.py`)."
        )

    st.sidebar.divider()
    st.sidebar.markdown("### Validation quick status")

    df = load_log()
    summary = compute_summary(df)
    st.sidebar.metric("Logged measurements", summary["n"])
    if summary["n"] > 0:
        st.sidebar.metric("RMSE", f"{summary['rmse']:.3f} MPa")
        st.sidebar.metric("MAE", f"{summary['mae']:.3f} MPa")
