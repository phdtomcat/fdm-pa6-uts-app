"""
Tab 1: UTS prediction from process parameters.
"""

from __future__ import annotations

import streamlit as st

from .inputs import render_process_inputs
from .predictor import PlaceholderUTSModel


def render(model: PlaceholderUTSModel) -> None:
    st.subheader("UTS Prediction")
    st.caption(
        "Enter print parameters - the model returns predicted "
        "Ultimate Tensile Strength with an uncertainty band."
    )

    inputs = render_process_inputs(key_prefix="predict")

    if st.button("Compute prediction", type="primary", use_container_width=True):
        result = model.predict(
            density_pct=inputs.density_pct,
            temp_c=inputs.temp_c,
            pattern=inputs.pattern,
        )
        _show_result(result)


def _show_result(result) -> None:
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Predicted UTS",
            value=f"{result.uts_mpa:.2f} MPa",
            delta=f"± {result.uncertainty_mpa:.2f} MPa",
        )
    with col2:
        st.metric(label="Aeff (load-bearing area)", value=f"{result.aeff:.3f}")
    with col3:
        st.metric(label="Dnorm (diffusion rate)", value=f"{result.d_norm:.3f}")

    lo = result.uts_mpa - result.uncertainty_mpa
    hi = result.uts_mpa + result.uncertainty_mpa
    st.info(
        f"**Confidence interval:** {lo:.2f} – {hi:.2f} MPa "
        f"(basis: CV RMSE around 1 MPa for the real model)"
    )

    if result.is_placeholder:
        st.warning(
            "⚠️ A PLACEHOLDER model is currently active (physics formula, no fit). "
            "Values are indicative until you load the real LightGBM trained on the CSV."
        )
