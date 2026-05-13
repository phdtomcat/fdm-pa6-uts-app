"""
Tab 2: Validation - log a measured UTS from an actual experiment.
Stores into the CSV log and shows deviation from the prediction.
"""

from __future__ import annotations

import streamlit as st

from .inputs import render_process_inputs
from .predictor import PlaceholderUTSModel
from .validation_log import append_entry


def render(model: PlaceholderUTSModel) -> None:
    st.subheader("Validation - compare against measurements")
    st.caption(
        "Enter print parameters AND the measured UTS from the tensile test. "
        "The app computes the prediction and stores both in the log for accuracy analysis."
    )

    inputs = render_process_inputs(key_prefix="validate")

    st.markdown("##### Measured value")
    col_m, col_n = st.columns([1, 2])
    with col_m:
        measured = st.number_input(
            "Measured UTS (MPa)",
            min_value=0.0,
            max_value=50.0,
            value=12.0,
            step=0.1,
            key="validate_measured",
        )
    with col_n:
        note = st.text_input(
            "Note (e.g. specimen ID, print date)",
            value="",
            key="validate_note",
        )

    if st.button("Save measurement", type="primary", use_container_width=True):
        prediction = model.predict(
            density_pct=inputs.density_pct,
            temp_c=inputs.temp_c,
            pattern=inputs.pattern,
        )
        append_entry(
            density_pct=inputs.density_pct,
            temp_c=inputs.temp_c,
            pattern=inputs.pattern,
            predicted_uts=prediction.uts_mpa,
            measured_uts=float(measured),
            model_name=prediction.model_name,
            note=note,
        )
        _show_comparison(prediction.uts_mpa, float(measured))


def _show_comparison(predicted: float, measured: float) -> None:
    abs_error = abs(predicted - measured)
    rel_error = (abs_error / measured * 100.0) if measured else 0.0

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", f"{predicted:.2f} MPa")
    with col2:
        st.metric("Measurement", f"{measured:.2f} MPa")
    with col3:
        st.metric(
            "Absolute error",
            f"{abs_error:.2f} MPa",
            delta=f"{rel_error:.1f} % relative",
            delta_color="inverse",
        )

    if abs_error <= 0.25:
        st.success("✅ Within High Precision tier (±0.25 MPa)")
    elif abs_error <= 0.50:
        st.success("✅ Within Engineering tier (±0.50 MPa) - gold standard")
    elif abs_error <= 1.00:
        st.info("ℹ️ Within Acceptable tier (±1.00 MPa)")
    elif abs_error <= 2.00:
        st.warning("⚠️ Within Upper Bound tier (±2.00 MPa)")
    else:
        st.error("❌ Outside all tiers - model is not reliable for this combination")

    st.caption("Measurement saved to log. See the Statistics tab for cumulative analysis.")
