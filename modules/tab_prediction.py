"""
Tab 1: UTS prediction from process parameters.
"""

from __future__ import annotations

import streamlit as st

from .config import DENSITY_MAX, DENSITY_MIN, TEMP_MAX, TEMP_MIN
from .inputs import render_process_inputs
from .predictor import PlaceholderUTSModel


def render(model: PlaceholderUTSModel) -> None:
    st.subheader("UTS Prediction")
    st.caption(
        "Enter print parameters - the model returns predicted "
        "Ultimate Tensile Strength with an uncertainty band."
    )

    _render_domain_box()

    inputs = render_process_inputs(key_prefix="predict")

    if st.button("Compute prediction", type="primary", use_container_width=True):
        _warn_if_out_of_domain(inputs)
        result = model.predict(
            density_pct=inputs.density_pct,
            temp_c=inputs.temp_c,
            pattern=inputs.pattern,
        )
        _show_result(result)


def _render_domain_box() -> None:
    """Prominent box telling the user the range in which the model is valid."""
    with st.container(border=True):
        st.markdown("### ⚠️ Valid model domain — print inside these ranges")
        st.markdown(
            f"- **Infill density:** {DENSITY_MIN:.0f} – {DENSITY_MAX:.0f} %\n"
            f"- **Nozzle temperature:** {TEMP_MIN:.0f} – {TEMP_MAX:.0f} °C\n"
            "- **Infill pattern:** Grid · Triangle · Triangle-Hexa"
        )
        st.markdown(
            "The model is tree-based (LightGBM) and **cannot extrapolate**. "
            "Outside this envelope it returns a flat, capped value and will "
            "systematically **under-predict** stronger specimens. For meaningful "
            "results, print and test specimens **within** these ranges."
        )


def _warn_if_out_of_domain(inputs) -> None:
    """Defensive check in case inputs ever land outside the training envelope."""
    issues = []
    if not (DENSITY_MIN <= inputs.density_pct <= DENSITY_MAX):
        issues.append(
            f"density {inputs.density_pct:.0f} % is outside {DENSITY_MIN:.0f}–{DENSITY_MAX:.0f} %"
        )
    if not (TEMP_MIN <= inputs.temp_c <= TEMP_MAX):
        issues.append(
            f"temperature {inputs.temp_c:.0f} °C is outside {TEMP_MIN:.0f}–{TEMP_MAX:.0f} °C"
        )
    if issues:
        st.error(
            "🚫 **Extrapolation — prediction unreliable.** "
            + "; ".join(issues)
            + ". The model has no training data here and will under-predict."
        )


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
