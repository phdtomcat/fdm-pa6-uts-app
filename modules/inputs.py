"""
Shared input widgets (density / temp / pattern) - used in both
Prediction and Validation tabs so there is no duplication.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from .config import (
    DENSITY_DEFAULT,
    DENSITY_MAX,
    DENSITY_MIN,
    DENSITY_STEP,
    PATTERNS,
    TEMP_DEFAULT,
    TEMP_MAX,
    TEMP_MIN,
    TEMP_STEP,
)


@dataclass
class ProcessInputs:
    density_pct: float
    temp_c: float
    pattern: str


def render_process_inputs(key_prefix: str) -> ProcessInputs:
    col1, col2, col3 = st.columns(3)
    with col1:
        density = st.number_input(
            "Infill density (%)",
            min_value=DENSITY_MIN,
            max_value=DENSITY_MAX,
            value=DENSITY_DEFAULT,
            step=DENSITY_STEP,
            key=f"{key_prefix}_density",
            help="Training domain range: 40–80 %",
        )
    with col2:
        temp = st.number_input(
            "Nozzle temperature (°C)",
            min_value=TEMP_MIN,
            max_value=TEMP_MAX,
            value=TEMP_DEFAULT,
            step=TEMP_STEP,
            key=f"{key_prefix}_temp",
            help="Training domain range: 220–260 °C",
        )
    with col3:
        pattern = st.selectbox(
            "Infill pattern",
            options=PATTERNS,
            index=PATTERNS.index("Triangle-Hexa"),
            key=f"{key_prefix}_pattern",
        )
    return ProcessInputs(density_pct=float(density), temp_c=float(temp), pattern=pattern)
