"""
Physics formulas from the report: Aeff (effective load-bearing area)
and Dnorm (normalized Arrhenius diffusion rate for PA6).
"""

from __future__ import annotations

import math
from typing import Literal

InfillPattern = Literal["Grid", "Triangle", "Triangle-Hexa"]

ETA_COEFFICIENTS: dict[InfillPattern, float] = {
    "Triangle-Hexa": 1.00,
    "Triangle": 0.85,
    "Grid": 0.70,
}

EA_PA6_J_PER_MOL = 68_000.0
R_GAS = 8.314

# Normalization range for Dnorm must match the training-data temperature
# envelope (dataset_33.csv: 220-260 C), otherwise Dnorm is on a different scale
# than what the model learned on.
T_MIN_C = 220.0
T_MAX_C = 260.0


def effective_load_bearing_area(density_pct: float, pattern: InfillPattern) -> float:
    rho = density_pct / 100.0
    return rho * ETA_COEFFICIENTS[pattern]


def _arrhenius(temp_c: float) -> float:
    t_k = temp_c + 273.15
    return math.exp(-EA_PA6_J_PER_MOL / (R_GAS * t_k))


def diffusion_rate_normalized(temp_c: float) -> float:
    d = _arrhenius(temp_c)
    d_min = _arrhenius(T_MIN_C)
    d_max = _arrhenius(T_MAX_C)
    if d_max == d_min:
        return 0.5
    return (d - d_min) / (d_max - d_min)
