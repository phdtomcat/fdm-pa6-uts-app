"""
Validation metrics: RMSE, MAE, R^2, and percentage of predictions
within each tolerance tier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TOLERANCE_TIERS


def compute_summary(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty:
        return {
            "n": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "mean_abs_error": float("nan"),
        }
    y_pred = df["predicted_uts"].to_numpy(dtype=float)
    y_true = df["measured_uts"].to_numpy(dtype=float)
    residuals = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n": int(len(df)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mean_abs_error": mae,
    }


def tier_accuracy(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {name: 0.0 for name in TOLERANCE_TIERS}
    errors = df["abs_error"].to_numpy(dtype=float)
    n = len(errors)
    return {
        name: float(np.sum(errors <= eps) / n * 100.0)
        for name, eps in TOLERANCE_TIERS.items()
    }
