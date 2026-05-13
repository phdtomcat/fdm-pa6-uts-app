"""
UTS prediction backends.

Two implementations:
  - LightGBMUTSModel: trained LightGBM from data/model.txt (Weighted Physics
    Augmented Approach i, hyperparameters from the report)
  - PlaceholderUTSModel: linear surrogate, fallback if model.txt is missing

load_model() picks automatically: if data/model.txt exists - LightGBM, otherwise placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import MODEL_PATH
from .physics import (
    InfillPattern,
    effective_load_bearing_area,
    diffusion_rate_normalized,
)


@dataclass
class Prediction:
    uts_mpa: float
    uncertainty_mpa: float
    aeff: float
    d_norm: float
    model_name: str
    is_placeholder: bool


class PlaceholderUTSModel:
    """
    Linear surrogate. Coefficients are chosen to produce realistic UTS values
    inside the 8-19 MPa range from the report, following the report's feature
    importance (Aeff dominant, Dnorm secondary, pattern offset).
    Not fit to data - exists only for UI smoke testing.
    """

    name = "Placeholder (physics-only, no fit)"
    is_placeholder = True
    uncertainty_mpa = 1.5

    BASE_UTS = 6.0
    K_AEFF = 14.0
    K_DNORM = 3.0
    PATTERN_OFFSET: dict[InfillPattern, float] = {
        "Triangle-Hexa": 0.5,
        "Triangle": 0.0,
        "Grid": -0.4,
    }

    def predict(
        self,
        density_pct: float,
        temp_c: float,
        pattern: InfillPattern,
    ) -> Prediction:
        aeff = effective_load_bearing_area(density_pct, pattern)
        d_norm = diffusion_rate_normalized(temp_c)
        uts = (
            self.BASE_UTS
            + self.K_AEFF * aeff
            + self.K_DNORM * d_norm
            + self.PATTERN_OFFSET[pattern]
        )
        return Prediction(
            uts_mpa=uts,
            uncertainty_mpa=self.uncertainty_mpa,
            aeff=aeff,
            d_norm=d_norm,
            model_name=self.name,
            is_placeholder=self.is_placeholder,
        )


class LightGBMUTSModel:
    """
    Wrapper around the trained LightGBM model (Weighted Physics-Augmented
    Approach i from the report). Loads the model from LightGBM native txt format.

    Features (same order as in training):
      Infill_Pattern_Grid, Infill_Pattern_Triangle, Infill_Pattern_Triangle-Hexa,
      Aeff, Dnorm
    """

    name = "LightGBM (Physics-Augmented, trained on 33 conditions)"
    is_placeholder = False

    FEATURE_ORDER = [
        "Infill_Pattern_Grid",
        "Infill_Pattern_Triangle",
        "Infill_Pattern_Triangle-Hexa",
        "Aeff",
        "Dnorm",
    ]

    def __init__(self, model_path: Path):
        import lightgbm as lgb

        self._booster = lgb.Booster(model_file=str(model_path))
        self.uncertainty_mpa = 1.15

    def predict(
        self,
        density_pct: float,
        temp_c: float,
        pattern: InfillPattern,
    ) -> Prediction:
        aeff = effective_load_bearing_area(density_pct, pattern)
        d_norm = diffusion_rate_normalized(temp_c)

        features = [[
            1.0 if pattern == "Grid" else 0.0,
            1.0 if pattern == "Triangle" else 0.0,
            1.0 if pattern == "Triangle-Hexa" else 0.0,
            aeff,
            d_norm,
        ]]
        uts = float(self._booster.predict(features)[0])

        return Prediction(
            uts_mpa=uts,
            uncertainty_mpa=self.uncertainty_mpa,
            aeff=aeff,
            d_norm=d_norm,
            model_name=self.name,
            is_placeholder=self.is_placeholder,
        )


def load_model(model_path: Path | None = None) -> "LightGBMUTSModel | PlaceholderUTSModel":
    """
    If data/model.txt exists, returns the trained LightGBM. Otherwise placeholder.
    """
    path = model_path if model_path is not None else MODEL_PATH
    if path.exists():
        try:
            return LightGBMUTSModel(path)
        except Exception as e:
            print(f"Error loading LightGBM model ({e}), falling back to placeholder.")
    return PlaceholderUTSModel()
