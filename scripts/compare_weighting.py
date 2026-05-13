"""
Compare four sample-weighting strategies for the LightGBM UTS model.

A: current weights = 1/std (no clipping)
B: clipped weights = 1/max(std, 0.3)
C: regularized weights = 1/sqrt(std^2 + median(std)^2)
D: uniform weights (no weighting)

Output: 5-fold CV R^2 and RMSE for each, plus train fit on full 33 points.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.physics import (
    effective_load_bearing_area,
    diffusion_rate_normalized,
)

LGB_PARAMS = {
    "boosting_type": "gbdt",
    "n_estimators": 182,
    "learning_rate": 0.05,
    "num_leaves": 4,
    "min_child_samples": 3,
    "max_depth": 2,
    "feature_fraction": 0.8,
    "reg_lambda": 0.5,
    "reg_alpha": 0.1,
    "random_state": 32,
    "verbosity": -1,
}

FEATURE_NAMES = [
    "Infill_Pattern_Grid",
    "Infill_Pattern_Triangle",
    "Infill_Pattern_Triangle-Hexa",
    "Aeff",
    "Dnorm",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Infill_Pattern_Grid"] = (df["pattern"] == "Grid").astype(int)
    out["Infill_Pattern_Triangle"] = (df["pattern"] == "Triangle").astype(int)
    out["Infill_Pattern_Triangle-Hexa"] = (df["pattern"] == "Triangle-Hexa").astype(int)
    out["Aeff"] = df.apply(
        lambda r: effective_load_bearing_area(r["density_pct"], r["pattern"]), axis=1
    )
    out["Dnorm"] = df["temp_c"].apply(diffusion_rate_normalized)
    return out[FEATURE_NAMES]


def weights_current(std: pd.Series) -> np.ndarray:
    eps = 1e-3
    return 1.0 / std.fillna(std.mean()).clip(lower=eps).to_numpy()


def weights_clipped(std: pd.Series, floor: float = 0.3) -> np.ndarray:
    return 1.0 / std.fillna(std.mean()).clip(lower=floor).to_numpy()


def weights_regularized(std: pd.Series) -> np.ndarray:
    s = std.fillna(std.mean()).to_numpy()
    sigma_0 = float(np.median(s))
    return 1.0 / np.sqrt(s**2 + sigma_0**2)


def weights_uniform(std: pd.Series) -> np.ndarray:
    return np.ones(len(std))


def evaluate(label: str, X: pd.DataFrame, y: pd.Series, w: np.ndarray) -> None:
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2, fold_rmse = [], []
    for train_idx, test_idx in kf.split(X_train):
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx],
            sample_weight=w_train[train_idx],
        )
        pred = m.predict(X_train.iloc[test_idx])
        fold_r2.append(r2_score(y_train.iloc[test_idx], pred))
        fold_rmse.append(np.sqrt(mean_squared_error(y_train.iloc[test_idx], pred)))

    final = lgb.LGBMRegressor(**LGB_PARAMS)
    final.fit(X_train, y_train, sample_weight=w_train)
    test_pred = final.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))

    full = lgb.LGBMRegressor(**LGB_PARAMS)
    full.fit(X, y, sample_weight=w)
    full_pred = full.predict(X)
    full_r2 = r2_score(y, full_pred)

    print(
        f"{label:35s} | CV R^2: {np.mean(fold_r2):+.4f} ± {np.std(fold_r2):.4f} "
        f"| CV RMSE: {np.mean(fold_rmse):.3f} "
        f"| Test R^2: {test_r2:+.4f} | Test RMSE: {test_rmse:.3f} "
        f"| Full-fit R^2: {full_r2:.4f} "
        f"| w range: {w.min():.2f}-{w.max():.2f}"
    )


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "dataset_33.csv")
    X = build_features(df)
    y = df["mean_uts_mpa"]
    std = df["std_uts_mpa"]

    print("Comparing 4 weighting strategies (80/20 split rs=42, 5-fold CV on train):")
    print("=" * 160)
    evaluate("A: 1/std (current)", X, y, weights_current(std))
    evaluate("B: 1/max(std, 0.3) clipped", X, y, weights_clipped(std, 0.3))
    evaluate("C: 1/sqrt(std^2 + median^2) reg.", X, y, weights_regularized(std))
    evaluate("D: uniform (no weighting)", X, y, weights_uniform(std))


if __name__ == "__main__":
    main()
