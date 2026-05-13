"""
Trains the Physics-Augmented LightGBM model (winner model from the report, Approach i).

Hyperparameters from the report:
  n_estimators=182, learning_rate=0.05, num_leaves=4, min_child_samples=3,
  max_depth=2, feature_fraction=0.8, reg_lambda=0.5, reg_alpha=0.1

Features: Infill Pattern OHE + Aeff (Effective Load-Bearing Area) + Dnorm (Arrhenius)
Target: mean UTS per Run

Sample weighting: uniform (no weighting).
Rationale: scripts/compare_weighting.py showed that with only 3 replicates per
condition, 1/std weights are dominated by statistical noise in the std estimate
(a single near-zero-std Run got 226x more weight than others). Uniform weighting
gave best Test R^2 (0.77 vs 0.45) and CV R^2 (0.25 vs 0.09).

Validation: 5-fold CV on the 33 mean-UTS points (80/20 train/test split first).
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.physics import (
    InfillPattern,
    effective_load_bearing_area,
    diffusion_rate_normalized,
)

DATASET_PATH = ROOT / "data" / "dataset_33.csv"
MODEL_OUT_PATH = ROOT / "data" / "model.txt"
FEATURES_OUT_PATH = ROOT / "data" / "model_features.txt"

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


def compute_sample_weights(std_values: pd.Series) -> np.ndarray:
    """
    Uniform weights (every condition counted equally).

    See module docstring for the rationale. The std_values argument is kept
    so the function signature stays stable if you want to A/B-test other
    weighting schemes via scripts/compare_weighting.py.
    """
    return np.ones(len(std_values))


def cross_validate(X: pd.DataFrame, y: pd.Series, weights: np.ndarray) -> dict:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2 = []
    fold_rmse = []
    for train_idx, test_idx in kf.split(X):
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            sample_weight=weights[train_idx],
        )
        pred = m.predict(X.iloc[test_idx])
        fold_r2.append(r2_score(y.iloc[test_idx], pred))
        fold_rmse.append(np.sqrt(mean_squared_error(y.iloc[test_idx], pred)))
    return {
        "r2_mean": float(np.mean(fold_r2)),
        "r2_std": float(np.std(fold_r2)),
        "rmse_mean": float(np.mean(fold_rmse)),
        "folds_r2": [round(x, 4) for x in fold_r2],
        "folds_rmse": [round(x, 4) for x in fold_rmse],
    }


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    X = build_features(df)
    y = df["mean_uts_mpa"]
    weights = compute_sample_weights(df["std_uts_mpa"])

    print(f"Dataset: {len(df)} stanja")
    print(f"Features: {FEATURE_NAMES}")
    print(f"Target: mean_uts_mpa (range {y.min():.2f} - {y.max():.2f})")

    print("\n=== Train/Test 80/20 split (random_state=42, kao u izvjestaju) ===")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("\n=== 5-Fold CV na trening setu ===")
    cv = cross_validate(X_train, y_train, w_train)
    print(f"R^2 (mean): {cv['r2_mean']:.4f} +- {cv['r2_std']:.4f}")
    print(f"RMSE (mean): {cv['rmse_mean']:.4f} MPa")
    print(f"Po foldu R^2: {cv['folds_r2']}")

    print("\n=== Test set evaluation ===")
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train, sample_weight=w_train)
    test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    test_abs_err = np.abs(test_pred - y_test.to_numpy())
    print(f"Test R^2: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f} MPa")
    for thr_name, thr in [
        ("High Precision (+-0.25)", 0.25),
        ("Engineering (+-0.50)", 0.50),
        ("Acceptable (+-1.00)", 1.00),
        ("Upper Bound (+-2.00)", 2.00),
    ]:
        pct = (test_abs_err <= thr).mean() * 100
        print(f"  {thr_name}: {pct:.1f}%")

    print("\n=== Final model (fit na cijelih 33 tocaka) ===")
    final_model = lgb.LGBMRegressor(**LGB_PARAMS)
    final_model.fit(X, y, sample_weight=weights)
    train_pred = final_model.predict(X)
    train_r2 = r2_score(y, train_pred)
    train_rmse = float(np.sqrt(mean_squared_error(y, train_pred)))
    print(f"Train R^2: {train_r2:.4f}, RMSE: {train_rmse:.4f} MPa")
    model = final_model

    print("\n=== Feature importance ===")
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: -x[1],
    )
    for name, imp in importances:
        print(f"  {name:30s} {imp}")

    model.booster_.save_model(str(MODEL_OUT_PATH))
    FEATURES_OUT_PATH.write_text("\n".join(FEATURE_NAMES))
    print(f"\nModel spremljen: {MODEL_OUT_PATH}")
    print(f"Lista featurea: {FEATURES_OUT_PATH}")


if __name__ == "__main__":
    main()
