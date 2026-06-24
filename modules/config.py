"""
Constants used throughout the app: input ranges, paths, tolerance tiers from the report.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VALIDATION_LOG_PATH = DATA_DIR / "validation_log.csv"
MODEL_PATH = DATA_DIR / "model.txt"

# Valid model domain = the envelope the LightGBM model was actually trained on
# (dataset_33.csv: densities 40-80 %, temps 220-260 C). LightGBM is tree-based
# and CANNOT extrapolate beyond this envelope - outside it the model returns a
# flat, capped value. Keep these ranges in sync with the training data.
DENSITY_MIN = 40.0
DENSITY_MAX = 80.0
DENSITY_STEP = 5.0
DENSITY_DEFAULT = 60.0

TEMP_MIN = 220.0
TEMP_MAX = 260.0
TEMP_STEP = 1.0
TEMP_DEFAULT = 240.0

PATTERNS = ["Grid", "Triangle", "Triangle-Hexa"]

TOLERANCE_TIERS = {
    "High Precision (±0.25 MPa)": 0.25,
    "Engineering (±0.50 MPa)": 0.50,
    "Acceptable (±1.00 MPa)": 1.00,
    "Upper Bound (±2.00 MPa)": 2.00,
}
