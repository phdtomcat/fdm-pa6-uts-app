"""
Constants used throughout the app: input ranges, paths, tolerance tiers from the report.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VALIDATION_LOG_PATH = DATA_DIR / "validation_log.csv"
MODEL_PATH = DATA_DIR / "model.txt"

DENSITY_MIN = 20.0
DENSITY_MAX = 80.0
DENSITY_STEP = 5.0
DENSITY_DEFAULT = 50.0

TEMP_MIN = 250.0
TEMP_MAX = 280.0
TEMP_STEP = 1.0
TEMP_DEFAULT = 265.0

PATTERNS = ["Grid", "Triangle", "Triangle-Hexa"]

TOLERANCE_TIERS = {
    "High Precision (±0.25 MPa)": 0.25,
    "Engineering (±0.50 MPa)": 0.50,
    "Acceptable (±1.00 MPa)": 1.00,
    "Upper Bound (±2.00 MPa)": 2.00,
}
