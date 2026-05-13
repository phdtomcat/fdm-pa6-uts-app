"""
Persistence for the validation log. Each entry: input parameters,
model prediction, measured value, computed error.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import VALIDATION_LOG_PATH

COLUMNS = [
    "timestamp",
    "density_pct",
    "temp_c",
    "pattern",
    "predicted_uts",
    "measured_uts",
    "abs_error",
    "rel_error_pct",
    "model_name",
    "note",
]


def _ensure_file(path: Path = VALIDATION_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)


def load_log(path: Path = VALIDATION_LOG_PATH) -> pd.DataFrame:
    _ensure_file(path)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    return df


def append_entry(
    *,
    density_pct: float,
    temp_c: float,
    pattern: str,
    predicted_uts: float,
    measured_uts: float,
    model_name: str,
    note: str = "",
    path: Path = VALIDATION_LOG_PATH,
) -> pd.DataFrame:
    _ensure_file(path)
    abs_error = abs(predicted_uts - measured_uts)
    rel_error_pct = (abs_error / measured_uts * 100.0) if measured_uts else float("nan")

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "density_pct": density_pct,
        "temp_c": temp_c,
        "pattern": pattern,
        "predicted_uts": round(predicted_uts, 3),
        "measured_uts": round(measured_uts, 3),
        "abs_error": round(abs_error, 3),
        "rel_error_pct": round(rel_error_pct, 2),
        "model_name": model_name,
        "note": note,
    }
    df = pd.read_csv(path)
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def delete_entry(index: int, path: Path = VALIDATION_LOG_PATH) -> pd.DataFrame:
    df = load_log(path)
    if 0 <= index < len(df):
        df = df.drop(index=index).reset_index(drop=True)
        df.to_csv(path, index=False)
    return df


def update_entry(
    index: int,
    *,
    density_pct: float,
    temp_c: float,
    pattern: str,
    predicted_uts: float,
    measured_uts: float,
    note: str,
    path: Path = VALIDATION_LOG_PATH,
) -> pd.DataFrame:
    df = load_log(path)
    if not (0 <= index < len(df)):
        return df

    abs_error = abs(predicted_uts - measured_uts)
    rel_error_pct = (abs_error / measured_uts * 100.0) if measured_uts else float("nan")

    df.at[index, "density_pct"] = density_pct
    df.at[index, "temp_c"] = temp_c
    df.at[index, "pattern"] = pattern
    df.at[index, "predicted_uts"] = round(predicted_uts, 3)
    df.at[index, "measured_uts"] = round(measured_uts, 3)
    df.at[index, "abs_error"] = round(abs_error, 3)
    df.at[index, "rel_error_pct"] = round(rel_error_pct, 2)
    df.at[index, "note"] = note
    df.to_csv(path, index=False)
    return df


def clear_log(path: Path = VALIDATION_LOG_PATH) -> None:
    _ensure_file(path)
    pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)
