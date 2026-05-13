"""
Spaja peak forces (3 replike po Run-u) s DOE matricom i racuna UTS.
Output: data/dataset_99.csv (sve replike) i data/dataset_33.csv (agregirano po Run-u).

UTS [MPa] = F_peak [N] / A_cross_section [mm^2]
Default presjek 4x3 mm = 12 mm^2 (parametriziran preko CROSS_SECTION_MM2).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from doe_matrix import load_doe

CROSS_SECTION_MM2 = 4.0 * 3.0

ROOT = Path(__file__).resolve().parent.parent
PEAK_FORCES_PATH = ROOT / "data" / "peak_forces_raw.csv"
DATASET_99_PATH = ROOT / "data" / "dataset_99.csv"
DATASET_33_PATH = ROOT / "data" / "dataset_33.csv"


def build_99_point_dataset() -> pd.DataFrame:
    peaks = pd.read_csv(PEAK_FORCES_PATH)
    doe = load_doe()
    df = peaks.merge(doe, left_on="specimen", right_on="run", how="left")
    df = df.drop(columns=["run", "structure_code"])
    df["uts_mpa"] = (df["peak_force_N"] / CROSS_SECTION_MM2).round(3)
    df = df.rename(columns={"specimen": "run"})
    df = df[
        ["run", "file", "density_pct", "temp_c", "pattern", "peak_force_N", "uts_mpa"]
    ]
    df = df.sort_values(["run", "file"]).reset_index(drop=True)
    return df


def aggregate_to_33(df_99: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df_99.groupby(["run", "density_pct", "temp_c", "pattern"], as_index=False)
        .agg(
            mean_peak_force_N=("peak_force_N", "mean"),
            std_peak_force_N=("peak_force_N", "std"),
            mean_uts_mpa=("uts_mpa", "mean"),
            std_uts_mpa=("uts_mpa", "std"),
            n_replicates=("uts_mpa", "count"),
        )
        .sort_values("run")
        .reset_index(drop=True)
    )
    for col in ("mean_peak_force_N", "std_peak_force_N", "mean_uts_mpa", "std_uts_mpa"):
        grouped[col] = grouped[col].round(3)
    return grouped


def main() -> None:
    df_99 = build_99_point_dataset()
    df_33 = aggregate_to_33(df_99)

    df_99.to_csv(DATASET_99_PATH, index=False)
    df_33.to_csv(DATASET_33_PATH, index=False)

    print(f"Presjek epruvete: {CROSS_SECTION_MM2:.2f} mm^2")
    print(f"\nRaspon UTS-a: {df_99['uts_mpa'].min():.2f} - {df_99['uts_mpa'].max():.2f} MPa")
    print(f"Sredina: {df_99['uts_mpa'].mean():.2f} MPa")
    print(f"\nDataset 99 (sve replike) -> {DATASET_99_PATH}")
    print(f"Dataset 33 (mean po Run-u) -> {DATASET_33_PATH}")
    print(f"\n=== 33-Point dataset (head) ===")
    print(df_33.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
