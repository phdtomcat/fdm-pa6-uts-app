"""
Ekstrahira peak Force po uzorku iz sirovih tensile CSV-ova.
Format: svaki uzorak ima 3 stupca (Time, Force, Stroke), redak 1 je broj uzorka,
redak 2 su nazivi, redak 3 jedinice, redci 4+ su podaci. Decimalni separator: zarez.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def extract_peak_forces(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        header=None,
        skiprows=3,
        decimal=",",
        low_memory=False,
    )
    n_specimens = df.shape[1] // 3
    rows = []
    for i in range(n_specimens):
        force_col = df.iloc[:, i * 3 + 1].astype(float)
        peak = force_col.max(skipna=True)
        rows.append({
            "file": csv_path.name,
            "specimen": i + 1,
            "peak_force_N": round(float(peak), 3),
        })
    return pd.DataFrame(rows)


def main(downloads_dir: Path) -> None:
    files = [
        "Plan pokusa-6.11.2024.csv",
        "Plan pokusa 15.11.2024 raw data.csv",
        "Plan pokusa 26.11.2024.csv",
    ]
    all_rows = []
    for fname in files:
        path = downloads_dir / fname
        if not path.exists():
            print(f"MISSING: {path}", file=sys.stderr)
            continue
        df = extract_peak_forces(path)
        all_rows.append(df)
        print(f"\n===== {fname} =====")
        print(df.to_string(index=False))

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        out = Path(__file__).resolve().parent.parent / "data" / "peak_forces_raw.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out, index=False)
        print(f"\nSpremljeno: {out}")
        print(f"Ukupno uzoraka: {len(combined)}")


if __name__ == "__main__":
    main(Path("C:/Users/ig/Downloads"))
