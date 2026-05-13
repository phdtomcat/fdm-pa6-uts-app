"""
DOE (Design of Experiments) matrica iz Design Expert PDF-a.
33 Run-a × 3 faktora (gustoca, temperatura, struktura).

CCD (Central Composite Design) rotabilan, alpha = sqrt(2) = 1.4142.
Faktor levels:
- Gustoca: 40, 45.86, 60, 74.14, 80 (-alfa, -1, 0, +1, +alfa)
- Temp: 220, 225.86, 240, 254.14, 260
- Struktura: A=Grid, B=Triangle, C=Triangle-Hexa

Specimen # u tensile CSV-ovima odgovara Run # u DOE-u.
"""

from __future__ import annotations

import pandas as pd

STRUCTURE_TO_PATTERN: dict[str, str] = {
    "A": "Triangle",
    "B": "Grid",
    "C": "Triangle-Hexa",
}

_DOE_ROWS: list[tuple[int, float, float, str]] = [
    (1, 60.0000, 260.000, "C"),
    (2, 45.8579, 254.142, "A"),
    (3, 60.0000, 240.000, "B"),
    (4, 74.1421, 225.858, "A"),
    (5, 45.8579, 225.858, "C"),
    (6, 60.0000, 240.000, "A"),
    (7, 45.8579, 225.858, "B"),
    (8, 40.0000, 240.000, "C"),
    (9, 74.1421, 254.142, "B"),
    (10, 60.0000, 240.000, "C"),
    (11, 74.1421, 254.142, "C"),
    (12, 40.0000, 240.000, "B"),
    (13, 45.8579, 254.142, "C"),
    (14, 45.8579, 225.858, "A"),
    (15, 60.0000, 240.000, "C"),
    (16, 60.0000, 240.000, "C"),
    (17, 74.1421, 225.858, "B"),
    (18, 74.1421, 254.142, "A"),
    (19, 60.0000, 240.000, "A"),
    (20, 60.0000, 220.000, "A"),
    (21, 60.0000, 220.000, "B"),
    (22, 60.0000, 240.000, "B"),
    (23, 40.0000, 240.000, "A"),
    (24, 80.0000, 240.000, "A"),
    (25, 60.0000, 240.000, "B"),
    (26, 80.0000, 240.000, "C"),
    (27, 60.0000, 260.000, "B"),
    (28, 74.1421, 225.858, "C"),
    (29, 80.0000, 240.000, "B"),
    (30, 45.8579, 254.142, "B"),
    (31, 60.0000, 240.000, "A"),
    (32, 60.0000, 260.000, "A"),
    (33, 60.0000, 220.000, "C"),
]


def load_doe() -> pd.DataFrame:
    df = pd.DataFrame(
        _DOE_ROWS,
        columns=["run", "density_pct", "temp_c", "structure_code"],
    )
    df["pattern"] = df["structure_code"].map(STRUCTURE_TO_PATTERN)
    return df


if __name__ == "__main__":
    doe = load_doe()
    print(doe.to_string(index=False))
    print(f"\nUkupno: {len(doe)} Run-ova")
    print(f"Po patternu: {doe['pattern'].value_counts().to_dict()}")
