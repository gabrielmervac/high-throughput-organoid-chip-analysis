"""Write analysis records to an Excel workbook.

Layout (as requested): a long-format main sheet where EVERY row is keyed by
Experiment / XY position / Z position / Time. Per-plane rows carry the plane
index in `Z_position`; the whole-volume aggregate row uses `Z_position = "VOLUME"`.

Extra sheets summarise per-track growth over time and per-position organoid counts.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

# Fixed leading key columns (the rest -- metric columns -- follow in discovery order).
KEY_COLS = ["Experiment", "XY_position", "Time_index", "Time_hours",
            "Z_position", "track_id", "tag", "state", "include", "notes"]


def _ordered_columns(df: pd.DataFrame) -> List[str]:
    keys = [c for c in KEY_COLS if c in df.columns]
    rest = [c for c in df.columns if c not in keys]
    return keys + rest


def export_records(rows: List[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=KEY_COLS)
    df = df[_ordered_columns(df)]

    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Organoids", index=False)

        # --- Growth summary: volume_um3 per track over time (from VOLUME rows) ---
        vol = df[df["Z_position"] == "VOLUME"] if "Z_position" in df.columns else df.iloc[0:0]
        if not vol.empty and "volume_um3" in vol.columns:
            growth = vol.pivot_table(index=["XY_position", "track_id"],
                                     columns="Time_index", values="volume_um3",
                                     aggfunc="first")
            growth.to_excel(writer, sheet_name="Growth_volume_um3")

        # --- Count summary: organoids per position over time ---
        if not vol.empty:
            counts = vol.groupby(["XY_position", "Time_index"])["track_id"].nunique()
            counts = counts.unstack("Time_index")
            counts.to_excel(writer, sheet_name="Counts_per_position")

    return path


if __name__ == "__main__":
    # Tiny self-test with synthetic rows.
    demo = [
        {"Experiment": "demo", "XY_position": 0, "Time_index": 0, "Time_hours": 0.0,
         "Z_position": 0, "track_id": 1, "tag": "", "state": "", "include": True,
         "notes": "", "area_px": 100, "FITC_mean": 12.0},
        {"Experiment": "demo", "XY_position": 0, "Time_index": 0, "Time_hours": 0.0,
         "Z_position": "VOLUME", "track_id": 1, "tag": "", "state": "", "include": True,
         "notes": "", "volume_vox": 500, "volume_um3": 42.0},
    ]
    out = export_records(demo, Path(r"D:\Yiyu\Organoid ID Finetuned\outputs\_demo.xlsx"))
    print("wrote", out)
