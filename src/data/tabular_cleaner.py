"""Domain-based tabular cleaning, outlier flagging, and missingness indicators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TABULAR_BOUNDS: dict[str, list[float]] = {
    "frontage_width": [1.0, 100.0],
    "house_depth": [1.0, 200.0],
    "road_width": [0.5, 100.0],
    "floor_count": [1.0, 60.0],
    "bedroom_count": [1.0, 50.0],
    "bathroom_count": [1.0, 50.0],
}


def clean_tabular_domain(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Clean extreme physical outliers in tabular property attributes and record missing/outlier indicators."""
    out = df.copy()
    bounds_cfg = cfg.get("tabular_bounds", DEFAULT_TABULAR_BOUNDS)

    if "area" in out.columns:
        area_s = pd.to_numeric(out["area"], errors="coerce").astype("float64")
        min_a = float(cfg.get("min_area_m2", 5.0))
        max_a = float(cfg.get("max_area_m2", 50000.0))
        out["area"] = area_s
        out["area_was_outlier"] = (area_s.notna() & ((area_s < min_a) | (area_s > max_a))).astype("int8")

    for col, bounds in bounds_cfg.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce").astype("float64")
        else:
            s = pd.Series(np.nan, index=out.index, dtype="float64")
        was_outlier = (s.notna() & ((s < lo) | (s > hi))).astype("int8")
        cleaned = s.mask(was_outlier == 1, np.nan)
        is_missing = cleaned.isna().astype("int8")
        out[col] = cleaned
        out[f"{col}_was_outlier"] = was_outlier
        out[f"{col}_is_missing"] = is_missing

    return out
