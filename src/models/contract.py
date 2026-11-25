"""Unified prediction DataFrame contract and serialization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PREDICTION_COLUMNS: list[str] = [
    "row_id",
    "split",
    "scenario",
    "model_id",
    "seed",
    "y_true",
    "y_pred",
    "q_lo",
    "q_hi",
]


def make_prediction_df(
    row_ids: Sequence[str],
    split: str,
    scenario: str,
    model_id: str,
    seed: int,
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
    q_lo: np.ndarray | Sequence[float] | None = None,
    q_hi: np.ndarray | Sequence[float] | None = None,
) -> pd.DataFrame:
    """Create a validated prediction DataFrame conforming to the project schema."""
    n = len(row_ids)
    q_lo_arr = np.full(n, np.nan, dtype="float64") if q_lo is None else np.asarray(q_lo, dtype="float64")
    q_hi_arr = np.full(n, np.nan, dtype="float64") if q_hi is None else np.asarray(q_hi, dtype="float64")
    df = pd.DataFrame(
        {
            "row_id": [str(r) for r in row_ids],
            "split": [str(split)] * n,
            "scenario": [str(scenario)] * n,
            "model_id": [str(model_id)] * n,
            "seed": [int(seed)] * n,
            "y_true": np.asarray(y_true, dtype="float64"),
            "y_pred": np.asarray(y_pred, dtype="float64"),
            "q_lo": q_lo_arr,
            "q_hi": q_hi_arr,
        }
    )
    validate_prediction_df(df)
    return df


def validate_prediction_df(df: pd.DataFrame) -> None:
    """Validate prediction schema, non-null targets/predictions, and finiteness."""
    if list(df.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"Expected columns {PREDICTION_COLUMNS}, got {list(df.columns)}")
    if df["row_id"].isna().any():
        raise ValueError("Prediction DataFrame contains null row_id values")
    if not np.isfinite(df["y_pred"].to_numpy()).all():
        raise ValueError("Prediction DataFrame contains non-finite y_pred values")
    if not np.isfinite(df["y_true"].to_numpy()).all():
        raise ValueError("Prediction DataFrame contains non-finite y_true values")


def save_predictions(df: pd.DataFrame, path: Path) -> Path:
    validate_prediction_df(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_prediction_df(df)
    return df
