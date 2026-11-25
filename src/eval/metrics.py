"""Unified regression, conformal interval, subgroup, and cluster-bootstrap metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute log-space and VND relative accuracy metrics."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    abs_log = np.abs(yp - yt)

    p_true = np.exp(yt)
    p_pred = np.exp(yp)
    abs_vnd = np.abs(p_pred - p_true)
    rel_err = abs_vnd / np.clip(p_true, 1.0, None)

    return {
        "n": int(len(yt)),
        "MAE_log": float(np.mean(abs_log)),
        "RMSE_log": float(np.sqrt(np.mean((yp - yt) ** 2))),
        "MdAPE_pct": float(100.0 * np.median(rel_err)),
        "within_10_pct": float(100.0 * np.mean(rel_err <= 0.10 + 1e-12)),
        "within_20_pct": float(100.0 * np.mean(rel_err <= 0.20 + 1e-12)),
        "WAPE_vnd_pct": float(100.0 * np.sum(abs_vnd) / np.clip(np.sum(p_true), 1.0, None)),
    }


def compute_interval_metrics(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    alpha: float = 0.10,
) -> dict[str, float]:
    """Compute conformal coverage, width, crossing rate, and Winkler interval score."""
    yt = np.asarray(y_true, dtype=np.float64)
    lo_raw = np.asarray(q_lo, dtype=np.float64)
    hi_raw = np.asarray(q_hi, dtype=np.float64)

    crossing_rate_raw = float(np.mean(lo_raw > hi_raw))
    lo = np.minimum(lo_raw, hi_raw)
    hi = np.maximum(lo_raw, hi_raw)

    covered = (yt >= lo) & (yt <= hi)
    width = hi - lo
    winkler = (
        width
        + (2.0 / alpha) * np.maximum(0.0, lo - yt)
        + (2.0 / alpha) * np.maximum(0.0, yt - hi)
    )
    return {
        "coverage": float(np.mean(covered)),
        "mean_width_log": float(np.mean(width)),
        "median_ratio_hi_lo": float(np.median(np.exp(width))),
        "interval_score": float(np.mean(winkler)),
        "crossing_rate_raw": crossing_rate_raw,
    }


def _assign_region_bucket(province_series: pd.Series) -> pd.Series:
    s = province_series.astype(str)
    return pd.Series(
        np.where(s == "Hà Nội", "Hà Nội", np.where(s == "Hồ Chí Minh", "TP.HCM", "Các tỉnh khác")),
        index=province_series.index,
    )


def evaluate_predictions_df(
    pred_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    alpha: float = 0.10,
    min_eval_group_size: int = 500,
) -> dict[str, Any]:
    """Compute overall and subgroup metrics from standardized prediction DataFrame."""
    meta_cols = [
        c for c in ["row_id", "property_type_name", "province_norm", "district_key", "month"]
        if c in meta_df.columns and (c == "row_id" or c not in pred_df.columns)
    ]
    merged = pred_df.merge(meta_df[meta_cols].drop_duplicates("row_id"), on="row_id", how="left") if len(meta_cols) > 1 else pred_df.copy()

    overall = compute_point_metrics(merged["y_true"].to_numpy(), merged["y_pred"].to_numpy())
    has_intervals = ("q_lo" in merged.columns) and ("q_hi" in merged.columns) and merged["q_lo"].notna().all() and merged["q_hi"].notna().all()
    if has_intervals:
        overall.update(
            compute_interval_metrics(
                merged["y_true"].to_numpy(),
                merged["q_lo"].to_numpy(),
                merged["q_hi"].to_numpy(),
                alpha=alpha,
            )
        )

    if "province_norm" in merged.columns:
        merged["region_bucket"] = _assign_region_bucket(merged["province_norm"])

    # Assign 4 price tiers strictly from train y_true quantiles when available
    if train_df is not None and len(train_df) >= 4:
        qs = np.quantile(train_df["y_true"].to_numpy(dtype=np.float64), [0.0, 0.25, 0.50, 0.75, 1.0])
    else:
        qs = np.quantile(merged["y_true"].to_numpy(dtype=np.float64), [0.0, 0.25, 0.50, 0.75, 1.0])
    qs[0] = -np.inf
    qs[-1] = np.inf
    merged["price_tier"] = pd.cut(
        merged["y_true"],
        bins=np.unique(qs) if len(np.unique(qs)) >= 2 else [-np.inf, np.inf],
        labels=False,
        include_lowest=True,
    ).astype(str)

    group_rows: list[dict[str, Any]] = []
    for dim in ["region_bucket", "property_type_name", "province_norm", "price_tier", "month"]:
        if dim not in merged.columns:
            continue
        for gval, sub in merged.groupby(dim):
            row = {"group_dim": dim, "group_val": str(gval), "eligible_min_n": len(sub) >= min_eval_group_size}
            row.update(compute_point_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()))
            if has_intervals:
                row.update(compute_interval_metrics(sub["y_true"].to_numpy(), sub["q_lo"].to_numpy(), sub["q_hi"].to_numpy(), alpha=alpha))
            group_rows.append(row)

    group_df = pd.DataFrame(group_rows)
    if has_intervals and len(group_df) > 0:
        elig = group_df[group_df["eligible_min_n"]]
        if len(elig) > 0:
            target_cov = 1.0 - alpha
            overall["worst_group_coverage"] = float(elig["coverage"].min())
            overall["max_coverage_gap"] = float(np.max(np.abs(elig["coverage"] - target_cov)))

    return {"overall": overall, "subgroups": group_df}


def cluster_bootstrap_mae_diff(
    pred_a_df: pd.DataFrame,
    pred_b_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """District-cluster bootstrap 95% CI for `MAE_log(A) - MAE_log(B)`."""
    merged = (
        pred_a_df[["row_id", "y_true", "y_pred"]]
        .rename(columns={"y_pred": "pred_a"})
        .merge(pred_b_df[["row_id", "y_pred"]].rename(columns={"y_pred": "pred_b"}), on="row_id")
        .merge(meta_df[["row_id", "district_key"]].drop_duplicates("row_id"), on="row_id", how="left")
    )
    merged["district_key"] = merged["district_key"].fillna("__NONE__")
    merged["err_a"] = np.abs(merged["pred_a"] - merged["y_true"])
    merged["err_b"] = np.abs(merged["pred_b"] - merged["y_true"])

    grp = merged.groupby("district_key").agg(
        sum_a=("err_a", "sum"),
        sum_b=("err_b", "sum"),
        n=("row_id", "count"),
    )
    sa = grp["sum_a"].to_numpy(dtype=np.float64)
    sb = grp["sum_b"].to_numpy(dtype=np.float64)
    sn = grp["n"].to_numpy(dtype=np.float64)
    k = len(grp)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, k, size=k)
        tot_n = np.sum(sn[idx])
        diffs[b] = (np.sum(sa[idx]) - np.sum(sb[idx])) / max(1.0, tot_n)

    point_diff = float(np.mean(merged["err_a"] - merged["err_b"]))
    return {
        "mae_diff_mean": point_diff,
        "ci_95_low": float(np.percentile(diffs, 2.5)),
        "ci_95_high": float(np.percentile(diffs, 97.5)),
    }
