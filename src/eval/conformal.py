"""Conformal prediction methods C0 (uncalibrated), C1 (residual), C2 (CQR), and C3 (group CQR)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the k-th smallest score, k = ceil((n+1)*(1-alpha)) clipped to [1, n] (exact order statistic)."""
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        raise ValueError("Calibration set cannot be empty.")
    # np.quantile(level=k/n, method="higher") returns rank k+1 when k/n is inexact in float; index directly instead.
    k = min(n, max(1, math.ceil((n + 1) * (1.0 - float(alpha)) - 1e-12)))
    return float(np.partition(arr, k - 1)[k - 1])


def fit_apply_conformal(
    cal_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    method: str,
    alpha: float = 0.10,
    min_group_size: int = 500,
) -> pd.DataFrame:
    """Calibrate prediction intervals on `cal_df` and apply them to `eval_df`."""
    method_u = method.upper()
    out = eval_df.copy().reset_index(drop=True)

    y_cal = cal_df["y_true"].to_numpy(dtype=np.float64)
    pred_cal = cal_df["y_pred"].to_numpy(dtype=np.float64)
    pred_eval = out["y_pred"].to_numpy(dtype=np.float64)

    has_quantiles = ("q_lo" in cal_df.columns) and ("q_hi" in cal_df.columns) and cal_df["q_lo"].notna().all()

    if method_u == "C0":
        if not has_quantiles:
            raise ValueError("C0 requires q_lo and q_hi columns.")
        lo_e = out["q_lo"].to_numpy(dtype=np.float64)
        hi_e = out["q_hi"].to_numpy(dtype=np.float64)
        out["crossed_raw"] = lo_e > hi_e
        out["crossing_rate_raw"] = float(np.mean(lo_e > hi_e))
        out["q_lo"] = np.minimum(lo_e, hi_e)
        out["q_hi"] = np.maximum(lo_e, hi_e)
        out["conformal_q"] = 0.0
        out["conformal_method"] = "C0"
        return out

    if method_u == "C1":
        res_cal = np.abs(y_cal - pred_cal)
        q_val = finite_sample_quantile(res_cal, alpha=alpha)
        out["q_lo"] = pred_eval - q_val
        out["q_hi"] = pred_eval + q_val
        out["conformal_q"] = q_val
        out["conformal_method"] = "C1"
        return out

    if method_u in {"C2", "C3"}:
        if not has_quantiles:
            raise ValueError(f"{method_u} requires q_lo and q_hi columns.")
        lo_c_raw = cal_df["q_lo"].to_numpy(dtype=np.float64)
        hi_c_raw = cal_df["q_hi"].to_numpy(dtype=np.float64)
        lo_c = np.minimum(lo_c_raw, hi_c_raw)
        hi_c = np.maximum(lo_c_raw, hi_c_raw)

        lo_e_raw = out["q_lo"].to_numpy(dtype=np.float64)
        hi_e_raw = out["q_hi"].to_numpy(dtype=np.float64)
        out["crossed_raw"] = lo_e_raw > hi_e_raw
        out["crossing_rate_raw"] = float(np.mean(lo_e_raw > hi_e_raw))
        lo_e = np.minimum(lo_e_raw, hi_e_raw)
        hi_e = np.maximum(lo_e_raw, hi_e_raw)

        scores_cal = np.maximum(lo_c - y_cal, y_cal - hi_c)
        global_q = finite_sample_quantile(scores_cal, alpha=alpha)

        if method_u == "C2":
            out["q_lo"] = lo_e - global_q
            out["q_hi"] = hi_e + global_q
            out["conformal_q"] = global_q
            out["conformal_method"] = "C2"
            return out

        # C3: group-conditional CQR by (property_type_name, pred_price_tier) with safe fallback
        pt_cal = cal_df["property_type_name"].astype(str).to_numpy()
        pt_eval = out["property_type_name"].astype(str).to_numpy()
        # Use predicted y_pred terciles from cal so test labels never leak into group definition
        edges = np.quantile(pred_cal, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        edges[0] = -np.inf
        edges[-1] = np.inf
        tier_cal = np.digitize(pred_cal, edges[1:-1])
        tier_eval = np.digitize(pred_eval, edges[1:-1])

        # Compute per-(ptype, tier) and per-ptype Q tables
        pt_tier_q: dict[tuple[str, int], float] = {}
        for pt in np.unique(pt_cal):
            for tr in [0, 1, 2]:
                mask = (pt_cal == pt) & (tier_cal == tr)
                if int(mask.sum()) >= min_group_size:
                    pt_tier_q[(str(pt), int(tr))] = finite_sample_quantile(scores_cal[mask], alpha=alpha)

        pt_only_q: dict[str, float] = {}
        for pt in np.unique(pt_cal):
            mask = pt_cal == pt
            if int(mask.sum()) >= min_group_size:
                pt_only_q[str(pt)] = finite_sample_quantile(scores_cal[mask], alpha=alpha)

        q_arr = np.empty(len(out), dtype=np.float64)
        fallback_flags = np.zeros(len(out), dtype=bool)
        for i, (pt, tr) in enumerate(zip(pt_eval, tier_eval)):
            key = (str(pt), int(tr))
            if key in pt_tier_q:
                q_arr[i] = pt_tier_q[key]
            elif str(pt) in pt_only_q:
                q_arr[i] = pt_only_q[str(pt)]
                fallback_flags[i] = True
            else:
                q_arr[i] = global_q
                fallback_flags[i] = True

        out["q_lo"] = lo_e - q_arr
        out["q_hi"] = hi_e + q_arr
        out["conformal_q"] = q_arr
        out["c3_fallback_used"] = fallback_flags
        out["conformal_method"] = "C3"
        return out

    raise ValueError(f"Unsupported conformal method: {method}")
