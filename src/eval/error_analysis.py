"""Error bucket sampling and text-signal residual correlation analysis."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.text_normalizer import redact_pii


DEFAULT_SIGNAL_KEYWORDS: list[str] = [
    "bán gấp",
    "ngộp",
    "kẹt tiền",
    "chính chủ",
    "sổ hồng",
    "hẻm xe hơi",
    "mặt tiền",
]


def sample_error_buckets(
    pred_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    n_per_bucket: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample good, medium, and large absolute-error listings with PII redacted."""
    cols = [
        c for c in ["row_id", "property_type_name", "province_norm", "district_norm", "area", "name_norm", "description_norm", "text_masked"]
        if c in meta_df.columns and (c == "row_id" or c not in pred_df.columns)
    ]
    merged = pred_df.merge(meta_df[cols].drop_duplicates("row_id"), on="row_id", how="left") if len(cols) > 1 else pred_df.copy()
    merged["abs_log_err"] = np.abs(merged["y_pred"] - merged["y_true"])
    merged["signed_log_err"] = merged["y_true"] - merged["y_pred"]

    q33, q67 = np.quantile(merged["abs_log_err"], [0.33, 0.67])
    merged["error_bucket"] = np.where(
        merged["abs_log_err"] <= q33,
        "good",
        np.where(merged["abs_log_err"] <= q67, "medium", "large"),
    )

    samples = []
    for bucket in ["good", "medium", "large"]:
        sub = merged[merged["error_bucket"] == bucket]
        if len(sub) > 0:
            samples.append(sub.sample(n=min(n_per_bucket, len(sub)), random_state=seed))
    out = pd.concat(samples, axis=0, ignore_index=True) if samples else merged.head(0)
    out["price_vnd"] = np.exp(out["y_true"])
    out["pred_vnd"] = np.exp(out["y_pred"])
    if "name_norm" in out.columns:
        out["name_redacted"] = [redact_pii(v) for v in out["name_norm"]]
    if "description_norm" in out.columns:
        out["description_redacted"] = [redact_pii(v)[:500] for v in out["description_norm"]]
    if "text_masked" in out.columns:
        out["text_masked_redacted"] = [redact_pii(v)[:500] for v in out["text_masked"]]
    drop_raw = [c for c in ["name_norm", "description_norm", "text_masked"] if c in out.columns]
    return out.drop(columns=drop_raw)


def analyze_residual_keywords(
    pred_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    keywords: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Measure signed residual correlation (not causal) for domain urgency/quality keywords."""
    kws = list(keywords or DEFAULT_SIGNAL_KEYWORDS)
    if "text_masked" in pred_df.columns:
        work = pred_df.copy()
    else:
        work = pred_df.merge(meta_df[["row_id", "text_masked"]].drop_duplicates("row_id"), on="row_id", how="left")

    texts = work["text_masked"].fillna("").astype(str).tolist()
    res = (work["y_true"] - work["y_pred"]).to_numpy(dtype=np.float64)
    abs_res = np.abs(res)

    rows = []
    for kw in kws:
        pat = re.compile(re.escape(kw), re.IGNORECASE)
        mask = np.fromiter((bool(pat.search(t)) for t in texts), dtype=bool, count=len(texts))
        cnt = int(mask.sum())
        rows.append(
            {
                "keyword": kw,
                "count_present": cnt,
                "pct_present": 100.0 * cnt / max(1, len(texts)),
                "mean_signed_residual_present": float(np.mean(res[mask])) if cnt > 0 else np.nan,
                "mean_signed_residual_absent": float(np.mean(res[~mask])) if cnt < len(texts) else np.nan,
                "mean_abs_err_present": float(np.mean(abs_res[mask])) if cnt > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)
