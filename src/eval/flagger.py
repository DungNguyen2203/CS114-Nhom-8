"""Mispricing and text-consistency flagging (G1 model-interval + G2 parser) and E6 audit export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.text_normalizer import redact_pii


def flag_mispriced_listings(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Assign G1 (outside conformal interval) and G2 (text-vs-tabular mismatch) flags."""
    out = eval_df.copy().reset_index(drop=True)
    yt = out["y_true"].to_numpy(dtype=np.float64)
    lo = np.minimum(out["q_lo"].to_numpy(dtype=np.float64), out["q_hi"].to_numpy(dtype=np.float64))
    hi = np.maximum(out["q_lo"].to_numpy(dtype=np.float64), out["q_hi"].to_numpy(dtype=np.float64))

    g1 = np.where(yt < lo, "below_interval", np.where(yt > hi, "above_interval", "in_interval"))
    out["flag_g1"] = g1

    text_statuses = out["text_status"].astype(str) if "text_status" in out.columns else pd.Series(["clean_no_text_price"] * len(out))
    area_flags = out["area_mismatch_flag"].fillna(False).astype(bool) if "area_mismatch_flag" in out.columns else pd.Series([False] * len(out))

    g2_list: list[str] = []
    for st, af in zip(text_statuses, area_flags):
        if st == "noisy_order_of_magnitude":
            g2_list.append("order_of_magnitude_mismatch")
        elif st == "noisy_price_mismatch":
            g2_list.append("price_mismatch")
        elif st in {"noisy_extreme_area", "noisy_extreme_price", "noisy_extreme_ppm2"}:
            g2_list.append("extreme_domain_mismatch")
        elif bool(af):
            g2_list.append("area_mismatch")
        else:
            g2_list.append("none")

    out["flag_g2"] = g2_list
    out["is_flagged"] = (out["flag_g1"] != "in_interval") | (out["flag_g2"] != "none")
    return out


def export_flag_audit_samples(
    flagged_clean_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    noisy_df: pd.DataFrame | None,
    out_path: Path,
    n_per_class: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    """Export PII-redacted E6 audit sheet of `n_per_class` flagged and `n_per_class` unflagged listings."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols_needed = [
        c for c in ["row_id", "property_type_name", "province_norm", "district_norm", "area", "price", "text_price_vnd", "text_status", "area_mismatch_flag", "name_norm", "description_norm"]
        if c in meta_df.columns
    ]
    joined = flagged_clean_df.merge(meta_df[cols_needed].drop_duplicates("row_id"), on="row_id", how="left", suffixes=("", "_meta"))

    flagged_pool = joined[joined["is_flagged"]].copy()
    unflagged_pool = joined[~joined["is_flagged"]].copy()

    # Include G2 noisy_flagged rows in the flagged audit pool if available
    if noisy_df is not None and len(noisy_df) > 0:
        n_copy = noisy_df.copy()
        n_copy["flag_g1"] = "not_in_clean_eval"
        n_copy["flag_g2"] = np.where(
            n_copy["text_status"] == "noisy_order_of_magnitude",
            "order_of_magnitude_mismatch",
            np.where(n_copy["text_status"] == "noisy_price_mismatch", "price_mismatch", "extreme_domain_mismatch"),
        )
        n_copy["is_flagged"] = True
        n_copy["y_true"] = np.log(np.clip(pd.to_numeric(n_copy["price"], errors="coerce").fillna(1.0), 1.0, None))
        n_copy["y_pred"] = np.nan
        n_copy["q_lo"] = np.nan
        n_copy["q_hi"] = np.nan
        common = [c for c in flagged_pool.columns if c in n_copy.columns]
        flagged_pool = pd.concat([flagged_pool, n_copy[common]], axis=0, ignore_index=True)

    # Stratified sample across G1/G2 subtypes
    flagged_pool["audit_subtype"] = np.where(
        flagged_pool["flag_g2"] != "none",
        "G2:" + flagged_pool["flag_g2"].astype(str),
        "G1:" + flagged_pool["flag_g1"].astype(str),
    )
    n_flag = min(n_per_class, len(flagged_pool))
    if n_flag > 0:
        flag_sample = (
            flagged_pool.groupby("audit_subtype", group_keys=False)
            .apply(lambda g: g.sample(n=max(1, int(round(len(g) / len(flagged_pool) * n_flag))), random_state=seed))
            .head(n_flag)
            .reset_index(drop=True)
        )
        if len(flag_sample) < n_flag:
            rem = flagged_pool[~flagged_pool["row_id"].isin(flag_sample["row_id"])]
            flag_sample = pd.concat([flag_sample, rem.sample(n=min(len(rem), n_flag - len(flag_sample)), random_state=seed)], ignore_index=True)
    else:
        flag_sample = flagged_pool.head(0)

    n_unflag = min(n_per_class, len(unflagged_pool))
    unflag_sample = unflagged_pool.sample(n=n_unflag, random_state=seed).reset_index(drop=True) if n_unflag > 0 else unflagged_pool.head(0)
    unflag_sample["audit_subtype"] = "unflagged"

    audit_df = pd.concat([flag_sample, unflag_sample], axis=0, ignore_index=True)
    audit_df["price_vnd"] = np.exp(audit_df["y_true"])
    audit_df["pred_vnd"] = np.where(audit_df["y_pred"].notna(), np.exp(audit_df["y_pred"]), np.nan)
    audit_df["q_lo_vnd"] = np.where(audit_df["q_lo"].notna(), np.exp(audit_df["q_lo"]), np.nan)
    audit_df["q_hi_vnd"] = np.where(audit_df["q_hi"].notna(), np.exp(audit_df["q_hi"]), np.nan)
    audit_df["name_redacted"] = [redact_pii(v) for v in audit_df.get("name_norm", [""] * len(audit_df))]
    audit_df["description_redacted"] = [redact_pii(v)[:500] for v in audit_df.get("description_norm", [""] * len(audit_df))]

    keep_cols = [
        c for c in [
            "row_id", "is_flagged", "audit_subtype", "flag_g1", "flag_g2",
            "property_type_name", "province_norm", "district_norm", "area",
            "price_vnd", "text_price_vnd", "pred_vnd", "q_lo_vnd", "q_hi_vnd",
            "name_redacted", "description_redacted",
        ] if c in audit_df.columns
    ]
    export_df = audit_df[keep_cols].copy()
    export_df["human_confirmed_error_or_anomaly"] = ""
    export_df["human_reason_code"] = ""
    export_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    return {
        "n_flagged_exported": len(flag_sample),
        "n_unflagged_exported": len(unflag_sample),
        "subtype_counts": flag_sample["audit_subtype"].value_counts().to_dict() if len(flag_sample) > 0 else {},
    }
