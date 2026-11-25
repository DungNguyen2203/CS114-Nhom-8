"""Scenario A (temporal) and Scenario B (unseen district) split builders and budget samplers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_scenario_a_split(df: pd.DataFrame, split_cfg: dict[str, Any]) -> pd.DataFrame:
    """Build chronological split keeping the earliest record per duplicate group."""
    work = df.copy()
    if "month" not in work.columns:
        work["month"] = work["published_at"].astype(str).str.slice(0, 7)
    if "y_true" not in work.columns:
        work["y_true"] = np.log(pd.to_numeric(work["price"], errors="coerce"))

    work = work.sort_values(["published_at", "row_id"], kind="mergesort")
    work = work.drop_duplicates(subset=["dup_group_id"], keep="first").reset_index(drop=True)

    month_to_split: dict[str, str] = {}
    for m in split_cfg.get("train_months", []):
        month_to_split[str(m)] = "train"
    for m in split_cfg.get("val_months", []):
        month_to_split[str(m)] = "val"
    for m in split_cfg.get("cal_months", []):
        month_to_split[str(m)] = "cal"
    for m in split_cfg.get("test_months", []):
        month_to_split[str(m)] = "test"

    work["split"] = work["month"].map(month_to_split)
    work = work[work["split"].notna()].copy()

    stale_months = set(str(m) for m in split_cfg.get("stale_cal_months", []))
    work["is_stale_cal_candidate"] = work["month"].isin(stale_months)

    cols = [
        "row_id",
        "split",
        "dup_group_id",
        "district_key",
        "province_norm",
        "property_type_name",
        "month",
        "y_true",
        "is_stale_cal_candidate",
    ]
    return work[[c for c in cols if c in work.columns]].reset_index(drop=True)


def build_scenario_b_split(df: pd.DataFrame, split_cfg: dict[str, Any], seed: int = 42) -> pd.DataFrame:
    """Build unseen-district split ensuring zero overlap on `district_key` and `dup_group_id`."""
    work = df.copy()
    if "month" not in work.columns:
        work["month"] = work["published_at"].astype(str).str.slice(0, 7)
    if "y_true" not in work.columns:
        work["y_true"] = np.log(pd.to_numeric(work["price"], errors="coerce"))

    # Keep earliest listing per duplicate cluster so no duplicate spans splits or districts
    work = work.sort_values(["published_at", "row_id"], kind="mergesort")
    work = work.drop_duplicates(subset=["dup_group_id"], keep="first").reset_index(drop=True)

    if "district_norm" in work.columns:
        work = work[work["district_norm"] != "__MISSING__"].copy()
    work = work[~work["district_key"].astype(str).str.endswith("|__MISSING__")].copy()

    ratios = [float(x) for x in split_cfg.get("ratios", [0.70, 0.10, 0.10, 0.10])]
    split_names = ["train", "val", "cal", "test"]
    min_group_rows = int(split_cfg.get("min_group_rows", 1))

    grp_counts = work.groupby("district_key").size()
    eligible_groups = grp_counts[grp_counts >= min_group_rows].index.tolist()
    work = work[work["district_key"].isin(eligible_groups)].copy()

    # Stratify districts by log size quantile, shuffle within stratum, and assign proportionally
    rng = np.random.default_rng(seed)
    sizes = grp_counts.loc[eligible_groups].sort_values(ascending=False)
    ordered_groups: list[str] = []
    n_strata = min(5, max(1, len(sizes) // 4))
    strata = np.array_split(sizes.index.to_numpy(), n_strata)
    for st in strata:
        arr = st.copy()
        rng.shuffle(arr)
        ordered_groups.extend(arr.tolist())

    # Balanced greedy assignment tracking group count targets so even small sets populate all 4 splits
    n_groups = len(ordered_groups)
    target_groups = [max(1, int(round(r * n_groups))) if n_groups >= 4 else 1 for r in ratios]
    group_to_split: dict[str, str] = {}

    if n_groups >= 4:
        # Guarantee at least 1 group per split first (prioritizing val/cal/test with diverse sizes)
        assigned_counts = {s: 0 for s in split_names}
        assigned_rows = {s: 0 for s in split_names}
        total_rows = int(sizes.sum())
        target_rows = {s: r * total_rows for s, r in zip(split_names, ratios)}

        # Round-robin seed 1 group to val, cal, test, train first if small, else fill proportionally
        remaining = list(ordered_groups)
        for s in ["test", "cal", "val", "train"]:
            if remaining:
                g = remaining.pop(0)
                group_to_split[g] = s
                assigned_counts[s] += 1
                assigned_rows[s] += int(sizes[g])

        for g in remaining:
            # Pick split with greatest deficit in group ratio & row ratio
            best_s = max(
                split_names,
                key=lambda s: (
                    (target_groups[split_names.index(s)] - assigned_counts[s]) / max(1, target_groups[split_names.index(s)]),
                    (target_rows[s] - assigned_rows[s]) / max(1.0, target_rows[s]),
                ),
            )
            group_to_split[g] = best_s
            assigned_counts[best_s] += 1
            assigned_rows[best_s] += int(sizes[g])
    else:
        for i, g in enumerate(ordered_groups):
            group_to_split[g] = split_names[i % len(split_names)]

    work["split"] = work["district_key"].map(group_to_split)
    cols = [
        "row_id",
        "split",
        "dup_group_id",
        "district_key",
        "province_norm",
        "property_type_name",
        "month",
        "y_true",
    ]
    return work[[c for c in cols if c in work.columns]].reset_index(drop=True)


def stratified_sample_df(
    df: pd.DataFrame,
    group_cols: str | list[str],
    n_total: int,
    seed: int = 42,
    id_col: str | None = "row_id",
) -> pd.DataFrame:
    """Proportionally sample up to n_total rows across strata while preserving all columns."""
    if n_total <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    if len(df) <= n_total:
        return df.reset_index(drop=True).copy()

    cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    frac = n_total / len(df)
    parts: list[pd.DataFrame] = []
    for _, grp in df.groupby(cols, dropna=False, sort=False):
        target = min(len(grp), max(1, int(round(len(grp) * frac))))
        if len(grp) <= 1 or target >= len(grp):
            parts.append(grp)
        else:
            parts.append(grp.sample(n=target, random_state=seed))

    sampled = pd.concat(parts, axis=0)
    if len(sampled) > n_total:
        sampled = sampled.sample(n=n_total, random_state=seed)
    elif len(sampled) < n_total:
        if id_col and id_col in df.columns and df[id_col].is_unique:
            rem = df[~df[id_col].isin(sampled[id_col])]
        else:
            rem = df.loc[~df.index.isin(sampled.index)]
        if len(rem) > 0:
            sampled = pd.concat(
                [sampled, rem.sample(n=min(len(rem), n_total - len(sampled)), random_state=seed)],
                axis=0,
            )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def apply_stratified_budget(
    manifest_df: pd.DataFrame,
    sampling_cfg: dict[str, Any],
    seed: int = 42,
) -> pd.DataFrame:
    """Subsample each split up to its configured row budget stratified by province bucket, property type, and month."""
    budgets = {
        "train": sampling_cfg.get("train_max_rows"),
        "val": sampling_cfg.get("val_max_rows"),
        "cal": sampling_cfg.get("cal_max_rows"),
        "test": sampling_cfg.get("test_max_rows"),
    }
    parts: list[pd.DataFrame] = []
    for split_name in ["train", "val", "cal", "test"]:
        sub = manifest_df[manifest_df["split"] == split_name]
        max_rows = budgets.get(split_name)
        if max_rows is None or len(sub) <= int(max_rows):
            parts.append(sub)
            continue
        max_rows = int(max_rows)
        prov_bkt = np.where(
            sub["province_norm"].isin(["Hà Nội", "Hồ Chí Minh"]),
            sub["province_norm"],
            "Other",
        )
        strata_key = (
            pd.Series(prov_bkt, index=sub.index).astype(str)
            + "|"
            + sub["property_type_name"].astype(str)
            + "|"
            + sub["month"].astype(str)
        )
        sampled = stratified_sample_df(
            sub.assign(_strata=strata_key),
            "_strata",
            n_total=max_rows,
            seed=seed,
            id_col="row_id",
        )
        if "_strata" in sampled.columns:
            sampled = sampled.drop(columns=["_strata"])
        parts.append(sampled)

    return pd.concat(parts, axis=0).reset_index(drop=True)
