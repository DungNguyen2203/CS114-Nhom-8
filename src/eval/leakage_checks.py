"""Split overlap and masked-text money leakage verification checks."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.text_masker import find_leaked_money


def verify_split_leakage(
    manifest_df: pd.DataFrame,
    text_df: pd.DataFrame | None = None,
    scenario: str = "A",
    strict_text: bool = True,
) -> dict[str, Any]:
    """Verify zero overlap across train/val/cal/test for row_id, dup_group_id, and (in B) district_key."""
    splits = [s for s in ["train", "val", "cal", "test"] if s in set(manifest_df["split"].astype(str))]
    overlap_report: dict[str, int] = {
        "row_id_overlap": 0,
        "dup_group_id_overlap": 0,
        "district_key_overlap": 0,
    }

    for i, s1 in enumerate(splits):
        df1 = manifest_df[manifest_df["split"] == s1]
        r1 = set(df1["row_id"].astype(str))
        g1 = set(df1["dup_group_id"].astype(str)) if "dup_group_id" in df1.columns else set()
        d1 = set(df1["district_key"].astype(str)) if "district_key" in df1.columns else set()

        for s2 in splits[i + 1 :]:
            df2 = manifest_df[manifest_df["split"] == s2]
            r2 = set(df2["row_id"].astype(str))
            g2 = set(df2["dup_group_id"].astype(str)) if "dup_group_id" in df2.columns else set()
            d2 = set(df2["district_key"].astype(str)) if "district_key" in df2.columns else set()

            overlap_report["row_id_overlap"] += len(r1.intersection(r2))
            overlap_report["dup_group_id_overlap"] += len(g1.intersection(g2))
            if scenario.upper() == "B":
                overlap_report["district_key_overlap"] += len(d1.intersection(d2))

    assert overlap_report["row_id_overlap"] == 0, f"row_id overlap detected: {overlap_report}"
    assert overlap_report["dup_group_id_overlap"] == 0, f"dup_group_id overlap detected: {overlap_report}"
    if scenario.upper() == "B":
        assert overlap_report["district_key_overlap"] == 0, f"district_key overlap detected: {overlap_report}"

    leaked_rows = 0
    sample_leaks: list[dict[str, Any]] = []
    if text_df is not None and "text_masked" in text_df.columns:
        for rid, txt in zip(text_df["row_id"].astype(str), text_df["text_masked"].astype(str)):
            leaks = find_leaked_money(txt)
            if leaks:
                leaked_rows += 1
                if len(sample_leaks) < 10:
                    sample_leaks.append({"row_id": rid, "leaks": leaks})
        if strict_text:
            assert leaked_rows == 0, f"Money leakage found in {leaked_rows} masked texts: {sample_leaks[:3]}"

    return {
        "scenario": scenario,
        "passed": True,
        **overlap_report,
        "leaked_money_rows": leaked_rows,
        "sample_leaks": sample_leaks,
    }
