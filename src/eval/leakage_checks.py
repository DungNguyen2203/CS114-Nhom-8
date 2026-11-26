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
    max_leak_rate: float = 0.0,
) -> dict[str, Any]:
    """Verify zero overlap across train/val/cal/test for row_id, dup_group_id, and (in B) district_key.

    Raises ValueError (not `assert`, so it survives `python -O`) on any overlap, and on masked-text money leaks
    above `max_leak_rate` (fraction of rows) when `strict_text`.
    """
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

    if overlap_report["row_id_overlap"] != 0:
        raise ValueError(f"row_id overlap detected: {overlap_report}")
    if overlap_report["dup_group_id_overlap"] != 0:
        raise ValueError(f"dup_group_id overlap detected: {overlap_report}")
    if scenario.upper() == "B" and overlap_report["district_key_overlap"] != 0:
        raise ValueError(f"district_key overlap detected: {overlap_report}")

    leaked_rows = 0
    n_texts = 0
    sample_leaks: list[dict[str, Any]] = []
    if text_df is not None and "text_masked" in text_df.columns:
        for rid, txt in zip(text_df["row_id"].astype(str), text_df["text_masked"].astype(str)):
            n_texts += 1
            leaks = find_leaked_money(txt)
            if leaks:
                leaked_rows += 1
                if len(sample_leaks) < 10:
                    sample_leaks.append({"row_id": rid, "leaks": leaks})
    leak_rate = leaked_rows / max(1, n_texts)
    text_ok = leak_rate <= max_leak_rate
    if strict_text and not text_ok:
        raise ValueError(
            f"Money leakage in {leaked_rows}/{n_texts} masked texts ({leak_rate:.4%} > {max_leak_rate:.4%}): {sample_leaks[:3]}"
        )

    return {
        "scenario": scenario,
        "passed": bool(text_ok),
        **overlap_report,
        "leaked_money_rows": leaked_rows,
        "n_texts_checked": n_texts,
        "leak_rate": leak_rate,
        "sample_leaks": sample_leaks,
    }
