"""Split overlap and masked-text money leakage verification checks."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.data.text_masker import find_leaked_money


_LISTED_PRICE = re.compile(
    r"\bgiá\s*(?:bán|chào)?\s*[:=]?\s*(\d+(?:[.,]\d+)?)\b"
    r"(?![.,]\d|/\d|\s*[-–—]\s*\d|\s*(?:m[2²]?\b|mét\b|ha\b|tầng\b|phòng\b|%))", re.I,
)


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
    expected = {"train", "val", "cal", "test"}
    required = {"row_id", "split", "dup_group_id"}
    if scenario.upper() == "B":
        required.add("district_key")
    if scenario.upper() not in {"A", "B"} or not 0 <= max_leak_rate <= 1:
        raise ValueError("Expected scenario A/B and a leak rate in [0, 1].")
    if not required.issubset(manifest_df) or manifest_df[list(required)].isna().any().any():
        raise ValueError("Manifest lacks required identifiers or contains null split/group identifiers.")
    if any(manifest_df[col].astype(str).str.strip().eq("").any() for col in required):
        raise ValueError("Manifest contains empty split/group identifiers.")
    split_values = set(manifest_df["split"])
    if strict_text and split_values != expected:
        raise ValueError("Strict checks require all four train/val/cal/test splits.")
    if not manifest_df["row_id"].astype(str).is_unique:
        raise ValueError("Duplicate row_id in manifest.")
    if strict_text:
        if text_df is None or not {"row_id", "text_masked"}.issubset(text_df):
            raise ValueError("Strict checks require one masked text per manifest row_id.")
        text_ids = text_df["row_id"].astype(str)
        if (text_df[["row_id", "text_masked"]].isna().any().any() or not text_ids.is_unique
                or set(text_ids) != set(manifest_df["row_id"].astype(str))):
            raise ValueError("Strict checks require non-null masked text and exactly matching unique row_ids.")
    splits = [s for s in ["train", "val", "cal", "test"] if s in split_values]
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
    detector_rows = 0
    matched_price_rows = 0
    n_texts = 0
    sample_leaks: list[dict[str, Any]] = []
    if text_df is not None and "text_masked" in text_df.columns:
        prices = pd.to_numeric(text_df["price"], errors="raise") if "price" in text_df else [None] * len(text_df)
        for rid, txt, price in zip(text_df["row_id"].astype(str), text_df["text_masked"].astype(str), prices):
            n_texts += 1
            leaks = find_leaked_money(txt)
            detector_rows += bool(leaks)
            # Independent label-aware audit; the tabular price is never fed to the text masker.
            matched = False
            if price is not None:
                for hit in _LISTED_PRICE.finditer(txt):
                    amount = float(hit[1].replace(",", "."))
                    if abs(amount * 1e9 - price) < 1 or abs(amount - price) < 1:
                        matched = True
                        break
            matched_price_rows += matched
            if leaks or matched:
                leaked_rows += 1
                if len(sample_leaks) < 10:
                    sample_leaks.append({"row_id": rid, "leaks": leaks, "matches_asking_price": matched})
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
        "detector_rows": detector_rows,
        "matched_asking_price_rows": matched_price_rows if text_df is not None and "price" in text_df else None,
        "n_texts_checked": n_texts,
        "leak_rate": leak_rate,
        "sample_leaks": sample_leaks,
    }
