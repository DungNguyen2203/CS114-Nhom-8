"""Train-only tabular feature extraction with rare-category grouping and text signal flags."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


CATEGORICAL_COLS: list[str] = [
    "property_type_name",
    "province_norm",
    "district_norm",
    "ward_norm",
    "street_norm",
    "project_norm",
    "house_direction_norm",
    "balcony_direction_norm",
]

NUMERIC_BASE_COLS: list[str] = [
    "area",
    "floor_count",
    "frontage_width",
    "house_depth",
    "road_width",
    "bedroom_count",
    "bathroom_count",
]

FLAG_COLS: list[str] = [
    "floor_count_is_missing",
    "frontage_width_is_missing",
    "house_depth_is_missing",
    "road_width_is_missing",
    "bedroom_count_is_missing",
    "bathroom_count_is_missing",
    "area_was_outlier",
    "frontage_width_was_outlier",
    "house_depth_was_outlier",
    "road_width_was_outlier",
    "floor_count_was_outlier",
    "bedroom_count_was_outlier",
    "bathroom_count_was_outlier",
]

_KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {
    "has_so_hong": re.compile(r"\b(?:sổ\s*hồng|sổ\s*đỏ|shr|pháp\s*lý\s*chuẩn)\b", re.IGNORECASE),
    "has_hem_xe_hoi": re.compile(r"\b(?:hẻm\s*(?:xe\s*hơi|ô\s*tô)|ngõ\s*ô\s*tô|ô\s*tô\s*(?:vào|tránh|đỗ))\b", re.IGNORECASE),
    "has_mat_tien": re.compile(r"\b(?:mặt\s*tiền|mặt\s*phố|mtkd)\b", re.IGNORECASE),
    "has_thang_may": re.compile(r"\bthang\s*máy\b", re.IGNORECASE),
    "has_full_noi_that": re.compile(r"\b(?:full\s*nội\s*thất|nội\s*thất\s*đầy\s*đủ|cao\s*cấp)\b", re.IGNORECASE),
    "is_ban_gap": re.compile(r"\b(?:bán\s*gấp|ngộp|kẹt\s*tiền|cắt\s*lỗ)\b", re.IGNORECASE),
    "is_chinh_chu": re.compile(r"\bchính\s*chủ\b", re.IGNORECASE),
}


class TabularFeatureExtractor:
    """Fit rare category vocabularies strictly on train and transform tabular + keyword features."""

    def __init__(self, min_freq: int = 15) -> None:
        self.min_freq = int(min_freq)
        self.allowed_categories: dict[str, set[str]] = {}
        self.cat_cols: list[str] = list(CATEGORICAL_COLS)
        self.feature_names_: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> "TabularFeatureExtractor":
        self.allowed_categories = {}
        for col in self.cat_cols:
            if col in train_df.columns:
                vals = train_df[col].fillna("__MISSING__").astype(str)
                counts = vals.value_counts()
                allowed = set(counts[counts >= self.min_freq].index.tolist())
                allowed.add("__MISSING__")
                self.allowed_categories[col] = allowed
            else:
                self.allowed_categories[col] = {"__MISSING__"}
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        out = pd.DataFrame(index=df.index)

        for col in self.cat_cols:
            if col in df.columns:
                raw = df[col].fillna("__MISSING__").astype(str)
            else:
                raw = pd.Series("__MISSING__", index=df.index, dtype="object")
            allowed = self.allowed_categories.get(col, {"__MISSING__"})
            mapped = raw.where(raw.isin(allowed), "__RARE__")
            out[col] = mapped.astype(str)

        for col in NUMERIC_BASE_COLS:
            if col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            else:
                out[col] = np.nan

        out["log_area"] = np.log(np.clip(out["area"].to_numpy(), 1e-3, None))

        for col in FLAG_COLS:
            if col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")
            else:
                out[col] = np.int8(0)

        texts = (
            df["text_masked"].fillna("").astype(str).tolist()
            if "text_masked" in df.columns
            else [""] * len(df)
        )
        for kw_col, pat in _KEYWORD_PATTERNS.items():
            out[kw_col] = np.fromiter((1 if pat.search(t) else 0 for t in texts), dtype=np.int8, count=len(texts))
        out["text_char_len"] = np.fromiter((len(t) for t in texts), dtype=np.int32, count=len(texts))

        self.feature_names_ = list(out.columns)
        return out, list(self.cat_cols)

    def fit_transform(self, train_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        self.fit(train_df)
        return self.transform(train_df)
