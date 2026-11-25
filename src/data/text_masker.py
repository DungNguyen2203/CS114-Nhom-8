"""Money leakage masking and residual money detection for Vietnamese listing texts."""

from __future__ import annotations

import re
from typing import Any

from src.data.text_normalizer import normalize_text


# Order matters: longer composite phrases first so sub-parts do not remain unmasked.
_MASK_PATTERNS = [
    # e.g. "9 tỷ 100 triệu", "3 tỷ 950" (do NOT eat bedroom/floor count like "5 ty 2 ngủ")
    re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:tỷ|tỉ|ty)\s*\d{1,3}\s*(?:triệu|tr)\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|mét|tháng|th|t\b|căn|lô))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:tỷ|tỉ|ty)\s*\d{1,3}\b(?!\s*(?:ngủ|pn|wc|vs|tầng|lầu|mặt|m2|m²|phòng|lô|căn|m\b|x\b))",
        re.IGNORECASE,
    ),
    # e.g. "8,x tỷ", "5.4xx tỷ", "6xxtr", "1x tỷ"
    re.compile(
        r"\b\d+(?:[.,]?\d*)?[xX]+\s*(?:tỷ|tỉ|ty|triệu|tr)\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|tháng|th|t\b|căn|lô))?",
        re.IGNORECASE,
    ),
    # e.g. "7.45 tỷ", "7,5tỷ", "6 tỷ rưỡi"
    re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:tỷ|tỉ|ty)\b(?:\s*(?:rưỡi|hơn))?(?:\s*/\s*(?:1\s*)?(?:m2|m²|mét|tháng|th|t\b|ngày|năm|căn|lô))?",
        re.IGNORECASE,
    ),
    # e.g. "500tr", "110tr/m2", "62tr/1m2", "20tr/t", "300 triệu", "12 triệu/tháng" (strict \b avoids 1 trệt / 2 Củ Chi)
    re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:triệu|tr|củ(?!\s+chi\b))\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|mét|1m|tháng|th|t\b|ngày|năm|căn|lô)|\s+1m\b)?",
        re.IGNORECASE,
    ),
    # e.g. "41.000.000đ/m2", "500.000 đ", "500 ngàn/m2", "300 nghìn/tháng", "500.000 vnđ", "500k/m2"
    re.compile(
        r"\b\d+(?:[.,]\d{3})+\s*(?:đ|đồng|vnđ|vnd)\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|mét|tháng|th|t\b|ngày|năm))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+(?:[.,]\d+)*\s*(?:ngàn|nghìn|vnđ|vnd|đồng)\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|tháng|th|t\b))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+(?:[.,]\d+)?\s*k\s*/\s*(?:m2|m²|tháng|th|ngày)\b",
        re.IGNORECASE,
    ),
    # Written money words: "một tỷ rưỡi", "hai tỷ ba", "năm trăm triệu", "tỷ rưỡi"
    re.compile(
        r"\b(?:một|hai|ba|bốn|tư|năm|sáu|bảy|bẩy|tám|chín|mười)?\s*(?:tỷ|tỉ|ty)\s+(?:rưỡi|một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s+(?:trăm|chục)\s+triệu(?:\s+rưỡi)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s+(?:tỷ|tỉ|ty|triệu)\b",
        re.IGNORECASE,
    ),
]

_RESIDUAL_MONEY_RE = re.compile(
    r"(?:\b\d+(?:[.,]?\d*)?(?:[.,]?[xX]+)?\s*(?:tỷ|tỉ|ty|triệu|tr|củ(?!\s+chi\b)|ngàn|nghìn|vnđ|vnd|đồng)\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|tháng|th|t\b))?|"
    r"\b\d+(?:[.,]\d{3})+\s*đ\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|tháng|th|t\b))?|"
    r"\b(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s+(?:tỷ|tỉ|ty|trăm\s+triệu|chục\s+triệu|triệu)\b)",
    re.IGNORECASE,
)

_MONEY_HINT_RE = re.compile(r"(?:tỷ|tỉ|ty|triệu|tr|củ|ngàn|nghìn|vnđ|vnd|đồng|k\s*/|\d\s*đ\b|xx)", re.IGNORECASE)
_CUE_X_PRICE_RE = re.compile(r"(\b(?:giá|chỉ|tầm|nhỉnh|từ)\s+)\d+(?:[.,]?\d*)?[xX]{1,3}\b", re.IGNORECASE)


def mask_money_text(text: Any, token: str = "MASK_GIA") -> str:
    """Replace all money mentions with `token` while keeping area, floor, room, and street numbers."""
    s = normalize_text(text)
    if not s:
        return ""
    if not _MONEY_HINT_RE.search(s):
        return s
    for pat in _MASK_PATTERNS:
        if _MONEY_HINT_RE.search(s):
            s = pat.sub(f" {token} ", s)
    if "x" in s or "X" in s:
        s = _CUE_X_PRICE_RE.sub(rf"\1{token} ", s)
    if token in s:
        s = re.sub(rf"(?:\b{re.escape(token)}\b\s*){{2,}}", f"{token} ", s)
        s = re.sub(r"\s+", " ", s).strip()
    return s


def find_leaked_money(text: Any) -> list[str]:
    """Detect any remaining currency/price mentions in masked text."""
    s = normalize_text(text)
    if not s:
        return []
    return [m.group(0).strip() for m in _RESIDUAL_MONEY_RE.finditer(s)]
