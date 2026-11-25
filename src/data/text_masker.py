"""Money leakage masking and residual money detection for Vietnamese listing texts."""

from __future__ import annotations

import re
from typing import Any

from src.data.text_normalizer import normalize_text


# Order matters: longer composite phrases first so sub-parts do not remain unmasked.
_MASK_PATTERNS = [
    # e.g. "9 tỷ 100 triệu", "3 tỷ 950"
    re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:tỷ|tỉ|ty)\s*\d{1,3}\s*(?:triệu|tr)?(?:\s*/\s*(?:1\s*)?(?:m2|m²|mét|tháng|th|t\b|căn|lô))?",
        re.IGNORECASE,
    ),
    # e.g. "8,x tỷ", "8.x ty"
    re.compile(
        r"\b\d+[.,]x\s*(?:tỷ|tỉ|ty)(?:\s*/\s*(?:1\s*)?(?:m2|m²|tháng|th|t\b|căn|lô))?",
        re.IGNORECASE,
    ),
    # e.g. "7.45 tỷ", "7,5tỷ", "6 tỷ rưỡi"
    re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:tỷ|tỉ|ty)(?:\s*(?:rưỡi|hơn))?(?:\s*/\s*(?:1\s*)?(?:m2|m²|mét|tháng|th|t\b|ngày|năm|căn|lô))?",
        re.IGNORECASE,
    ),
    # e.g. "500tr", "110tr/m2", "62tr/1m2", "20tr/t", "300 triệu", "12 triệu/tháng"
    re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:triệu|tr|củ)(?:\s*/\s*(?:1\s*)?(?:m2|m²|mét|tháng|th|t\b|ngày|năm|căn|lô))?",
        re.IGNORECASE,
    ),
    # e.g. "500 ngàn/m2", "300 nghìn/tháng", "500000 đ", "500.000 vnđ", "500k/m2"
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
    r"(?:\b\d+(?:[.,xX]\d+)?\s*(?:tỷ|tỉ|ty|triệu|tr|củ|ngàn|nghìn|vnđ|vnd|đồng)\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|tháng|th|t))?|"
    r"\b(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s+(?:tỷ|tỉ|ty|trăm\s+triệu|chục\s+triệu|triệu)\b)",
    re.IGNORECASE,
)


def mask_money_text(text: Any, token: str = "MASK_GIA") -> str:
    """Replace all money mentions with `token` while keeping area, floor, room, and street numbers."""
    s = normalize_text(text)
    if not s:
        return ""
    for pat in _MASK_PATTERNS:
        s = pat.sub(f" {token} ", s)
    # Collapse repeated adjacent mask tokens & tidy spaces
    s = re.sub(rf"(?:\b{re.escape(token)}\b\s*){{2,}}", f"{token} ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_leaked_money(text: Any) -> list[str]:
    """Detect any remaining currency/price mentions in masked text."""
    s = normalize_text(text)
    if not s:
        return []
    return [m.group(0).strip() for m in _RESIDUAL_MONEY_RE.finditer(s)]
