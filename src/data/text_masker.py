"""Money leakage masking and residual money detection for Vietnamese listing texts."""

from __future__ import annotations

import re
from typing import Any

from src.data.text_normalizer import normalize_text


# Unit spellings. "tỏi" = slang for tỷ; "ti"/"trieu" = diacritic-less tỷ/triệu ("3 ti vi", "ti le" are not money).
_TY = r"(?:tỷ|tỉ|ty|tỏi|ti(?!\s*(?:vi|le|lệ)\b))"
_TRIEU = r"(?:triệu|trieu|tr)"
# Spoken number words (hai mươi, ba trăm năm mươi, ...).
_NUMW = r"(?:một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín|mười|mươi|chục|trăm|linh|lẻ|rưỡi)"

# Order matters: longer composite phrases first so sub-parts do not remain unmasked.
_MASK_PATTERNS = [
    # Range lower bound without its own unit: "6-7 tỷ", "từ 6 đến 7 tỷ", "500 - 700tr" -> mask the "6-" part.
    # Must run first: the lookahead needs the upper bound still unmasked.
    re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:-|–|—|~|đến|tới)\s*"
        r"(?=(?:\d+(?:[.,]\d+)?\s*(?:-|–|—|~|đến|tới)\s*)*"  # chained "20 - 25 - 30 tỷ"
        r"\d+(?:[.,]?\d*)?[xX]*[.,]{0,3}\s*(?:tỷ|tỉ|ty|tỏi|ti\b|triệu|trieu|tr\b|củ(?!\s+chi\b)|k\s*/))",
        re.IGNORECASE,
    ),
    # e.g. "9 tỷ 100 triệu", "3 tỷ 950", "6.tỷ 350", "4,Tỷ 780" (do NOT eat bedroom/floor count like "5 ty 2 ngủ")
    re.compile(
        r"\b\d+(?:[.,]\d+)?[.,]{0,3}\s*" + _TY + r"\s*\d{1,3}\s*" + _TRIEU + r"\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|mét|tháng|th|t\b|căn|lô))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+(?:[.,]\d+)?[.,]{0,3}\s*" + _TY + r"\s*\d{1,3}\b(?!\s*(?:ngủ|pn|wc|vs|tầng|lầu|mặt|m2|m²|phòng|lô|căn|m\b|x\b))",
        re.IGNORECASE,
    ),
    # e.g. "8,x tỷ", "5.4xx tỷ", "6xxtr", "1x tỷ", "7.x đồng"
    re.compile(
        r"\b\d+(?:[.,]?\d*)?[.,]{0,3}[xX]+\s*(?:tỷ|tỉ|ty|tỏi|triệu|trieu|tr|đồng)\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|tháng|th|t\b|căn|lô))?",
        re.IGNORECASE,
    ),
    # e.g. "7.45 tỷ", "7,5tỷ", "4. tỷ", "5.8..tỷ", "4,.tỷ", "13.TỶ", "40, tỷ", "8.2.tỷ", "6 tỷ rưỡi", "3.5 tỏi", "5ti"
    re.compile(
        r"\b\d+(?:[.,]\d+)?[.,]{0,3}\s*" + _TY + r"\b(?:\s*(?:rưỡi|hơn))?(?:\s*/\s*(?:1\s*)?(?:m2|m²|mét|tháng|th|t\b|ngày|năm|căn|lô))?",
        re.IGNORECASE,
    ),
    # e.g. "500tr", "500.triệu", "800trieu", "45.Tr/Tháng", "110tr/m2", "62tr/1m2", "20tr/t" (strict \b avoids 1 trệt / 2 Củ Chi)
    re.compile(
        r"\b\d+(?:[.,]\d+)?[.,]{0,3}\s*(?:triệu|trieu|tr|củ(?!\s+chi\b))\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|mét|1m|tháng|th|t\b|ngày|năm|căn|lô)|\s+1m\b)?",
        re.IGNORECASE,
    ),
    # e.g. "41.000.000đ/m2", "500.000 đ", "21000 đ/m2", "500 ngàn/m2", "300 nghìn/tháng", "500.000 vnđ", "500k/m2"
    re.compile(
        r"\b\d+(?:[.,]\d{3})+\s*(?:đ|đồng|vnđ|vnd)\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|mét|tháng|th|t\b|ngày|năm))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+(?:[.,]\d+)?[.,]{0,3}\s*(?:ngàn|nghìn|vnđ|vnd)\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|tháng|th|t\b))?",
        re.IGNORECASE,
    ),
    # Colloquial "quanh 6, đồng" or explicit >=4-digit "50000 đồng" / "3000đ/m2" (never matches "1181 Đ Lê Văn Lương" or "đường 6, Đồng Mai")
    re.compile(
        r"\b((?:quanh|tầm|chỉ|khoảng|giá|nhỉnh|từ|dưới|trên)\s+)\d+(?:[.,]\d*)?[.,]{0,3}\s*đồng\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{4,}\s*(?:đồng\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|tháng|th|t\b))?|đ\b\s*/\s*(?:1\s*)?(?:m2|m²|mét|tháng|th|t\b))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+(?:[.,]\d+)?[.,]{0,3}\s*k\s*/\s*(?:m2|m²|tháng|th|ngày)\b",
        re.IGNORECASE,
    ),
    # Written money words: "một tỷ rưỡi", "hai tỷ ba", "năm trăm triệu", "tỷ rưỡi"
    re.compile(
        r"\b(?:một|hai|ba|bốn|tư|năm|sáu|bảy|bẩy|tám|chín|mười)?\s*(?:tỷ|tỉ|ty)\s+(?:rưỡi|một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín)\b",
        re.IGNORECASE,
    ),
    # Spoken compounds: "hai mươi triệu", "ba trăm năm mươi triệu", "một chục triệu", "trăm triệu"
    re.compile(r"\b" + _NUMW + r"(?:\s+" + _NUMW + r")*\s+(?:tỷ|tỉ|ty|triệu|trieu)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s+(?:trăm|chục)\s+triệu(?:\s+rưỡi)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s+(?:tỷ|tỉ|ty|triệu)\b",
        re.IGNORECASE,
    ),
]

# Words left dangling after a masked unit: "hai tỷ bốn trăm" -> "MASK_GIA trăm"; "3 củ rưỡi" -> "MASK_GIA rưỡi".
_TRAILING_WORDS = (
    r"(?:\s+(?:(?:một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín|mười)\s+)?(?:trăm|chục|rưỡi)\b"
    r"(?:\s+(?:(?:một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín)\s+)?(?:mươi|mười|trăm|chục|rưỡi|triệu|trieu|nghìn|ngàn)\b)*)+"
)

# Bare grouped amounts with no unit: "Giá 1.450.000.000", "2,190,000,000". Skipped when an area unit follows
# ("16.000.000 m2") or a phone cue precedes ("LH 091.234.567").
_BARE_GROUPED_RE = re.compile(r"(?<![\w.,])\d{1,3}(?:[.,]\d{3}){2,}(?!\d)(?![.,]\d)")
_AREA_AFTER_RE = re.compile(r"\s*(?:m2|m²|m\b|mét|ha\b|hecta|%|cm\b|km\b)", re.IGNORECASE)
_PHONE_BEFORE_RE = re.compile(r"(?:sđt|sdt|đt|dt|lh|hotline|zalo|tel|phone|liên hệ|gọi|call)\W{0,6}$", re.IGNORECASE)


def _bare_price_like(s: str, m: re.Match[str]) -> bool:
    return not (_AREA_AFTER_RE.match(s, m.end()) or _PHONE_BEFORE_RE.search(s[max(0, m.start() - 16): m.start()]))


# Residual detector. Written separately from `_MASK_PATTERNS` and deliberately broader: it flags digit+unit forms
# directly, so a form the masker misses is still caught here (it must not share the masker's blind spots).
_RESIDUAL_MONEY_RE = re.compile(
    r"(?:\b\d+(?:[.,]?\d*)?[.,]{0,3}(?:[xX]+)?\s*(?:tỷ|tỉ|ty|tỏi|triệu|trieu|tr|củ(?!\s+chi\b)|ngàn|nghìn|vnđ|vnd)\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|tháng|th|t\b))?|"
    r"\b\d+(?:[.,]?\d*)?[.,]{0,3}\s*ti(?!\s*(?:vi|le|lệ)\b)\b|"
    r"\b\d+(?:[.,]?\d*)?[.,]{0,3}[xX]+\s*đồng\b|"
    r"\b(?:quanh|tầm|chỉ|khoảng|giá|nhỉnh|từ|dưới|trên)\s+\d+(?:[.,]\d*)?[.,]{0,3}\s*đồng\b|"
    r"\b\d+(?:[.,]\d{3})+\s*(?:đ|đồng|vnđ|vnd)\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|tháng|th|t\b))?|"
    r"\b\d{4,}\s*(?:đồng\b(?:\s*/\s*(?:1\s*)?(?:m2|m²|tháng|th|t\b))?|đ\b\s*/\s*(?:1\s*)?(?:m2|m²|mét|tháng|th|t\b))|"
    r"\b(?:giá|chỉ|tầm|nhỉnh|từ|quanh|khoảng)\s+\d+(?:[.,]?\d*)?[.,]{0,3}[xX]{1,3}\b|"
    r"\b" + _NUMW + r"(?:\s+" + _NUMW + r")*\s+(?:tỷ|tỉ|ty|triệu|trieu)\b)",
    re.IGNORECASE,
)

_MONEY_HINT_RE = re.compile(
    r"(?:tỷ|tỉ|ty|tỏi|ti\b|triệu|trieu|tr|củ|ngàn|nghìn|vnđ|vnd|đồng|k\s*/|\d\s*đ\b|\d[.,]{0,3}[xX]+\b|\d[.,]\d{3}[.,]\d{3})",
    re.IGNORECASE,
)
_CUE_X_PRICE_RE = re.compile(r"(\b(?:giá|chỉ|tầm|nhỉnh|từ|quanh|khoảng)\s+)\d+(?:[.,]?\d*)?[.,]{0,3}[xX]{1,3}\b", re.IGNORECASE)


def mask_money_text(text: Any, token: str = "MASK_GIA") -> str:
    """Replace all money mentions with `token` while keeping area, floor, room, and street numbers."""
    s = normalize_text(text)
    if not s:
        return ""
    if not _MONEY_HINT_RE.search(s):
        return s
    for pat in _MASK_PATTERNS:
        if _MONEY_HINT_RE.search(s):
            if r"(\b(?:quanh|" in pat.pattern:
                s = pat.sub(rf"\1{token} ", s)
            else:
                s = pat.sub(f" {token} ", s)
    if "x" in s or "X" in s:
        s = _CUE_X_PRICE_RE.sub(rf"\1{token} ", s)
    src = s
    s = _BARE_GROUPED_RE.sub(lambda m: f" {token} " if _bare_price_like(src, m) else m.group(0), src)
    if token in s:
        s = re.sub(rf"\b{re.escape(token)}{_TRAILING_WORDS}", f" {token} ", s)
        s = re.sub(rf"(?:\b{re.escape(token)}\b\s*){{2,}}", f"{token} ", s)
        s = re.sub(r"\s+", " ", s).strip()
    return s


def find_leaked_money(text: Any, token: str = "MASK_GIA") -> list[str]:
    """Detect any remaining currency/price mentions in masked text."""
    s = normalize_text(text)
    if not s:
        return []
    hits = [m.group(0).strip() for m in _RESIDUAL_MONEY_RE.finditer(s)]
    hits += [m.group(0) for m in _BARE_GROUPED_RE.finditer(s) if _bare_price_like(s, m)]
    # Range lower bound left next to the token: "6- MASK_GIA", "từ 6 đến MASK_GIA"
    # (digit must start its own token: "80m2 - MASK_GIA", "G2- MASK_GIA" are area/block then a price, not a range)
    hits += [m.group(0) for m in re.finditer(rf"(?<![\w²])\d+(?:[.,]\d+)?\s*(?:-|–|—|~|đến|tới)\s*{re.escape(token)}", s, re.IGNORECASE)]
    # Dangling words after the token: "MASK_GIA rưỡi", "MASK_GIA trăm"
    hits += [m.group(0) for m in re.finditer(rf"\b{re.escape(token)}{_TRAILING_WORDS}", s, re.IGNORECASE)]
    return hits
