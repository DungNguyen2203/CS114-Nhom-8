"""Independent rule-based Vietnamese real-estate price/area parser and label reconciler.

IMPORTANT: `parse_listing_text` NEVER takes tabular `price` as input so there is zero label leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from src.data.text_normalizer import normalize_text


@dataclass
class MoneyMention:
    raw: str
    value_vnd: float
    role: str  # total_sale | unit_per_m2 | rental_income | deposit_or_discount | other
    is_approx: bool = False
    confidence: str = "high"


@dataclass
class ParsedListing:
    tx_type: str  # sale | rent | unknown
    is_bundle: bool
    is_negotiable: bool
    sale_price_vnd: float | None
    unit_price_per_m2_vnd: float | None
    area_m2: float | None
    is_approx: bool
    confidence: str
    mentions: list[MoneyMention] = field(default_factory=list)


@dataclass
class ReconcileResult:
    bucket: str  # clean | noisy | rental | missing_or_negotiable | multi_or_unknown
    status: str
    area_mismatch_flag: bool
    price_ratio: float | None


_APPROX_CUE_RE = re.compile(r"\b(?:nhỉnh|hơn|tầm|khoảng|gần|chỉ\s+từ|từ)\b|~", re.IGNORECASE)
_NEGOTIABLE_RE = re.compile(r"\b(?:giá\s+thỏa\s+thuận|thỏa\s+thuận|liên\s+hệ(?:\s+để\s+biết\s+giá)?)\b", re.IGNORECASE)
_SALE_CUE_RE = re.compile(
    r"\b(?:cần\s+bán|chính\s+chủ\s+bán|bán\s+gấp|bán\s+nhanh|bán\s+nhà|bán\s+đất|bán\s+căn\s+hộ|"
    r"bán\s+biệt\s+thự|bán\s+shophouse|giá\s+bán|chào\s+bán|sổ\s+đỏ|sổ\s+hồng|sang\s+tên|bán)\b",
    re.IGNORECASE,
)
_PURE_RENT_CUE_RE = re.compile(
    r"\b(?:cho\s+thuê|cần\s+thuê|thuê\s+gấp|sang\s+nhượng\s+mặt\s+bằng)\b",
    re.IGNORECASE,
)
_EMBEDDED_RENT_CTX_RE = re.compile(
    r"(?:hđt|hdt|hợp\s+đồng\s+thuê|đang\s+cho\s+thuê|cho\s+thuê\s+sẵn|cho\s+thuê\s+lại|"
    r"dòng\s+tiền(?:\s+cho\s+thuê)?|thu\s+nhập(?:\s+thuê)?|khai\s+thác\s+thuê|"
    r"lợi\s+nhuận|cam\s+kết(?:\s+lợi\s+nhuận)?|\bln)\s*[:\-]?\s*$",
    re.IGNORECASE,
)
_RENT_UNIT_RE = re.compile(r"(?:/\s*(?:tháng|th|t|ngày|năm)\b|\b(?:tháng|th)\b)", re.IGNORECASE)
_UNIT_M2_RE = re.compile(r"(?:/\s*(?:\d*\s*)?(?:m2|m²|mét(?:\s*vuông)?|m)\b|\b1m\b)", re.IGNORECASE)
_DISCOUNT_OR_DEPOSIT_CTX_RE = re.compile(
    r"(?:giảm(?:\s+(?:ngay|sốc|mạnh))?|bớt|cắt\s+lỗ|chiết\s+khấu|lì\s+xì|tặng|"
    r"đặt\s+cọc|cọc|trả\s+trước|chỉ\s+cần|thanh\s+toán(?:\s+trước)?|vốn(?:\s+tự\s+có)?|hỗ\s+trợ\s+vay)\s*[:\-]?\s*$",
    re.IGNORECASE,
)
_DISCOUNT_TO_PRICE_RE = re.compile(r"(?:giảm|hạ|còn|xuống|về)\s*[:\-]?\s*$", re.IGNORECASE)

_BUNDLE_RE = re.compile(
    r"\b(?:bán\s+(?:cả\s+)?cặp|tổng\s+\d+\s*(?:căn|lô|nền)|"
    r"bán\s+(?:cả\s+)?[2-9]\s*(?:căn|lô|nền)\s+(?:liền|kề|gộp|giá)|combo\s+[2-9])\b",
    re.IGNORECASE,
)

# Fast numeric money patterns (prefix approx cues checked ONLY around matches for ~5x speedup)
_TY_TRIEU_RE = re.compile(
    r"(?P<ty>\d+(?:[.,]\d+)?)[.,]?\s*(?:tỷ|tỉ|ty)\s*(?P<tr_num>\d{1,3})\s*(?P<tr_unit>triệu|tr)?\b",
    re.IGNORECASE,
)
_APPROX_X_TY_RE = re.compile(
    r"(?P<ty>\d+)[.,]x\s*(?:tỷ|tỉ|ty)\b",
    re.IGNORECASE,
)
_TY_ONLY_RE = re.compile(
    r"(?P<ty>\d+(?:[.,]\d+)?)[.,]?\s*(?:tỷ|tỉ|ty)\b(?P<extra>\s*hơn|\s*rưỡi)?(?P<unit_sfx>\s*/\s*(?:1\s*)?(?:m2|m²|tháng|th|t\b|ngày|năm|căn|lô))?",
    re.IGNORECASE,
)
_TRIEU_RE = re.compile(
    r"(?P<tr_val>\d+(?:[.,]\d+)?)[.,]?\s*(?:triệu|tr|củ(?!\s+chi\b))\b(?P<unit_sfx>\s*/\s*(?:1\s*)?(?:m2|m²|mét|1m|tháng|th|t\b|ngày|năm)|\s+1m\b)?",
    re.IGNORECASE,
)
_DONG_RE = re.compile(
    r"\b(?P<d_val>\d+(?:[.,]\d{3})+)\s*(?:đ|đồng|vnđ|vnd)\b(?P<unit_sfx>\s*/\s*(?:1\s*)?(?:m2|m²|mét|tháng|th|t\b|ngày|năm))?",
    re.IGNORECASE,
)
_NON_TR_SUB_FOLLOW_RE = re.compile(r"^\s*(?:ngủ|pn|wc|vs|tầng|lầu|mặt|m2|m²|phòng|lô|căn|x\b|m\b)", re.IGNORECASE)
_PRE_APPROX_RE = re.compile(r"(?:nhỉnh|hơn|tầm|khoảng|gần|từ)\s*$", re.IGNORECASE)

_WORD_DIGITS = {
    "một": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "tư": 4,
    "năm": 5,
    "lăm": 5,
    "sáu": 6,
    "bảy": 7,
    "bẩy": 7,
    "tám": 8,
    "chín": 9,
    "mười": 10,
}
_WORD_MONEY_RE = re.compile(
    r"\b(?P<lead>một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s+(?P<scale>tỷ|tỉ|ty|trăm\s+triệu|chục\s+triệu)\s*(?P<tail>rưỡi|một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín)?\b",
    re.IGNORECASE,
)

_AREA_DIM_RE = re.compile(
    r"(?<!đường\s)(?<!ngõ\s)(?<!hẻm\s)(?P<w>\d+(?:[.,]\d+)?)\s*m?\s*[xX×*]\s*(?P<d>\d+(?:[.,]\d+)?)\s*m?\b(?!\s*(?:tầng|lầu|trệt))",
    re.IGNORECASE,
)
_AREA_EXPLICIT_RE = re.compile(
    r"(?:diện\s*tích|dt|dt\s*đất|dt\s*sàn)\s*[:=]?\s*(?P<a>\d+(?:[.,]\d+)?)\s*(?:m2|m²|mét(?:\s*vuông)?)?",
    re.IGNORECASE,
)
_AREA_UNIT_RE = re.compile(r"(?P<a>\d+(?:[.,]\d+)?)\s*(?:m2|m²|mét\s*vuông)\b", re.IGNORECASE)


def _to_float(num_str: str) -> float:
    return float(num_str.replace(",", "."))


def _classify_role(text: str, start: int, end: int, unit_sfx: str | None) -> str:
    prefix = text[max(0, start - 36) : start]
    suffix = text[end : min(len(text), end + 20)]

    if _EMBEDDED_RENT_CTX_RE.search(prefix) or _RENT_UNIT_RE.search(unit_sfx or "") or re.match(r"^\s*/\s*(?:tháng|th|t|ngày|năm)\b", suffix, re.IGNORECASE):
        return "rental_income"
    if _UNIT_M2_RE.search(unit_sfx or "") or re.match(r"^\s*/\s*(?:1\s*)?(?:m2|m²|mét|1m\b)", suffix, re.IGNORECASE):
        return "unit_per_m2"
    if _DISCOUNT_OR_DEPOSIT_CTX_RE.search(prefix) and not _DISCOUNT_TO_PRICE_RE.search(prefix):
        return "deposit_or_discount"
    if re.search(r"\b(?:giảm\s+\d|bớt\s+\d)", text[max(0, start - 10) : end], re.IGNORECASE):
        return "deposit_or_discount"
    if re.search(r"(?:phí\s+(?:dịch\s+vụ|quản\s+lý)|hoa\s+hồng)\s*[:\-]?\s*$", prefix, re.IGNORECASE):
        return "other"
    return "total_sale"


def _extract_mentions(text: str) -> list[MoneyMention]:
    mentions: list[MoneyMention] = []
    occupied: list[tuple[int, int]] = []

    def _overlaps(s: int, e: int) -> bool:
        return any(not (e <= os or s >= oe) for os, oe in occupied)

    for m in _TY_TRIEU_RE.finditer(text):
        s, e = m.span()
        if _overlaps(s, e):
            continue
        has_tr_unit = bool(m.group("tr_unit"))
        if not has_tr_unit and _NON_TR_SUB_FOLLOW_RE.match(text[e : e + 16]):
            continue
        ty = _to_float(m.group("ty"))
        tr_str = m.group("tr_num")
        tr_num = float(tr_str)
        if has_tr_unit:
            sub_vnd = tr_num * 1e6
        elif len(tr_str) == 1:
            sub_vnd = tr_num * 1e8
        elif len(tr_str) == 2:
            sub_vnd = tr_num * 1e7
        else:
            sub_vnd = tr_num * 1e6
        val = ty * 1e9 + sub_vnd
        role = _classify_role(text, s, e, None)
        is_approx = bool(_PRE_APPROX_RE.search(text[max(0, s - 14) : s]))
        mentions.append(MoneyMention(m.group(0), val, role, is_approx=is_approx, confidence="high"))
        occupied.append((s, e))

    for m in _APPROX_X_TY_RE.finditer(text):
        s, e = m.span()
        if _overlaps(s, e):
            continue
        val = (float(m.group("ty")) + 0.5) * 1e9
        role = _classify_role(text, s, e, None)
        mentions.append(MoneyMention(m.group(0), val, role, is_approx=True, confidence="medium"))
        occupied.append((s, e))

    for m in _TY_ONLY_RE.finditer(text):
        s, e = m.span()
        if _overlaps(s, e):
            continue
        ty_raw = m.group("ty")
        ty = _to_float(ty_raw)
        extra = (m.group("extra") or "").strip().lower()
        if extra == "rưỡi":
            ty += 0.5
        val = ty * 1e9
        unit_sfx = m.group("unit_sfx")
        role = _classify_role(text, s, e, unit_sfx)
        is_approx = bool(_PRE_APPROX_RE.search(text[max(0, s - 14) : s])) or extra == "hơn"
        mentions.append(MoneyMention(m.group(0), val, role, is_approx=is_approx, confidence="high"))
        occupied.append((s, e))

    for m in _TRIEU_RE.finditer(text):
        s, e = m.span()
        if _overlaps(s, e):
            continue
        val = _to_float(m.group("tr_val")) * 1e6
        unit_sfx = m.group("unit_sfx")
        role = _classify_role(text, s, e, unit_sfx)
        is_approx = bool(_PRE_APPROX_RE.search(text[max(0, s - 14) : s]))
        mentions.append(MoneyMention(m.group(0), val, role, is_approx=is_approx, confidence="high"))
        occupied.append((s, e))

    for m in _DONG_RE.finditer(text):
        s, e = m.span()
        if _overlaps(s, e):
            continue
        digits_only = re.sub(r"[.,]", "", m.group("d_val"))
        val = float(digits_only)
        if val < 100_000:
            continue
        unit_sfx = m.group("unit_sfx")
        role = _classify_role(text, s, e, unit_sfx)
        is_approx = bool(_PRE_APPROX_RE.search(text[max(0, s - 14) : s]))
        mentions.append(MoneyMention(m.group(0), val, role, is_approx=is_approx, confidence="high"))
        occupied.append((s, e))

    for m in _WORD_MONEY_RE.finditer(text):
        s, e = m.span()
        if _overlaps(s, e):
            continue
        lead = _WORD_DIGITS[m.group("lead").lower()]
        scale = m.group("scale").lower()
        tail = (m.group("tail") or "").lower()
        if "tỷ" in scale or "tỉ" in scale or "ty" in scale:
            val = lead * 1e9
            if tail == "rưỡi":
                val += 5e8
            elif tail in _WORD_DIGITS:
                val += _WORD_DIGITS[tail] * 1e8
        elif "trăm" in scale:
            val = lead * 1e8
            if tail == "rưỡi":
                val += 5e7
        else:
            val = lead * 1e7
        role = _classify_role(text, s, e, None)
        mentions.append(MoneyMention(m.group(0), val, role, is_approx=False, confidence="medium"))
        occupied.append((s, e))

    return mentions


def _extract_area_m2(text: str) -> float | None:
    for m in _AREA_DIM_RE.finditer(text):
        w = _to_float(m.group("w"))
        d = _to_float(m.group("d"))
        if 1.5 <= w <= 200.0 and 2.0 <= d <= 500.0:
            prod = w * d
            if 5.0 <= prod <= 100000.0:
                return prod
    for m in _AREA_EXPLICIT_RE.finditer(text):
        tail = text[m.end() : m.end() + 8]
        if re.match(r"^\s*m?\s*[xX×*]", tail):
            continue
        val = _to_float(m.group("a"))
        if 4.0 <= val <= 100000.0:
            return val
    for m in _AREA_UNIT_RE.finditer(text):
        val = _to_float(m.group("a"))
        if 4.0 <= val <= 100000.0:
            return val
    return None


def parse_listing_text(
    name: str | None,
    description: str | None,
    tabular_area: float | None = None,
) -> ParsedListing:
    """Parse transaction type, sale price, unit price, and area from listing text only."""
    name_norm = normalize_text(name)
    desc_norm = normalize_text(description)
    combined = f"{name_norm}. {desc_norm}".strip(". ").strip()

    mentions = _extract_mentions(combined)
    area_m2 = _extract_area_m2(combined)
    is_bundle = bool(_BUNDLE_RE.search(combined))
    is_negotiable = bool(_NEGOTIABLE_RE.search(combined))

    sale_mentions = [m for m in mentions if m.role == "total_sale"]
    unit_mentions = [m for m in mentions if m.role == "unit_per_m2"]
    rent_mentions = [m for m in mentions if m.role == "rental_income"]

    # Determine transaction type:
    # A sale listing can mention rental yield (HDT / đang cho thuê X tr/th) if it also states sale cues or sale price.
    has_sale_cue = bool(_SALE_CUE_RE.search(name_norm) or _SALE_CUE_RE.search(desc_norm))
    head_text = f"{name_norm} {desc_norm[:180]}"
    has_pure_rent_cue = bool(_PURE_RENT_CUE_RE.search(head_text))

    if has_pure_rent_cue and not (has_sale_cue and (sale_mentions or unit_mentions)):
        tx_type = "rent"
    elif rent_mentions and not sale_mentions and not unit_mentions and not has_sale_cue:
        tx_type = "rent"
    elif sale_mentions or unit_mentions or has_sale_cue:
        tx_type = "sale"
    elif is_negotiable:
        tx_type = "sale"
    else:
        tx_type = "unknown"

    unit_price = unit_mentions[0].value_vnd if unit_mentions else None
    sale_price = sale_mentions[0].value_vnd if sale_mentions else None
    if sale_price is None and unit_price is not None:
        ref_area = tabular_area if (tabular_area is not None and tabular_area > 0) else area_m2
        if ref_area is not None and ref_area > 0:
            sale_price = unit_price * float(ref_area)

    any_approx = any(m.is_approx for m in (sale_mentions + unit_mentions))
    if not any_approx and _APPROX_CUE_RE.search(combined):
        any_approx = True

    if sale_mentions or unit_mentions:
        confidence = sale_mentions[0].confidence if sale_mentions else unit_mentions[0].confidence
    else:
        confidence = "low"

    return ParsedListing(
        tx_type=tx_type,
        is_bundle=is_bundle,
        is_negotiable=is_negotiable,
        sale_price_vnd=sale_price,
        unit_price_per_m2_vnd=unit_price,
        area_m2=area_m2,
        is_approx=any_approx,
        confidence=confidence,
        mentions=mentions,
    )


def reconcile_labels(
    price: float | None,
    area: float | None,
    parsed: ParsedListing,
    domain_cfg: dict[str, Any],
) -> ReconcileResult:
    """Compare tabular `price` and `area` against independently parsed text signals."""
    if parsed.tx_type == "rent":
        return ReconcileResult("rental", "rental", False, None)

    price_missing = price is None or price != price or float(price) <= 0
    if price_missing:
        return ReconcileResult("missing_or_negotiable", "missing_or_negotiable", False, None)

    if parsed.is_bundle or parsed.tx_type == "unknown":
        return ReconcileResult("multi_or_unknown", "multi_or_unknown", False, None)

    area_missing = area is None or area != area or float(area) <= 0
    if area_missing:
        return ReconcileResult("missing_or_negotiable", "missing_area", False, None)

    p = float(price)
    a = float(area)
    if not (domain_cfg["min_area_m2"] <= a <= domain_cfg["max_area_m2"]):
        return ReconcileResult("noisy", "noisy_extreme_area", False, None)
    if not (domain_cfg["min_price_vnd"] <= p <= domain_cfg["max_price_vnd"]):
        return ReconcileResult("noisy", "noisy_extreme_price", False, None)

    ppm2 = p / a
    if not (domain_cfg["min_price_per_m2_vnd"] <= ppm2 <= domain_cfg["max_price_per_m2_vnd"]):
        return ReconcileResult("noisy", "noisy_extreme_ppm2", False, None)

    area_mismatch_flag = False
    if parsed.area_m2 is not None and parsed.area_m2 > 0:
        area_ratio = max(a / parsed.area_m2, parsed.area_m2 / a)
        if area_ratio > float(domain_cfg["area_mismatch_ratio"]):
            area_mismatch_flag = True

    if parsed.sale_price_vnd is None or parsed.sale_price_vnd <= 0:
        return ReconcileResult("clean", "clean_no_text_price", area_mismatch_flag, None)

    ratio = max(p / parsed.sale_price_vnd, parsed.sale_price_vnd / p)
    if ratio >= float(domain_cfg["order_of_magnitude_ratio"]):
        return ReconcileResult("noisy", "noisy_order_of_magnitude", area_mismatch_flag, ratio)

    max_allowed = (
        float(domain_cfg["price_mismatch_ratio_approx"])
        if parsed.is_approx
        else float(domain_cfg["price_mismatch_ratio_strict"])
    )
    if ratio > max_allowed:
        return ReconcileResult("noisy", "noisy_price_mismatch", area_mismatch_flag, ratio)

    return ReconcileResult("clean", "clean_consistent", area_mismatch_flag, ratio)
