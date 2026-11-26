"""Unicode/HTML normalization and PII redaction for Vietnamese real estate listings."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any


_HTML_BREAK_RE = re.compile(r"</?(?:br|p|div|li|tr|h[1-6])\b[^>]*>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE)
_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+|(?:zalo\.me|fb\.com|facebook\.com)/\S+)", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.\-]*\d){8,10}(?!\d)")
_CONTACT_PHONE_HINT_RE = re.compile(
    r"\b(?:zalo|đt|sdt|sđt|hotline|call|lh|liên\s*hệ)\s*[:\-]?\s*(?:\+?84|0|x)[\dx\s.\-*]{5,15}",
    re.IGNORECASE,
)
_CONTACT_NAME_RE = re.compile(
    r"(?i:\b(?:liên\s*hệ|lh|gặp)\b)\s*[:\-]?\s*(?:(?i:anh|chị|em|cô|chú|bác|mr|ms|mrs)\.?\s+)?[A-ZĐÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZĐÀ-Ỹ][\wÀ-ỹ]*){0,2}"
)
_PERSON_NAME_RE = re.compile(r"(?i:\b(?:anh|chị|chú|cô|bác|ông|bà)\b)\s+[A-ZĐÀ-Ỹ][\wÀ-ỹ]+")
_HOUSE_NUM_RE = re.compile(
    r"\b(?:nhà\s+)?số\s+\d+(?:/\d+){1,4}\b|\b\d+/\d+(?:/\d+)?(?=\s+(?:đường|phố|ngõ|hẻm|khu|p\.|phường))",
    re.IGNORECASE,
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        # Handles float('nan'), np.nan, pd.NA safely without forcing pandas import
        return value != value
    except Exception:
        return False


def normalize_text(text: Any) -> str:
    """Normalize HTML entities/tags, Unicode NFC, and whitespace."""
    if _is_missing(text):
        return ""
    s = str(text)
    s = html.unescape(s)
    s = _HTML_BREAK_RE.sub(" ", s)
    s = _HTML_TAG_RE.sub(" ", s)
    s = unicodedata.normalize("NFC", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def redact_pii(text: Any) -> str:
    """Redact phones, emails, URLs/Zalo, broker contact names, and specific house numbers."""
    s = normalize_text(text)
    if not s:
        return ""
    s = _URL_RE.sub("[LINK]", s)
    s = _EMAIL_RE.sub("[EMAIL]", s)
    s = _CONTACT_PHONE_HINT_RE.sub("LH [SĐT]", s)
    s = _PHONE_RE.sub("[SĐT]", s)
    s = _CONTACT_NAME_RE.sub("LH [LIÊN_HỆ]", s)
    s = _PERSON_NAME_RE.sub("[TÊN]", s)
    s = _HOUSE_NUM_RE.sub("số [ĐỊA_CHỈ]", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s
