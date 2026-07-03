from __future__ import annotations

import hashlib
import re
import unicodedata

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    value = str(text or "")
    decomposed = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return value.replace("đ", "d").replace("Đ", "D").lower()


def tokenize(text: object) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def compact_text(text: object) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def normalized_query(text: object) -> str:
    return compact_text(normalize_text(text))


def stable_hash(text: object) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def truncate(text: object, limit: int) -> str:
    value = compact_text(text)
    return value if len(value) <= limit else value[:limit].rstrip() + "…"
