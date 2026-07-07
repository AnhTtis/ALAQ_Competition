from __future__ import annotations

import re

from ...core.schema import CaseSegment
from ...core.text_utils import normalize_text


B_REJECTION_PHRASES = [
    "khong chap nhan yeu cau khoi kien",
    "khong chap nhan yeu cau cua nguyen don",
    "khong chap nhan toan bo",
    "khong duoc chap nhan",
    "bac yeu cau khoi kien",
    "bac toan bo yeu cau",
    "bac toan bo",
    "khong co co so",
    "khong co can cu",
    "khong co can cu chap nhan",
    "dinh chi yeu cau",
]

A_FULL_ACCEPT_PHRASES = [
    "chap nhan toan bo yeu cau khoi kien",
    "chap nhan toan bo yeu cau cua nguyen don",
    "chap nhan toan bo yeu cau",
    "chap nhan yeu cau khoi kien cua nguyen don",
    "chap nhan yeu cau cua nguyen don",
    "buoc bi don thanh toan toan bo",
]

PARTIAL_ACCEPT_PHRASES = [
    "chap nhan mot phan yeu cau khoi kien",
    "chap nhan mot phan yeu cau cua nguyen don",
    "chap nhan mot phan yeu cau",
    "mot phan yeu cau cua nguyen don",
]

HALF_OR_LESS_SIGNALS = [
    "1/2",
    "½",
    "mot nua",
    "phan nua",
    "50%",
    "50 phan tram",
    "nam muoi phan tram",
    "khong qua 50",
    "duoi 50",
    "tu 50 tro xuong",
    "bang 50",
    ": 2",
    "chia doi",
]

DECISION_ZONE_MARKERS = [
    "quyet dinh",
    "tuyen xu",
    "vi cac le tren",
    "hoi dong xet xu nhan dinh",
    "toa an nhan dinh",
    "xet thay",
    "can cu vao",
    "chap nhan",
    "khong chap nhan",
    "bac yeu cau",
    "bac toan bo",
    "buoc",
    "dinh chi",
    "an phi",
]

NARRATION_MARKERS = [
    "nguyen don trinh bay",
    "bi don trinh bay",
    "nguoi co quyen loi nghia vu lien quan trinh bay",
    "theo don khoi kien",
    "nguyen don yeu cau toa an",
    "de nghi toa an",
    "yeu cau khoi kien cua nguyen don la",
]


def decision_rule_prediction(case_segments: list[CaseSegment]) -> str | None:
    text = _decision_text(case_segments)
    if not text:
        return None
    if _has_any(text, B_REJECTION_PHRASES):
        return "B_WIN"
    if _has_any(text, PARTIAL_ACCEPT_PHRASES):
        amount_prediction = _amount_ratio_prediction("", case_segments)
        if amount_prediction in {"PARTIAL_A_WIN", "PARTIAL_B_WIN"}:
            return amount_prediction
        return "PARTIAL_B_WIN" if _has_any(text, HALF_OR_LESS_SIGNALS) else "PARTIAL_A_WIN"
    if _has_any(text, A_FULL_ACCEPT_PHRASES):
        return "A_WIN"
    return None


def heuristic_prediction(case_query: str, case_segments: list[CaseSegment]) -> str:
    rule_prediction = decision_rule_prediction(case_segments)
    if rule_prediction:
        return rule_prediction
    text = normalize_text(" ".join([case_query] + [segment.text for segment in case_segments]))
    if _has_any(text, B_REJECTION_PHRASES):
        return "B_WIN"
    if _has_any(text, PARTIAL_ACCEPT_PHRASES):
        amount_prediction = _amount_ratio_prediction(case_query, case_segments)
        if amount_prediction:
            return amount_prediction
        return "PARTIAL_B_WIN" if _has_any(text, HALF_OR_LESS_SIGNALS) else "PARTIAL_A_WIN"
    if _has_any(text, A_FULL_ACCEPT_PHRASES):
        return "A_WIN"
    amount_prediction = _amount_ratio_prediction(case_query, case_segments)
    if amount_prediction:
        return amount_prediction
    if "buoc bi don phai" in text or "buoc bi don thanh toan" in text:
        return "A_WIN"
    return "A_WIN"


def _decision_text(case_segments: list[CaseSegment]) -> str:
    decision_chunks: list[str] = []
    for segment in case_segments[:12]:
        text = normalize_text(segment.text)
        if _is_decision_zone(text):
            decision_chunks.append(text)
    return " ".join(decision_chunks[:6])


def _is_decision_zone(text: str) -> bool:
    if not _has_any(text, DECISION_ZONE_MARKERS):
        return False
    if _has_any(text, ("quyet dinh", "tuyen xu", "vi cac le tren", "hoi dong xet xu", "toa an nhan dinh", "xet thay", "can cu vao")):
        return True
    return not _has_any(text, NARRATION_MARKERS)


def _amount_ratio_prediction(case_query: str, case_segments: list[CaseSegment]) -> str | None:
    decision_text = _decision_text(case_segments) or normalize_text(" ".join(segment.text for segment in case_segments[:8]))
    if not any(marker in decision_text for marker in ("chap nhan", "buoc", "thanh toan", "boi thuong")):
        return None
    requested = max(_money_values(normalize_text(case_query)), default=0.0)
    awarded = max(_money_values(decision_text), default=0.0)
    if requested <= 0 or awarded <= 0 or awarded > requested * 1.2:
        return None
    ratio = awarded / requested
    if ratio >= 0.95:
        return "A_WIN"
    return "PARTIAL_A_WIN" if ratio > 0.5 else "PARTIAL_B_WIN"


def _money_values(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"\b(\d[\d\.,]*)\s*(ty|trieu|dong|vnd)?\b", text):
        raw, unit = match.groups()
        digits = re.sub(r"[^\d]", "", raw)
        if not digits:
            continue
        value = float(digits)
        if unit == "ty":
            value *= 1_000_000_000
        elif unit == "trieu":
            value *= 1_000_000
        elif unit in {"dong", "vnd"} or len(digits) >= 6:
            pass
        else:
            continue
        values.append(value)
    return values


def _has_any(text: str, phrases: list[str] | tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)
