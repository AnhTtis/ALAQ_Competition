from __future__ import annotations

from ...core.schema import CaseSegment
from ...core.text_utils import normalize_text


B_REJECTION_PHRASES = [
    "khong chap nhan yeu cau khoi kien",
    "khong chap nhan yeu cau cua nguyen don",
    "bac yeu cau khoi kien",
    "bac toan bo yeu cau",
    "khong co can cu chap nhan",
    "dinh chi yeu cau",
]

A_FULL_ACCEPT_PHRASES = [
    "chap nhan toan bo yeu cau khoi kien",
    "chap nhan toan bo yeu cau cua nguyen don",
    "chap nhan toan bo yeu cau",
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


def decision_rule_prediction(case_segments: list[CaseSegment]) -> str | None:
    text = normalize_text(" ".join(segment.text for segment in case_segments[:8]))
    if _has_any(text, A_FULL_ACCEPT_PHRASES):
        return "A_WIN"
    if _has_any(text, B_REJECTION_PHRASES):
        return "B_WIN"
    if _has_any(text, PARTIAL_ACCEPT_PHRASES):
        return "PARTIAL_B_WIN" if _has_any(text, HALF_OR_LESS_SIGNALS) else "PARTIAL_A_WIN"
    return None


def heuristic_prediction(case_query: str, case_segments: list[CaseSegment]) -> str:
    text = normalize_text(" ".join([case_query] + [segment.text for segment in case_segments]))
    if _has_any(text, A_FULL_ACCEPT_PHRASES):
        return "A_WIN"
    if _has_any(text, B_REJECTION_PHRASES):
        return "B_WIN"
    if _has_any(text, PARTIAL_ACCEPT_PHRASES):
        return "PARTIAL_B_WIN" if _has_any(text, HALF_OR_LESS_SIGNALS) else "PARTIAL_A_WIN"
    if "buoc bi don phai" in text or "buoc bi don thanh toan" in text:
        return "A_WIN"
    return "A_WIN"


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)
