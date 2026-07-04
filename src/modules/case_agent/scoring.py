from __future__ import annotations

from ...core.schema import CaseSegment
from ...core.text_utils import normalize_text

WEIGHTED_DECISION_PHRASES = {
    "quyet dinh": 5.0,
    "quyet dinh cua toa an": 5.5,
    "tuyen xu": 5.0,
    "xet thay": 3.0,
    "can cu vao": 3.0,
    "chap nhan yeu cau": 4.0,
    "khong chap nhan yeu cau": 5.0,
    "bac yeu cau": 5.0,
    "bac toan bo": 5.5,
    "chap nhan mot phan": 5.0,
    "buoc bi don": 3.5,
    "buoc bi don thanh toan": 4.5,
    "buoc ong": 2.5,
    "buoc ba": 2.5,
    "dinh chi xet xu": 3.5,
    "ghi nhan su tu nguyen": 3.0,
    "sua ban an so tham": 3.5,
    "giu nguyen ban an so tham": 3.5,
    "an phi": 2.0,
    "nhan dinh cua toa": 2.0,
    "hoi dong xet xu": 2.0,
    "co can cu": 1.5,
    "khong co can cu": 2.5,
    "yeu cau phan to": 1.5,
}


def decision_score(text: str) -> float:
    normalized = normalize_text(text)
    return sum(weight for phrase, weight in WEIGHTED_DECISION_PHRASES.items() if phrase in normalized)


def has_decision_signal(text: str) -> bool:
    return decision_score(text) >= 5.0


def rank_case_segments(segments: list[CaseSegment]) -> list[CaseSegment]:
    return sorted(segments, key=lambda segment: (decision_score(segment.text), segment.score), reverse=True)
