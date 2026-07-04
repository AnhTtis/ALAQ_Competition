from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OUTCOME_LABELS = {"A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"}


@dataclass(frozen=True)
class CaseInput:
    case_id: str
    case_query: str


@dataclass(frozen=True)
class CaseGold:
    case_id: str
    verdict_label: str | None = None
    related_law_provisions: str | None = None
    case_fact: str | None = None
    court_reasoning: str | None = None
    court_verdict: str | None = None
    judgment_text: str | None = None


@dataclass(frozen=True)
class LawArticle:
    aid: str
    law_id: str
    text: str
    score: float = 0.0
    source: str = ""
    article_no: str = ""

    @property
    def evidence_id(self) -> str:
        return f"{self.law_id}:{self.aid}"


@dataclass(frozen=True)
class CaseSegment:
    chunk_id: str
    text: str
    score: float = 0.0
    query: str = ""


@dataclass(frozen=True)
class CaseUnderstanding:
    case_id: str
    original_query: str
    normalized_query: str
    dispute_type: str = ""
    plaintiff_requests: tuple[str, ...] = ()
    defendant_positions: tuple[str, ...] = ()
    legal_keywords: tuple[str, ...] = ()
    article_refs: tuple[str, ...] = ()
    case_search_queries: tuple[str, ...] = ()
    law_search_queries: tuple[str, ...] = ()


@dataclass
class PredictionRecord:
    case_id: str
    prediction: str
    law_evidence: list[LawArticle] = field(default_factory=list)
    case_evidence: list[CaseSegment] = field(default_factory=list)
    api_calls: int = 0
    retrieval_queries: list[str] = field(default_factory=list)
    retrieval_query_families: list[str] = field(default_factory=list)
    retrieval_query_new_segments: list[int] = field(default_factory=list)
    retrieval_query_result_chunks: list[list[str]] = field(default_factory=list)
    case_segments_retrieved: int = 0
    no_new_case_queries: int = 0
    confidence: float | None = None
    reasoning_summary: str = ""
    raw_model_output: str = ""
    fallback_used: bool = False
    llm_prediction: str | None = None
    decision_rule_prediction: str | None = None
    override_reason: str = ""

    def to_submission(self) -> dict[str, Any]:
        law_evidence: list[Any] = [
            {"law_id": article.law_id, "aid": _json_aid(article.aid)}
            for article in self.law_evidence
        ]
        return {
            "case_id": self.case_id,
            "prediction": self.prediction,
            "case_evidence": _dedupe_jsonable([segment.chunk_id for segment in self.case_evidence]),
            "law_evidence": _dedupe_jsonable(law_evidence),
        }


def _json_aid(value: object) -> object:
    text = str(value)
    return int(text) if text.isdigit() else text


def _dedupe_jsonable(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for item in items:
        key = repr(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out
