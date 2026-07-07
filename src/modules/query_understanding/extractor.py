from __future__ import annotations

import json
import re

from ...core.schema import CaseSegment, CaseUnderstanding, LawArticle
from ...core.text_utils import compact_text, normalize_text, truncate
from ...llm.base import LLMClient
from ..case_agent.query_strategy import build_case_queries, clean_case_query_for_retrieval
from .prompts import QUERY_UNDERSTANDING_SYSTEM_PROMPT, ROUND_CASE_QUERY_SYSTEM_PROMPT, ROUND_LAW_QUERY_SYSTEM_PROMPT
from .templates import detect_dispute_type, extract_article_refs, extract_legal_keywords, initial_case_queries, initial_law_queries


class CaseQueryUnderstanding:
    def __init__(self, llm: LLMClient | None = None, *, max_new_tokens: int = 1024):
        self.llm = llm
        self.max_new_tokens = max_new_tokens

    def analyze(self, case_id: str, case_query: str) -> CaseUnderstanding:
        fallback = self._deterministic(case_id, case_query)
        if self.llm is None:
            return fallback
        try:
            raw = self.llm.generate(
                [
                    {"role": "system", "content": QUERY_UNDERSTANDING_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({"case_id": case_id, "case_query": case_query}, ensure_ascii=False)},
                ],
                json_mode=True,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
            )
            data = _parse_json(raw)
            if not isinstance(data, dict):
                return fallback
            legal_keywords = _merge_lists(_string_list(data.get("legal_keywords")), list(fallback.legal_keywords))
            article_refs = _merge_lists(_string_list(data.get("article_refs")), list(fallback.article_refs))
            case_queries = _merge_lists(_string_list(data.get("case_search_queries")), list(fallback.case_search_queries))[:24]
            law_queries = _merge_lists(_string_list(data.get("law_search_queries")), list(fallback.law_search_queries))
            return CaseUnderstanding(
                case_id=case_id,
                original_query=case_query,
                normalized_query=normalize_text(case_query),
                dispute_type=compact_text(data.get("dispute_type") or fallback.dispute_type),
                plaintiff_requests=tuple(_string_list(data.get("plaintiff_requests")) or fallback.plaintiff_requests),
                defendant_positions=tuple(_string_list(data.get("defendant_positions")) or fallback.defendant_positions),
                legal_keywords=tuple(legal_keywords),
                article_refs=tuple(article_refs),
                case_search_queries=tuple(case_queries),
                law_search_queries=tuple(law_queries),
            )
        except Exception:
            return fallback

    def generate_round_law_queries(
        self,
        *,
        case_id: str,
        case_query: str,
        understanding: CaseUnderstanding,
        round_id: int,
        previous_laws: list[LawArticle],
        case_segments: list[CaseSegment],
        max_queries: int = 4,
    ) -> list[str]:
        fallback = self._round_law_fallback(case_query, understanding, previous_laws, case_segments, max_queries=max_queries)
        if self.llm is None:
            return fallback
        payload = {
            "case_id": case_id,
            "round_id": round_id,
            "case_query": case_query,
            "dispute_type": understanding.dispute_type,
            "legal_keywords": list(understanding.legal_keywords),
            "article_refs": list(understanding.article_refs),
            "existing_law_evidence": [_law_summary(article) for article in previous_laws[:12]],
            "case_evidence": [_case_summary(segment) for segment in case_segments[:8]],
            "previous_law_queries": list(understanding.law_search_queries),
        }
        queries = self._generate_query_list(
            system_prompt=ROUND_LAW_QUERY_SYSTEM_PROMPT,
            payload=payload,
            key="law_search_queries",
            fallback=fallback,
        )
        return _merge_lists(queries, fallback)[:max_queries]

    def generate_round_case_queries(
        self,
        *,
        case_id: str,
        case_query: str,
        understanding: CaseUnderstanding,
        round_id: int,
        law_articles: list[LawArticle],
        case_segments: list[CaseSegment],
        max_queries: int = 4,
    ) -> list[str]:
        fallback = self._round_case_fallback(case_query, understanding, law_articles, max_queries=max_queries)
        if self.llm is None:
            return fallback
        payload = {
            "case_id": case_id,
            "round_id": round_id,
            "case_query": case_query,
            "dispute_type": understanding.dispute_type,
            "plaintiff_requests": list(understanding.plaintiff_requests),
            "defendant_positions": list(understanding.defendant_positions),
            "legal_keywords": list(understanding.legal_keywords),
            "top_law_evidence": [_law_summary(article) for article in law_articles[:max_queries]],
            "case_evidence": [_case_summary(segment) for segment in case_segments[:8]],
        }
        queries = self._generate_query_list(
            system_prompt=ROUND_CASE_QUERY_SYSTEM_PROMPT,
            payload=payload,
            key="case_search_queries",
            fallback=fallback,
        )
        return _merge_lists(queries, fallback)[:max_queries]

    def _generate_query_list(self, *, system_prompt: str, payload: dict[str, object], key: str, fallback: list[str]) -> list[str]:
        try:
            raw = self.llm.generate(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                json_mode=True,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
            )
            data = _parse_json(raw)
            if isinstance(data, dict):
                return _string_list(data.get(key)) or fallback
        except Exception:
            pass
        return fallback

    def _round_law_fallback(
        self,
        case_query: str,
        understanding: CaseUnderstanding,
        previous_laws: list[LawArticle],
        case_segments: list[CaseSegment],
        *,
        max_queries: int,
    ) -> list[str]:
        queries = list(understanding.law_search_queries)
        if not queries:
            queries.extend(initial_law_queries(case_query, list(understanding.legal_keywords), list(understanding.article_refs)))
        for segment in case_segments[:4]:
            queries.append(compact_text(f"{understanding.dispute_type} {truncate(segment.text, 360)} căn cứ pháp lý điều luật áp dụng"))
        for article in previous_laws[:4]:
            article_ref = article.article_no or str(article.aid)
            queries.append(compact_text(f"{understanding.dispute_type} Điều {article_ref} {truncate(article.text, 240)}"))
        queries.append(compact_text(f"{case_query} căn cứ pháp luật trách nhiệm nghĩa vụ điều kiện chấp nhận yêu cầu"))
        return _dedupe_texts(queries)[:max_queries]

    def _round_case_fallback(
        self,
        case_query: str,
        understanding: CaseUnderstanding,
        law_articles: list[LawArticle],
        *,
        max_queries: int,
    ) -> list[str]:
        clean_query = clean_case_query_for_retrieval(case_query)
        queries = build_case_queries(understanding, max_queries=max_queries * 3)
        law_guided = []
        for article in law_articles[:3]:
            article_ref = article.article_no or str(article.aid)
            law_guided.extend(
                [
                    f"{clean_query} áp dụng Điều {article_ref} nhận định của tòa",
                    f"{clean_query} Điều {article_ref} phần quyết định tuyên xử chấp nhận không chấp nhận yêu cầu",
                    f"{clean_query} căn cứ pháp lý Điều {article_ref} nghĩa vụ cụ thể của các bên",
                ]
            )
        return _dedupe_texts(law_guided + queries)[:max_queries]

    def _deterministic(self, case_id: str, case_query: str) -> CaseUnderstanding:
        legal_keywords = extract_legal_keywords(case_query)
        article_refs = extract_article_refs(case_query)
        return CaseUnderstanding(
            case_id=case_id,
            original_query=compact_text(case_query),
            normalized_query=normalize_text(case_query),
            dispute_type=detect_dispute_type(case_query),
            plaintiff_requests=tuple(_extract_side_sentences(case_query, ["nguyên đơn", "người khởi kiện", "người yêu cầu", "yêu cầu"])),
            defendant_positions=tuple(_extract_side_sentences(case_query, ["bị đơn", "người bị kiện", "phản tố", "không đồng ý"])),
            legal_keywords=tuple(legal_keywords),
            article_refs=tuple(article_refs),
            case_search_queries=tuple(initial_case_queries(case_query, legal_keywords)),
            law_search_queries=tuple(initial_law_queries(case_query, legal_keywords, article_refs)),
        )


def _extract_side_sentences(text: str, markers: list[str]) -> list[str]:
    normalized_markers = [normalize_text(marker) for marker in markers]
    sentences = re.split(r"(?<=[.!?。])\s+|;", compact_text(text))
    out = []
    for sentence in sentences:
        normalized = normalize_text(sentence)
        if any(marker in normalized for marker in normalized_markers):
            out.append(compact_text(sentence))
    return out[:4]


def _parse_json(raw: str) -> object:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            text = match.group(0)
    try:
        return json.loads(text)
    except Exception:
        return None


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [compact_text(item) for item in value if compact_text(item)]
    text = compact_text(value)
    return [text] if text else []


def _merge_lists(primary: list[str], fallback: list[str]) -> list[str]:
    return _dedupe_texts(primary + fallback)


def _dedupe_texts(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = compact_text(item)
        key = normalize_text(text)
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _law_summary(article: LawArticle) -> dict[str, object]:
    return {
        "law_id": article.law_id,
        "aid": article.aid,
        "article_no": article.article_no,
        "text": truncate(article.text, 520),
        "score": article.score,
    }


def _case_summary(segment: CaseSegment) -> dict[str, object]:
    return {
        "chunk_id": segment.chunk_id,
        "query": segment.query,
        "text": truncate(segment.text, 720),
        "score": segment.score,
    }
