from __future__ import annotations

import json
import re

from ...core.schema import OUTCOME_LABELS, CaseSegment, LawArticle


def parse_prediction_payload(raw: str) -> dict[str, object]:
    data = _parse_json(raw)
    if isinstance(data, dict):
        pred = str(data.get("prediction") or "").strip().upper()
        if pred in OUTCOME_LABELS:
            data["prediction"] = pred
        return data
    upper = raw.upper()
    for label in OUTCOME_LABELS:
        if label in upper:
            return {"prediction": label}
    return {}


def selected_law_evidence(parsed: dict[str, object], fallback: list[LawArticle]) -> list[LawArticle]:
    requested = parsed.get("law_evidence")
    if not isinstance(requested, list):
        return fallback[:16]
    by_aid = {str(article.aid): article for article in fallback}
    by_law_article = {(article.law_id, str(article.article_no)): article for article in fallback if article.article_no}
    selected: list[LawArticle] = []
    seen: set[str] = set()
    for item in requested:
        article = None
        if isinstance(item, str):
            article = by_aid.get(item)
        elif isinstance(item, dict):
            law_id = str(item.get("law_id") or "")
            aid = str(item.get("aid") or item.get("id") or "")
            article_no = str(item.get("article_no") or item.get("article") or item.get("dieu") or "")
            article = by_aid.get(aid) or by_law_article.get((law_id, article_no))
        if article and article.evidence_id not in seen:
            selected.append(article)
            seen.add(article.evidence_id)
    return selected or fallback[:16]


def selected_case_evidence(parsed: dict[str, object], fallback: list[CaseSegment]) -> list[CaseSegment]:
    requested = parsed.get("case_evidence")
    if not isinstance(requested, list):
        return fallback[:16]
    index = {segment.chunk_id: segment for segment in fallback}
    selected: list[CaseSegment] = []
    seen: set[str] = set()
    for item in requested:
        chunk_id = ""
        if isinstance(item, str):
            chunk_id = item
        elif isinstance(item, dict):
            chunk_id = str(item.get("chunk_id") or item.get("hash_id") or item.get("id") or "")
        if chunk_id in index and chunk_id not in seen:
            selected.append(index[chunk_id])
            seen.add(chunk_id)
    return selected or fallback[:16]


def float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _parse_json(raw: str) -> object:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            text = match.group(0)
    try:
        return json.loads(text)
    except Exception:
        return None
