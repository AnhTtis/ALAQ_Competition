from __future__ import annotations

import json
import re
from pathlib import Path

from .schema import CaseGold, CaseInput, LawArticle
from .text_utils import compact_text, normalize_text

ARTICLE_RE = re.compile(r"(?:đi[eê]u|dieu)\s*(\d+)", re.IGNORECASE)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cases(path: Path) -> tuple[list[CaseInput], dict[str, CaseGold]]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list of cases in {path}")
    cases: list[CaseInput] = []
    gold: dict[str, CaseGold] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("case_id") or "").strip()
        case_query = compact_text(item.get("case_query") or "")
        if not case_id or not case_query:
            continue
        cases.append(CaseInput(case_id=case_id, case_query=case_query))
        gold[case_id] = CaseGold(
            case_id=case_id,
            verdict_label=_optional_str(item.get("verdict_label")),
            related_law_provisions=_optional_str(item.get("related_law_provisions")),
            case_fact=_optional_str(item.get("case_fact")),
            court_reasoning=_optional_str(item.get("court_reasoning")),
            court_verdict=_optional_str(item.get("court_verdict")),
            judgment_text=_optional_str(item.get("judgment_text")),
        )
    return cases, gold


def load_law_articles(path: Path) -> list[LawArticle]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list of laws in {path}")
    articles: list[LawArticle] = []
    for law in raw:
        if not isinstance(law, dict):
            continue
        law_id = str(law.get("law_id") or law.get("id") or "").strip()
        for index, article in enumerate(law.get("content") or [], start=1):
            if not isinstance(article, dict):
                continue
            aid = str(article.get("aid") or "").strip()
            text = compact_text(article.get("content_Article") or article.get("text") or "")
            if law_id and aid and text:
                articles.append(LawArticle(aid=aid, law_id=law_id, text=text, article_no=str(index)))
    if not articles:
        raise ValueError(f"No law articles found in {path}")
    return articles


def parse_related_law_articles(text: str | None) -> set[str]:
    if not text:
        return set()
    out: set[str] = set()
    for line in str(text).splitlines():
        normalized = normalize_text(line)
        for match in ARTICLE_RE.finditer(normalized):
            out.add(match.group(1))
    return out


def _optional_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None
