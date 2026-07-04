from __future__ import annotations

import re
from collections import Counter

from ...core.schema import LawArticle
from ...core.text_utils import normalize_text, tokenize
from .bm25 import BM25LawRetriever

ARTICLE_REF_RE = re.compile(r"\bdieu\s*(\d{1,4})\b")
LAW_FAMILY_ALIASES = {
    "91/2015/QH13": ["bo luat dan su", "blds"],
    "92/2015/QH13": ["bo luat to tung dan su", "blttds", "to tung dan su"],
    "45/2013/QH13": ["luat dat dai", "dat dai"],
    "47/2010/QH12": ["luat cac to chuc tin dung", "to chuc tin dung", "tin dung", "ngan hang"],
    "52/2014/QH13": ["luat hon nhan va gia dinh", "hon nhan gia dinh", "ly hon"],
    "66/2014/QH13": ["luat kinh doanh bat dong san", "kinh doanh bat dong san"],
    "65/2014/QH13": ["luat nha o", "nha o"],
    "50/2014/QH13": ["luat xay dung", "xay dung"],
    "100/2015/QH13": ["bo luat hinh su", "blhs", "hinh su"],
    "326/2016/UBTVQH14": ["an phi", "le phi toa an"],
}


def article_reference_hits(query: str, articles: list[LawArticle], bm25: BM25LawRetriever) -> list[LawArticle]:
    normalized_query = normalize_text(query)
    article_numbers = set(ARTICLE_REF_RE.findall(normalized_query))
    if not article_numbers:
        return []
    family_hints = _law_family_hints(normalized_query)
    query_tokens = Counter(tokenize(query))
    hits = []
    for article, tokens in zip(articles, bm25.doc_tokens):
        if article.article_no not in article_numbers:
            continue
        if family_hints and article.law_id not in family_hints:
            continue
        boost = 1000.0 if family_hints else 8.0
        score = boost + bm25.score_tokens(query_tokens, tokens)
        hits.append(LawArticle(article.aid, article.law_id, article.text, score, "article_ref", article.article_no))
    return sorted(hits, key=lambda article: article.score, reverse=True)


def _law_family_hints(normalized_query: str) -> set[str]:
    hints: set[str] = set()
    for law_id, aliases in LAW_FAMILY_ALIASES.items():
        if any(alias in normalized_query for alias in aliases):
            hints.add(law_id)
    return hints
