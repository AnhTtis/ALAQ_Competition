from __future__ import annotations

import re
from collections import Counter

from ...core.schema import LawArticle
from ...core.text_utils import normalize_text, tokenize
from .bm25 import BM25LawRetriever

ARTICLE_REF_RE = re.compile(r"\bdieu\s*(\d{1,4})\b")


def article_reference_hits(query: str, articles: list[LawArticle], bm25: BM25LawRetriever) -> list[LawArticle]:
    article_numbers = set(ARTICLE_REF_RE.findall(normalize_text(query)))
    if not article_numbers:
        return []
    query_tokens = Counter(tokenize(query))
    hits = []
    for article, tokens in zip(articles, bm25.doc_tokens):
        if article.article_no not in article_numbers:
            continue
        score = 1000.0 + bm25.score_tokens(query_tokens, tokens)
        hits.append(LawArticle(article.aid, article.law_id, article.text, score, "article_ref", article.article_no))
    return sorted(hits, key=lambda article: article.score, reverse=True)
