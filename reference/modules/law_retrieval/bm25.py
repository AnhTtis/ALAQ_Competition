from __future__ import annotations

import math
from collections import Counter

from ...core.schema import LawArticle
from ...core.text_utils import tokenize


class BM25LawRetriever:
    def __init__(self, articles: list[LawArticle], *, k1: float = 1.5, b: float = 0.75):
        if not articles:
            raise ValueError("articles must not be empty")
        self.articles = articles
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(article.text) for article in articles]
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))
        self.avg_len = sum(len(tokens) for tokens in self.doc_tokens) / max(len(self.doc_tokens), 1)

    def search(self, query: str, *, top_k: int) -> list[LawArticle]:
        query_tokens = Counter(tokenize(query))
        scored: list[LawArticle] = []
        for article, tokens in zip(self.articles, self.doc_tokens):
            score = self.score_tokens(query_tokens, tokens)
            if score > 0:
                scored.append(LawArticle(article.aid, article.law_id, article.text, score, "bm25", article.article_no))
        ranked = sorted(scored, key=lambda item: item.score, reverse=True)
        return ranked[:top_k] or [
            LawArticle(article.aid, article.law_id, article.text, 0.0, "bm25_fallback", article.article_no)
            for article in self.articles[:top_k]
        ]

    def score_tokens(self, query_tokens: Counter[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        term_freq = Counter(doc_tokens)
        n_docs = len(self.articles)
        score = 0.0
        for term, query_weight in query_tokens.items():
            frequency = term_freq.get(term, 0)
            if not frequency:
                continue
            doc_frequency = self.doc_freq.get(term, 0)
            idf = math.log(1 + ((n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5)))
            denom = frequency + self.k1 * (1 - self.b + self.b * len(doc_tokens) / max(self.avg_len, 1.0))
            score += query_weight * idf * ((frequency * (self.k1 + 1)) / denom)
        return score
