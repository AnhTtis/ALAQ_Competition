from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from ...core.schema import LawArticle


def reciprocal_rank_fusion(ranked_lists: Iterable[list[LawArticle]], *, k: int = 60) -> list[LawArticle]:
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, LawArticle] = {}
    for ranked in ranked_lists:
        for rank, article in enumerate(ranked, start=1):
            key = article.evidence_id
            scores[key] += 1.0 / (k + rank)
            if key not in best or article.score > best[key].score:
                best[key] = article
    out = []
    for key, score in scores.items():
        article = best[key]
        out.append(LawArticle(article.aid, article.law_id, article.text, score, "hybrid", article.article_no))
    return sorted(out, key=lambda item: item.score, reverse=True)


def diversify_by_law(articles: list[LawArticle], *, top_k: int, max_per_law_id: int = 8) -> list[LawArticle]:
    counts: Counter[str] = Counter()
    selected: list[LawArticle] = []
    for article in articles:
        if counts[article.law_id] >= max_per_law_id:
            continue
        selected.append(article)
        counts[article.law_id] += 1
        if len(selected) >= top_k:
            return selected
    return selected or articles[:top_k]
