from __future__ import annotations

from ...core.config import Settings
from ...core.schema import LawArticle
from ...core.text_utils import compact_text
from .article_refs import article_reference_hits
from .bm25 import BM25LawRetriever
from .dense import DenseLawRetriever
from .fusion import diversify_by_law, reciprocal_rank_fusion
from .reranker import CrossEncoderReranker


class LawRetriever:
    def __init__(self, articles: list[LawArticle], settings: Settings):
        self.articles = articles
        self.settings = settings
        self.bm25 = BM25LawRetriever(articles)
        self.dense = DenseLawRetriever(
            articles,
            model_id=settings.embedding_model_id,
            cache_path=settings.cache_dir / "bge_law_embeddings.npz",
        )
        self.reranker = CrossEncoderReranker(model_id=settings.cross_encoder_model_id)

    def search(self, queries: list[str] | str, *, top_k: int | None = None) -> list[LawArticle]:
        query_list = _as_queries(queries)
        if not query_list:
            return []
        top_k = top_k or self.settings.law_top_k
        ranked_lists: list[list[LawArticle]] = []
        for query in query_list:
            ranked_lists.extend(self._ranked_lists_for_query(query, top_k=top_k))
        fused = reciprocal_rank_fusion(ranked_lists)
        candidates = diversify_by_law(fused, top_k=max(self.settings.law_rerank_top_k, top_k), max_per_law_id=12)
        if self.settings.enable_cross_encoder_rerank and self.settings.llm_backend.lower() != "mock":
            rerank_query = _combined_rerank_query(query_list)
            candidates = self.reranker.rerank(rerank_query, candidates, top_k=max(self.settings.law_rerank_top_k, top_k))
        return diversify_by_law(candidates, top_k=top_k, max_per_law_id=8)

    def _ranked_lists_for_query(self, query: str, *, top_k: int) -> list[list[LawArticle]]:
        ranked_lists: list[list[LawArticle]] = []
        exact_hits = article_reference_hits(query, self.articles, self.bm25)
        if exact_hits:
            ranked_lists.append(exact_hits)
        ranked_lists.append(self.bm25.search(query, top_k=max(self.settings.law_bm25_candidates, top_k)))
        if self.settings.enable_dense_retrieval and self.settings.llm_backend.lower() != "mock":
            dense_hits = self.dense.search(query, top_k=max(self.settings.law_dense_candidates, top_k))
            if dense_hits:
                ranked_lists.append(dense_hits)
        return ranked_lists


def _as_queries(queries: list[str] | str) -> list[str]:
    if isinstance(queries, str):
        return [compact_text(queries)] if compact_text(queries) else []
    out: list[str] = []
    seen: set[str] = set()
    for query in queries:
        text = compact_text(query)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _combined_rerank_query(queries: list[str], *, max_queries: int = 4, max_chars: int = 900) -> str:
    combined = compact_text(" ; ".join(queries[:max_queries]))
    return combined[:max_chars]
