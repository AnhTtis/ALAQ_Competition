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
        query = _join_queries(queries)
        if not query:
            return []
        top_k = top_k or self.settings.law_top_k
        ranked_lists = []
        exact_hits = article_reference_hits(query, self.articles, self.bm25)
        if exact_hits:
            ranked_lists.append(exact_hits)
        ranked_lists.append(self.bm25.search(query, top_k=max(self.settings.law_bm25_candidates, top_k)))
        if self.settings.enable_dense_retrieval and self.settings.llm_backend.lower() != "mock":
            dense_hits = self.dense.search(query, top_k=max(self.settings.law_dense_candidates, top_k))
            if dense_hits:
                ranked_lists.append(dense_hits)
        fused = reciprocal_rank_fusion(ranked_lists)
        candidates = diversify_by_law(fused, top_k=max(self.settings.law_rerank_top_k, top_k), max_per_law_id=12)
        if self.settings.enable_cross_encoder_rerank and self.settings.llm_backend.lower() != "mock":
            candidates = self.reranker.rerank(query, candidates, top_k=max(self.settings.law_rerank_top_k, top_k))
        return diversify_by_law(candidates, top_k=top_k, max_per_law_id=8)


def _join_queries(queries: list[str] | str) -> str:
    if isinstance(queries, str):
        return compact_text(queries)
    return compact_text(" ".join(compact_text(query) for query in queries if compact_text(query)))
