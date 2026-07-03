from __future__ import annotations

from ...core.schema import LawArticle


class CrossEncoderReranker:
    def __init__(self, *, model_id: str):
        self.model_id = model_id
        self._model = None

    def rerank(self, query: str, articles: list[LawArticle], *, top_k: int) -> list[LawArticle]:
        if not articles:
            return []
        try:
            model = self._load_model()
            pairs = [(query, article.text) for article in articles]
            scores = model.predict(pairs)
            scored = []
            for article, score in zip(articles, scores):
                scored.append(LawArticle(article.aid, article.law_id, article.text, float(score), "cross_encoder", article.article_no))
            return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]
        except Exception:
            return articles[:top_k]

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id)
        return self._model
