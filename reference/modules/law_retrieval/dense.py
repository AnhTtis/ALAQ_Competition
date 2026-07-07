from __future__ import annotations

import hashlib
from pathlib import Path

from ...core.schema import LawArticle


class DenseLawRetriever:
    def __init__(self, articles: list[LawArticle], *, model_id: str, cache_path: Path):
        self.articles = articles
        self.model_id = model_id
        self.cache_path = cache_path
        self.corpus_fingerprint = _articles_fingerprint(articles)
        self._model = None
        self._matrix = None

    def search(self, query: str, *, top_k: int) -> list[LawArticle]:
        try:
            import numpy as np

            matrix = self._load_or_build_matrix()
            model = self._load_model()
            query_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            scores = np.asarray(query_vec) @ matrix.T
            ranked_indices = np.argsort(-scores[0])[:top_k]
            return [
                LawArticle(
                    self.articles[index].aid,
                    self.articles[index].law_id,
                    self.articles[index].text,
                    float(scores[0][index]),
                    "dense",
                    self.articles[index].article_no,
                )
                for index in ranked_indices
            ]
        except Exception:
            return []

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
        return self._model

    def _load_or_build_matrix(self):
        import numpy as np

        if self._matrix is not None:
            return self._matrix
        if self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=False)
            corpus_fingerprint = str(data["corpus_fingerprint"]) if "corpus_fingerprint" in data.files else ""
            article_count = int(data["article_count"]) if "article_count" in data.files else -1
            if str(data["model_id"]) == self.model_id and corpus_fingerprint == self.corpus_fingerprint and article_count == len(self.articles):
                self._matrix = data["embeddings"]
                return self._matrix
        model = self._load_model()
        texts = [article.text for article in self.articles]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            model_id=self.model_id,
            corpus_fingerprint=self.corpus_fingerprint,
            article_count=len(self.articles),
            embeddings=np.asarray(embeddings),
        )
        self._matrix = np.asarray(embeddings)
        return self._matrix


def _articles_fingerprint(articles: list[LawArticle]) -> str:
    digest = hashlib.sha256()
    for article in articles:
        digest.update(article.law_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(article.aid.encode("utf-8"))
        digest.update(b"\0")
        digest.update(article.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
