# Module C: Hybrid Law Retrieval

Code: `src/modules/law_retrieval/`

## Purpose

Retrieve law evidence from `data/corpus_law_pub.json` with high recall and rerank for precision.

## Retrieval tiers

1. Exact article-reference boost from query text.
2. BM25 lexical retrieval.
3. Dense retrieval using `BAAI/bge-m3`.
4. Reciprocal Rank Fusion across all active law queries.
5. Optional cross-encoder rerank using `BAAI/bge-reranker-v2-m3`.

## Runtime notes

- Law retrieval is called both during alternating RAG rounds and during final backfill before reasoning.
- The reranker now uses a compact combined query derived from multiple active law queries, instead of trusting only the first query string.
- Returned `LawArticle` candidates are later trimmed/backfilled by the reasoning layer to satisfy final evidence min/max requirements.

## Fallback behavior

Dense retrieval and reranking fail closed: if dependencies or models are unavailable, the system continues with exact refs + BM25.

## Output contract

The facade `LawRetriever.search(...)` returns `LawArticle` objects. Submission export converts them to official `{law_id, aid}` objects through `PredictionRecord.to_submission()`.
