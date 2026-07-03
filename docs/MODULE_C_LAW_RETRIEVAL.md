# Module C: Hybrid Law Retrieval

Code: `src/modules/law_retrieval/`

## Purpose

Retrieve law evidence from `data/corpus_law_pub.json` with high recall and rerank for precision.

## Retrieval tiers

1. Exact article-reference boost from query text.
2. BM25 lexical retrieval.
3. Dense retrieval using `BAAI/bge-m3`.
4. Reciprocal Rank Fusion.
5. Optional cross-encoder rerank using `BAAI/bge-reranker-v2-m3`.

## Fallback behavior

Dense retrieval and reranking fail closed: if dependencies or models are unavailable, the system continues with exact refs + BM25.

## Output contract

The facade `LawRetriever.search(...)` returns `LawArticle` objects. Submission export converts them to official `{law_id, aid}` objects through `PredictionRecord.to_submission()`.
