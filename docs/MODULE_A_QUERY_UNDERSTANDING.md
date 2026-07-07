# Module A: Case Query Understanding

Code: `src/modules/query_understanding/`

## Purpose

Convert the short `case_query` into structured search intent before any API call.

## Outputs

`CaseUnderstanding` contains:

- `dispute_type`
- `plaintiff_requests`
- `defendant_positions`
- `legal_keywords`
- `article_refs`
- `case_search_queries`
- `law_search_queries`

## Strategy

1. Deterministic extraction first: regex article references, legal keyword groups, dispute type detection.
2. Optional local LLM JSON extraction using `QUERY_UNDERSTANDING_SYSTEM_PROMPT`.
3. Merge LLM output with deterministic fallback.
4. Build retrieval plans for two targets:
   - `case_search_queries` for the official Top-1 Case API.
   - `law_search_queries` for the local law corpus.
5. Clean generated case queries before execution so meta text like outcome guesses or label names does not leak into retrieval.

## Runtime notes

- The maintained runtime path is `main.py -> src/pipeline/pipeline.py`.
- Query understanding is used both for the initial retrieval pass and for later round-based regeneration of law and case queries.
- Module A is not allowed to predict the final label; it only prepares retrieval intent.

## Guardrail prompt

Stored in `src/modules/query_understanding/prompts.py`. It requires JSON-only output, forbids outcome prediction, and asks for diverse high-value retrieval queries rather than final-answer reasoning.
