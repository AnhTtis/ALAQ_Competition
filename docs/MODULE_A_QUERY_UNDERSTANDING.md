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

## Guardrail prompt

Stored in `src/modules/query_understanding/prompts.py`. It requires JSON-only output, forbids outcome prediction, and asks for 5-15 diverse evidence-search queries.
