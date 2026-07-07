# Modules D/E: Legal Reasoning and Self-Consistency

Code: `src/modules/reasoning/`

## Module D: LegalReasoner

`LegalReasoner.predict_once(...)` builds a strict Vietnamese prompt and asks the local LLM to output JSON using only `case_query`, `case_evidence`, and `law_evidence` as final reasoning inputs.

Expected JSON fields:

- `prediction`
- `confidence`
- `case_evidence`
- `law_evidence`
- `reasoning_summary`

The prompt requires citing provided `chunk_id`, `aid`, and `law_id` only. It compares all four labels and prioritizes decision/verdict chunks over party claims.

## Module E: SelfConsistencyReasoner

`SelfConsistencyReasoner.predict(...)` runs Module D 3 times by default with temperature > 0 and votes by majority.

Tie-break order:

1. Higher average confidence.
2. Deterministic decision-rule heuristic.
3. First valid run.

Evidence from winning runs is merged first, then top retrieved evidence is used as fallback.

## Guardrail prompt

Stored in `src/modules/reasoning/prompts.py`. It forbids hallucinated evidence IDs and requires strict JSON-only output.
