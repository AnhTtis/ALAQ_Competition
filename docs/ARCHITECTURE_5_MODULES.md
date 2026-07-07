# ALQAC 5-Module Legal RAG Architecture

This architecture targets the latest ALQAC 2026 BTC task: four-label Vietnamese legal outcome prediction with required case-evidence and law-provision retrieval.

## Data flow

```text
public_test.json
  └─ case_id + case_query
      └─ Module A: Query Understanding
          ├─ dispute type, A/B claims, legal keywords
          └─ round query planner
              └─ MAX_RAG_ROUNDS alternating retrieval rounds
                  ├─ Module A generates law-corpus queries
                  ├─ Module C retrieves top law articles
                  │   ├─ exact article-reference boost
                  │   ├─ BM25
                  │   ├─ BGE-M3 dense retrieval
                  │   └─ multi-query cross-encoder rerank
                  ├─ Module A generates cleaned Case API queries from top laws
                  └─ Module B calls official /retrieve API
                      ├─ X-API-Key auth
                      ├─ Top-1 segment per call, paced at 5s+ per request
                      └─ EvidenceMemory dedupes chunk ids/text and tracks local call count
                          └─ Module D: Legal Reasoning
                              ├─ prompt input is only case_query + case_evidence + law_evidence
                              ├─ strict JSON prediction over 4 official labels
                              └─ cite only provided chunk_id + {law_id, aid}
                                  └─ Module E: Self-consistency
                                      └─ local open-weight runs + majority vote
                                          └─ submission.json
```

Default retrieval settings in `src/core/config.py`:

```text
MAX_RAG_ROUNDS=2
CASE_API_CALLS_PER_ROUND=4
ROUND_LAW_TOP_K=2
MAX_CASE_API_CALLS=8
FINAL_EVIDENCE_TOP_K=12
LAW_EVIDENCE_FOR_PROMPT=10
FINAL_LAW_OUTPUT_MAX=8
LLM_MODEL_ID=AITeamVN/Vi-Qwen2-7B-RAG
LLM_BACKEND=hf_transformers
LLM_TORCH_DTYPE=float16
```

## Official output contract

`submission.json` must be one JSON array with exactly one item per test case:

```json
{
  "case_id": "case_4101",
  "prediction": "A_WIN",
  "case_evidence": ["case_4101_chunk_3"],
  "law_evidence": [
    {"law_id": "47/2010/QH12", "aid": 270}
  ]
}
```

Rules reflected in the pipeline:

- `prediction` is one of `A_WIN`, `PARTIAL_A_WIN`, `PARTIAL_B_WIN`, `B_WIN`.
- `case_evidence` is required; empty list is valid but scores zero for case recall.
- `law_evidence` uses corpus article `{law_id, aid}` pairs from `corpus_law_pub.json`.
- `api_calls` is tracked locally for diagnostics only and is not exported.

## Scoring impact

```text
FinalScore = 0.70 * OutcomeAccuracy + 0.20 * PenalizedCaseRecall + 0.10 * LawF1micro
```

Module priorities follow the metric weights:

1. Reasoning and voting must optimize exact-match outcome accuracy across all four labels.
2. Module B must retrieve enough decisive case chunks while avoiding excessive API calls.
3. Module C must return valid corpus law articles for micro-F1.

The API-efficiency factor gives full credit up to `2*n_i` calls and decays to zero at `5*n_i`, where BTC measures call counts from server logs. The code default `MAX_CASE_API_CALLS=6` matches the current 2 rounds × 3 case queries budget in `src/core/config.py`.

## Module B behavior

The Case API returns exactly one top-ranked segment per query, so Module B does not fire random queries. In the round loop, Module A generates targeted queries from the current top laws and accumulated case evidence, then Module B executes only the current round's query batch.

- If a new `chunk_id` appears, it is stored and ranked.
- If the chunk contains decision/verdict signals, `decision_found` becomes true.
- Local `api_calls` is used for budget control and diagnostics only.
- The official leaderboard uses server-side API logs, not submitted `api_calls`.

## Package map

- `src/core/`: config, schemas, loaders, evaluation, text utils.
- `src/modules/query_understanding/`: Module A, initial analysis and round query generation.
- `src/modules/case_agent/`: Module B.
- `src/modules/law_retrieval/`: Module C.
- `src/modules/reasoning/`: Modules D/E.
- `src/pipeline/`: orchestration layer.
- `main.py`: primary CLI entrypoint.

## Competition constraints

- Use only open-weight models under 10B parameters for official runs.
- Do not use closed/proprietary model APIs.
- Do not use externally annotated legal QA or legal entailment datasets.
- Keep the team token under `token/` and never commit it.
