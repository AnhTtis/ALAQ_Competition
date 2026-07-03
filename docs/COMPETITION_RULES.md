# ALQAC 2026 Official Rules Summary

ALQAC 2026 is the Automated Legal Question Answering Competition associated with KSE 2026, held in Kanazawa, Japan on 11-14 November 2026.

This file tracks the latest BTC task/scoring documentation used by this repo. The leaderboard/task page is treated as authoritative when older overview text conflicts with it.

## Task

Single shared task: **Legal Case Outcome Prediction with Evidence Retrieval**.

For each public test item, the system receives only:

```json
{"case_id": "case_4101", "case_query": "..."}
```

The query is a short Vietnamese natural-language dispute description. It does not reveal the court reasoning, final decision, winner, gold case evidence, or gold law evidence.

The system must:

1. Read the provided case query.
2. Call the official Case Content API to retrieve relevant case segments.
3. Retrieve relevant legal provisions from `data/corpus_law_pub.json`.
4. Predict the final case outcome.
5. Submit the prediction with supporting `case_evidence` and `law_evidence`.

## Prediction labels

Outcome accuracy is exact-match over four labels:

- `A_WIN`: the court fully accepts all plaintiff claims.
- `PARTIAL_A_WIN`: the court partially accepts plaintiff claims and the accepted portion is greater than 50%.
- `PARTIAL_B_WIN`: the court partially accepts plaintiff claims, but the accepted portion is 50% or less.
- `B_WIN`: the court fully rejects all plaintiff claims.

If a query contains multiple claims, focus on the main claim described in `case_query`.

## Case Retrieval API

Endpoint:

```text
POST https://alqac-api.ngrok.pro/retrieve
```

Headers:

```text
X-API-Key: <team token>
Content-Type: application/json
```

Body:

```json
{"case_id": "case_1087_0037", "query": "tranh chấp quyền sử dụng đất"}
```

Response returns exactly Top-1 segment per call:

```json
{
  "results": [
    {
      "score": 0.886,
      "text": "Người có quyền lợi nghĩa vụ liên quan: ...",
      "chunk_id": "case_1087_0037_chunk_2"
    }
  ]
}
```

Rules:

- Every request must include the team token in `X-API-Key`; missing/invalid tokens return 403.
- Rate limit is 1 request every 5 seconds per team; exceeding it returns 429.
- Issue multiple targeted queries to gather more evidence because each call returns only one segment.
- Do not submit API call counts; BTC measures them from server logs.

## Official submission format

Upload one JSON array named `submission.json`, with exactly one object per test case:

```json
[
  {
    "case_id": "case_4101",
    "prediction": "A_WIN",
    "case_evidence": ["case_4101_chunk_3"],
    "law_evidence": [
      {"law_id": "47/2010/QH12", "aid": 270}
    ]
  }
]
```

Validation rules:

- Every official test `case_id` must appear exactly once.
- No duplicate `case_id`s.
- `prediction` must be one of `A_WIN`, `PARTIAL_A_WIN`, `PARTIAL_B_WIN`, `B_WIN`.
- `case_evidence` is required; an empty list is allowed but scores zero for case recall.
- `law_evidence` is required and must be a list of `{law_id, aid}` objects from `corpus_law_pub.json`.
- `aid` is the corpus article `aid`, not the article position and not a free-text string.
- Duplicate evidence ids may be de-duplicated before scoring.
- `explanation` is optional and not a principal scoring component.
- Do not include `api_calls` in `submission.json`.

## Score

```text
FinalScore = 0.70 * OutcomeAccuracy + 0.20 * PenalizedCaseRecall + 0.10 * LawF1micro
```

Components:

- **Outcome Accuracy (70%)**: exact-match prediction over the four labels.
- **Penalized Case Evidence Recall (20%)**: recall of gold case-content evidence, multiplied by the API-efficiency factor.
- **Micro Law Evidence F1 (10%)**: micro-averaged F1 over submitted law provisions for the full test set.

API-efficiency factor for case `i`:

```text
E_i = max(0, 1 - max(0, c_i - 2*n_i) / (3*n_i))
```

Where:

- `c_i` is the API call count measured by BTC logs.
- `n_i` is the number of segments in the case.
- There is no penalty up to `2*n_i` calls.
- The factor decays to zero at `5*n_i` calls.

## Identity and leaderboard limits

- BTC sets up teams and issues each team a secret token.
- Keep the token private; it is required for retrieval and submission.
- The public leaderboard shows each team's best run.
- Current leaderboard documentation states submissions are limited to **20 per team per 24 hours**.

## Restrictions

- Closed/proprietary systems are prohibited, including ChatGPT, GPT-4, Claude, Gemini, and other non-open API-based models.
- Only open-weight models with fewer than 10 billion parameters are allowed.
- Externally annotated legal QA or legal entailment datasets are not allowed.
- Online legal databases may be queried, but not pre-labeled QA/entailment resources created for the task.
- Teams should be ready to provide a short technical report, source/config/logs, retrieval strategy, reasoning method, models/tools, prompts/agent design, and validation steps for reproducibility.
