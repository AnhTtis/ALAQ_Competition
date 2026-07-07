# ALQAC 2026 Submission Runbook

## Current rule source

See `docs/COMPETITION_RULES.md` for the latest BTC rules reflected in this repo.

## Pipeline flow

The local pipeline uses a prompt-aligned alternating retrieval loop before final prediction:

1. Analyze `case_query` once.
2. Build initial cleaned Case API queries.
3. For each round:
   - Generate law-corpus queries with the local LLM or deterministic fallback.
   - Retrieve top law articles from `corpus_law_pub.json`.
   - Generate cleaned Case API queries using the current top laws and retrieved case chunks.
   - Call the official Top-1 Case API and accumulate/dedupe evidence.
4. Run final self-consistency voting over the accumulated case evidence and law evidence.
5. Export `submission.json`.

Default knobs in `src/core/config.py`:

```text
MAX_RAG_ROUNDS=2
CASE_API_CALLS_PER_ROUND=4
ROUND_LAW_TOP_K=2
MAX_CASE_API_CALLS=8
FINAL_EVIDENCE_TOP_K=12
LAW_EVIDENCE_FOR_PROMPT=10
FINAL_LAW_OUTPUT_MAX=8
LLM_MODEL_ID=Qwen/Qwen3-8B
LLM_BACKEND=hf_transformers
LLM_TORCH_DTYPE=float16
```

## Submission format

The pipeline writes `runs/outputs/submission.json` as one JSON array:

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

Important official requirements:

- One object per official test case, exactly once.
- `prediction` is one of `A_WIN`, `PARTIAL_A_WIN`, `PARTIAL_B_WIN`, `B_WIN`.
- `case_evidence` is required; an empty list is accepted but scores zero for case recall.
- `law_evidence` is a list of `{law_id, aid}` objects from `corpus_law_pub.json`, not string aids.
- `aid` is the corpus article id, not article position and not free text.
- `api_calls` is not submitted; BTC counts official API calls from retrieval-server logs.

## Token layout

Keep all local secrets in `token/`:

```text
token/.env       # ALQAC_API_KEY=alqac_xxx
token/hf.txt     # optional Hugging Face token
token/alqac.txt  # optional raw ALQAC token fallback
```

`token/` is ignored by git. `src/core/config.py` loads `token/.env` first, then falls back to root `.env` if present. Root `hf.txt` and `alqac.txt` are also ignored in case those fallback locations are used.

## Case API behavior

- Endpoint: `POST https://alqac-api.ngrok.pro/retrieve`.
- Auth header: `X-API-Key: <team token>`.
- Body: `{"case_id": "...", "query": "..."}`.
- Response: exactly one Top-1 segment in `results` per call.
- Rate limit: 1 request every 5 seconds per team.

The 20% case-evidence score is recall multiplied by an API-efficiency factor. Full efficiency is kept up to `2*n_i` calls and decays to zero at `5*n_i`, where `n_i` is the number of case segments.

## How BTC scores API usage without `api_calls`

`api_calls` is intentionally absent from `submission.json`. BTC computes `c_i` from the official retrieval API server logs by matching your team token, `case_id`, and requests made during the run. The submitted `case_evidence` ids are compared with gold evidence, then that recall is multiplied by the log-derived efficiency factor.

Local `api_calls` in `metrics.json` and `retrieval_trace.jsonl` is only for debugging your own budget; it is not trusted by the leaderboard.

## Cache and outputs

- Case API cache: `runs/cache/case_api_cache.jsonl`
- Dense law embedding cache: `runs/cache/bge_law_embeddings.npz`
- Outputs: `runs/outputs/`
- HF/model cache: `hf_cache/`

## Install

On GPU machines, install the CUDA-compatible PyTorch wheel first, then install the rest:

```bash
pip install -U torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Use a different PyTorch command if your CUDA version requires it. Do not add closed/proprietary model SDKs for official runs.

## Run commands

Debug without GPU/API:

```bash
python main.py --limit 1 --no-api --dry-run-cache-only --print-metrics
```

Production-style local open-weight model run with final voting:

```bash
python main.py --gpu-id 0 --self-consistency-runs 3 --print-metrics
```

Multi-GPU run (Transformers auto-shards across the listed visible GPUs):

```bash
python main.py --gpus 0,1 --self-consistency-runs 3 --print-metrics
```

Cache-only run:

```bash
python main.py --dry-run-cache-only --print-metrics
```

## Pre-submission checklist

1. `submission.json` is valid JSON and contains every official test case exactly once.
2. `prediction` uses one of the four official labels.
3. `case_evidence` exists for every case and contains official `chunk_id`s only.
4. `law_evidence` items are `{law_id, aid}` objects from `corpus_law_pub.json`.
5. `api_calls` is absent from `submission.json`.
6. Inspect `runs/outputs/retrieval_trace.jsonl` for cases with empty or weak evidence.
7. Official runs use only allowed open-weight models under 10B parameters.
8. Respect the current leaderboard limit: 20 submissions per team per 24 hours.
