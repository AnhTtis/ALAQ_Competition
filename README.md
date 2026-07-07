# ALQAC 2026 Legal Case Outcome Pipeline

This repository implements a modular local Legal RAG system for ALQAC 2026: Vietnamese legal case outcome prediction with evidence retrieval.

## Current BTC task assumptions

- Single task: Legal Case Outcome Prediction with Evidence Retrieval.
- Input: `case_id` and short `case_query`; the query does not reveal the verdict or gold evidence.
- Prediction labels: `A_WIN`, `PARTIAL_A_WIN`, `PARTIAL_B_WIN`, `B_WIN`.
- Case evidence comes from the official Top-1 `/retrieve` API using `X-API-Key` and 5-second pacing.
- Law evidence comes from `data/corpus_law_pub.json` as `{law_id, aid}` pairs.
- Final score: `0.70 * OutcomeAccuracy + 0.20 * PenalizedCaseRecall + 0.10 * LawF1micro`.
- Official runs must use open-weight models under 10B parameters; closed/proprietary model APIs are prohibited.

## Current flow

The maintained pipeline is a prompt-aligned alternating RAG flow followed by final voting:

```text
case_id + case_query
  -> Module A: initial query understanding
  -> initial cleaned Case API queries
  -> MAX_RAG_ROUNDS alternating retrieval rounds:
       1. LLM/fallback generates law-corpus queries
       2. Module C retrieves top law articles
       3. LLM/fallback generates cleaned Case API queries using those laws
       4. Module B calls the official Top-1 Case API and accumulates/dedupes case chunks
  -> Module D/E: final prompt sees only case_query + case_evidence + law_evidence
  -> self-consistency majority vote
  -> runs/outputs/submission.json
```

Default retrieval knobs in `src/core/config.py`:

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

`api_calls` is tracked locally for debugging but is not exported in `submission.json`; BTC computes official API usage from server logs.

## What has been done

- Refactored the old flat `src/` layout into readable packages:
  - `src/core/`: config, schema, loaders, evaluation, text utils.
  - `src/modules/query_understanding/`: Module A, case query analysis and round query generation.
  - `src/modules/case_agent/`: Module B, strategic Case API agent and Evidence Memory.
  - `src/modules/law_retrieval/`: Module C, BM25 + BGE-M3 dense retrieval + reranker.
  - `src/modules/reasoning/`: Modules D/E, legal reasoning and self-consistency voting.
  - `src/pipeline/`: orchestration and trace writing.
- Updated submission export to match the latest BTC format:
  - `prediction`: one of the four official labels.
  - `law_evidence`: `{law_id, aid}` objects.
  - `case_evidence`: required list of chunk IDs.
  - no `api_calls` field in `submission.json`.
- Moved local secrets into `token/`.
- Moved competition/rule docs into `docs/`.
- Kept `main.py` as the only maintained CLI entrypoint.
- Removed notebook-only dependencies from `requirements.txt`; notebooks are reference material, not runtime pipeline inputs.

## Current directory layout

```text
data/                       # official input and law corpus
src/core/                   # shared primitives
src/modules/query_understanding/
src/modules/case_agent/
src/modules/law_retrieval/
src/modules/reasoning/
src/pipeline/
docs/                       # architecture, module docs, BTC rules, runbook
token/                      # local secrets, ignored by git
runs/                       # runtime outputs/cache, ignored by git
hf_cache/                   # local HF/model cache, ignored by git
```

Runtime pipeline code lives under `src/` plus `main.py`. Notebooks are not imported by the maintained pipeline.

## Install

On GPU machines, install the CUDA-compatible PyTorch wheel first, then install the rest:

```bash
pip install -U torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If your CUDA/PyTorch build is different, use the matching PyTorch install command for that machine.

## Token setup

Create one of these local files:

```text
token/.env       # ALQAC_API_KEY=alqac_xxx
token/alqac.txt  # raw ALQAC token fallback
token/hf.txt     # optional Hugging Face token
```

Do not commit tokens. `token/`, `.env`, root `hf.txt`, and root `alqac.txt` are ignored.

## Run

Smoke test without API/model:

```bash
python main.py --limit 1 --no-api --dry-run-cache-only --print-metrics
```

Production-style single-GPU run with the default local Qwen3 model and voting:

```bash
python main.py --gpu-id 0 --self-consistency-runs 3 --print-metrics
```

Production-style multi-GPU run (let Transformers shard automatically across the visible GPUs):

```bash
python main.py --gpus 0,1 --self-consistency-runs 3 --print-metrics
```

Run with cached Case API responses only, useful when iterating on reasoning/retrieval code without spending API calls:

```bash
python main.py --limit 1 --dry-run-cache-only --print-metrics
```

Optional speed/debug switches:

```bash
python main.py --limit 5 --gpu-id 0 --no-dense --no-rerank --self-consistency-runs 1 --print-metrics
```

Useful tuning overrides for cases where chunk evidence is too thin or law evidence is too broad:

```bash
python main.py \
  --gpu-id 0 \
  --self-consistency-runs 3 \
  --max-case-api-calls 8 \
  --case-api-calls-per-round 4 \
  --round-law-top-k 2 \
  --final-evidence-top-k 12 \
  --law-evidence-for-prompt 10 \
  --final-law-output-max 8 \
  --print-metrics
```

## Outputs

- `runs/outputs/submission.json`: official submission file.
- `runs/outputs/metrics.json`: local diagnostic metrics.
- `runs/outputs/retrieval_trace.jsonl`: retrieved evidence trace.
- `runs/outputs/model_debug_outputs.jsonl`: raw reasoning outputs.

## Notes

- The maintained runtime path is `main.py` -> `src/` only.
- `reference/` and the Colab notebooks are archival/reference assets; they are not required for install or official pipeline runs.
- Final reasoning is intentionally aligned with `FINAL_REASONING_SYSTEM_PROMPT`: the model receives `case_query`, `case_evidence`, and `law_evidence` only, then returns JSON for the official labels and evidence IDs.
- Qwen3 defaults now disable thinking mode for JSON tasks and strip any residual `<think>...</think>` wrapper text before parsing.
