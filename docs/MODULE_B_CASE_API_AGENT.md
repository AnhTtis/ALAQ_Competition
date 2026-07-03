# Module B: Case API Agent

Code: `src/modules/case_agent/`

## Purpose

Strategically query the official ALQAC 2026 Case Content API and store retrieved Top-1 segments in Evidence Memory.

## Official API contract

- Endpoint: `POST https://alqac-api.ngrok.pro/retrieve`.
- Auth: `X-API-Key: <team token>`.
- Body: `{"case_id": "...", "query": "..."}`.
- Response: exactly one Top-1 segment per call, with `chunk_id`, `score`, and `text`.
- Rate limit: 1 request every 5 seconds per team.

## Key files

- `api_client.py`: cache, auth, endpoint mode, retries, and 5-second rate limiting.
- `evidence_memory.py`: query/chunk memory and stop-state fields.
- `query_strategy.py`: high-value query construction.
- `scoring.py`: decision/verdict signal scoring.
- `agent_case.py`: sequential control loop.

## Stop conditions

The agent stops when one of these is true:

- `MAX_CASE_API_CALLS` is reached.
- Minimum calls are reached and decision/verdict evidence is found.
- Minimum calls are reached and too many queries return no new chunk.

The local call count is used for diagnostics and budget control only. Official scoring uses BTC server logs; `api_calls` is not exported in `submission.json`.

## Query priorities

1. Quyết định / tuyên xử.
2. Chấp nhận yêu cầu.
3. Không chấp nhận / bác yêu cầu.
4. Chấp nhận một phần.
5. Nghĩa vụ thanh toán, bồi thường, giao trả.
6. Phản tố / yêu cầu độc lập.
7. Án phí / chi phí tố tụng.
8. Căn cứ pháp lý và nhận định của tòa.

## Scoring implication

The 20% Penalized Case Recall component rewards correct evidence but penalizes excessive official API calls. The BTC efficiency factor gives full credit up to `2*n_i` calls and decays to zero at `5*n_i`, so query generation should be targeted rather than exhaustive.
