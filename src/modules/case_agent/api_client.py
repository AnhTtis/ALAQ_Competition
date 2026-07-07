from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    requests = None

from ...core.schema import CaseSegment
from ...core.text_utils import normalized_query


@dataclass(frozen=True)
class CaseApiCacheEntry:
    case_id: str
    query_key: str
    query: str
    segments: list[dict[str, Any]]
    created_at: float
    api_mode: str = ""
    base_url: str = ""


class CaseRetrievalClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        api_mode: str,
        cache_path: Path,
        min_seconds_between_requests: float = 5.0,
        api_rate_limit_safety_seconds: float = 0.5,
        retry_after_default_seconds: float = 60.0,
        max_api_retries_per_query: int = 1,
        dry_run_cache_only: bool = False,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.api_mode = api_mode
        self.cache_path = cache_path
        self.min_seconds_between_requests = min_seconds_between_requests
        self.api_rate_limit_safety_seconds = api_rate_limit_safety_seconds
        self.retry_after_default_seconds = retry_after_default_seconds
        self.max_api_retries_per_query = max_api_retries_per_query
        self.dry_run_cache_only = dry_run_cache_only
        self.last_request_at: float | None = None
        self.calls_made = 0
        self._cache = self._load_cache()

    def search(self, case_id: str, query: str, *, retries: int | None = None) -> list[CaseSegment]:
        key = self._cache_key(case_id, query)
        legacy_key = self._cache_key(case_id, query, "", "")
        if key in self._cache:
            return self._segments_from_cache(self._cache[key], query)
        if legacy_key in self._cache:
            return self._segments_from_cache(self._cache[legacy_key], query)
        if self.dry_run_cache_only or not self.api_key:
            return []
        response = self._request(case_id, query, retries=self.max_api_retries_per_query if retries is None else retries)
        segments = self._parse_segments(response, query)
        self._save_cache_entry(case_id, query, segments)
        return segments

    def _request(self, case_id: str, query: str, *, retries: int) -> Any:
        if requests is None:
            raise RuntimeError("The requests package is required when case API calls are enabled.")
        url, headers = self._request_spec()
        payload = {"case_id": case_id, "query": query}
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            self._wait_rate_limit()
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            self.calls_made += 1
            if response.status_code == 429 and attempt < retries:
                time.sleep(self._retry_after_seconds(response))
                continue
            try:
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(self.min_seconds_between_requests + self.api_rate_limit_safety_seconds)
                    continue
        if last_error:
            raise last_error
        return {}

    def _request_spec(self) -> tuple[str, dict[str, str]]:
        if self.api_mode == "bearer_search":
            return (
                self.base_url + "/v1/case_segments/search",
                {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
        return self.base_url + "/retrieve", {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    def _wait_rate_limit(self) -> None:
        now = time.monotonic()
        if self.last_request_at is not None:
            min_gap = self.min_seconds_between_requests + self.api_rate_limit_safety_seconds
            remaining = min_gap - (now - self.last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_at = time.monotonic()

    def _retry_after_seconds(self, response: requests.Response) -> float:
        raw = response.headers.get("Retry-After")
        if raw:
            try:
                return max(float(raw), self.retry_after_default_seconds)
            except ValueError:
                pass
        return self.retry_after_default_seconds

    def _parse_segments(self, data: Any, query: str) -> list[CaseSegment]:
        hits = []
        if isinstance(data, dict):
            hits = data.get("results") or data.get("result") or []
        if isinstance(hits, dict):
            hits = [hits]
        if not isinstance(hits, list):
            return []
        segments: list[CaseSegment] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            chunk_id = str(hit.get("chunk_id") or hit.get("hash_id") or hit.get("id") or "").strip()
            text = str(hit.get("text") or hit.get("content") or "").strip()
            score = float(hit.get("score") or 0.0)
            if chunk_id and text:
                segments.append(CaseSegment(chunk_id=chunk_id, text=text, score=score, query=query))
        return segments

    def _load_cache(self) -> dict[str, CaseApiCacheEntry]:
        cache: dict[str, CaseApiCacheEntry] = {}
        if not self.cache_path.exists():
            return cache
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                entry = CaseApiCacheEntry(**raw)
                cache[self._cache_key(entry.case_id, entry.query, entry.api_mode, entry.base_url)] = entry
            except Exception:
                continue
        return cache

    def _save_cache_entry(self, case_id: str, query: str, segments: list[CaseSegment]) -> None:
        entry = CaseApiCacheEntry(
            case_id=case_id,
            query_key=normalized_query(query),
            query=query,
            segments=[asdict(segment) for segment in segments],
            created_at=time.time(),
            api_mode=self.api_mode,
            base_url=self.base_url,
        )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        self._cache[self._cache_key(case_id, query)] = entry

    def _cache_key(self, case_id: str, query: str, api_mode: str | None = None, base_url: str | None = None) -> str:
        mode = api_mode if api_mode is not None else self.api_mode
        base = (base_url if base_url is not None else self.base_url).rstrip("/")
        return f"{mode}\t{base}\t{case_id}\t{normalized_query(query)}"

    def _segments_from_cache(self, entry: CaseApiCacheEntry, query: str) -> list[CaseSegment]:
        segments: list[CaseSegment] = []
        for raw in entry.segments:
            item = dict(raw)
            item["query"] = query
            segments.append(CaseSegment(**item))
        return segments

