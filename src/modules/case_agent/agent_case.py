from __future__ import annotations

from ...core.config import Settings
from ...core.schema import CaseInput, CaseUnderstanding
from ...core.text_utils import compact_text, normalize_text
from .api_client import CaseRetrievalClient
from .evidence_memory import EvidenceMemory
from .query_strategy import build_case_queries
from .scoring import has_decision_signal, rank_case_segments


class CaseApiAgent:
    def __init__(self, client: CaseRetrievalClient, settings: Settings):
        self.client = client
        self.settings = settings

    def collect_evidence(self, case: CaseInput, understanding: CaseUnderstanding) -> EvidenceMemory:
        memory = EvidenceMemory(case_id=case.case_id)
        calls_before = self.client.calls_made
        queries = build_case_queries(understanding, max_queries=max(self.settings.max_case_api_calls, 5))
        for query in queries:
            api_calls_so_far = self.client.calls_made - calls_before
            if api_calls_so_far >= self.settings.max_case_api_calls:
                break
            if self._can_stop(memory, api_calls_so_far):
                break
            query = compact_text(query)
            if not query:
                continue
            memory.add_query(query)
            remaining_calls = self.settings.max_case_api_calls - api_calls_so_far
            retries = max(0, min(self.settings.max_api_retries_per_query, remaining_calls - 1))
            try:
                segments = self.client.search(case.case_id, query, retries=retries)
            except Exception as exc:
                memory.no_new_count += 1
                print(f"case retrieval failed for {case.case_id}: {type(exc).__name__}: {exc}", flush=True)
                continue
            memory.add_segments(segments)
            if any(has_decision_signal(segment.text) for segment in segments):
                memory.decision_found = True
        memory.api_calls = self.client.calls_made - calls_before
        memory.segments = rank_case_segments(memory.segments)
        return memory

    def run_queries(self, case: CaseInput, memory: EvidenceMemory, queries: list[str], *, max_attempts: int) -> int:
        attempted = 0
        seen_queries = {normalize_text(query) for query in memory.queries}
        calls_before = self.client.calls_made
        for raw_query in queries:
            if attempted >= max_attempts:
                break
            if memory.api_calls >= self.settings.max_case_api_calls:
                break
            query = compact_text(raw_query)
            query_key = normalize_text(query)
            if not query or query_key in seen_queries:
                continue
            seen_queries.add(query_key)
            memory.add_query(query)
            attempted += 1
            remaining_calls = self.settings.max_case_api_calls - memory.api_calls
            retries = max(0, min(self.settings.max_api_retries_per_query, remaining_calls - 1))
            try:
                segments = self.client.search(case.case_id, query, retries=retries)
            except Exception as exc:
                memory.no_new_count += 1
                memory.api_calls += self.client.calls_made - calls_before
                calls_before = self.client.calls_made
                print(f"case retrieval failed for {case.case_id}: {type(exc).__name__}: {exc}", flush=True)
                continue
            memory.api_calls += self.client.calls_made - calls_before
            calls_before = self.client.calls_made
            memory.add_segments(segments)
            if any(has_decision_signal(segment.text) for segment in segments):
                memory.decision_found = True
        memory.segments = rank_case_segments(memory.segments)
        return attempted

    def _can_stop(self, memory: EvidenceMemory, api_calls_so_far: int) -> bool:
        if api_calls_so_far < self.settings.min_case_api_calls:
            return False
        if memory.decision_found and memory.segments:
            return True
        return memory.no_new_count >= self.settings.max_no_new_segment_queries
