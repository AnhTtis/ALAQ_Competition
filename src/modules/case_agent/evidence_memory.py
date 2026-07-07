from __future__ import annotations

from dataclasses import dataclass, field

from ...core.schema import CaseSegment
from ...core.text_utils import normalized_query


@dataclass
class EvidenceMemory:
    case_id: str
    queries: list[str] = field(default_factory=list)
    query_families: list[str] = field(default_factory=list)
    query_new_segments: list[int] = field(default_factory=list)
    query_result_chunks: list[list[str]] = field(default_factory=list)
    segments: list[CaseSegment] = field(default_factory=list)
    seen_chunk_ids: set[str] = field(default_factory=set)
    seen_text_keys: set[str] = field(default_factory=set)
    no_new_count: int = 0
    api_calls: int = 0
    decision_found: bool = False

    def add_query(self, query: str, *, family: str = "general") -> None:
        self.queries.append(query)
        self.query_families.append(family)

    def add_query_result(self, segments: list[CaseSegment], added: int) -> None:
        self.query_result_chunks.append([segment.chunk_id for segment in segments])
        self.query_new_segments.append(added)

    def add_segments(self, segments: list[CaseSegment]) -> int:
        added = 0
        for segment in segments:
            text_key = normalized_query(segment.text)
            if segment.chunk_id in self.seen_chunk_ids or text_key in self.seen_text_keys:
                continue
            self.seen_chunk_ids.add(segment.chunk_id)
            if text_key:
                self.seen_text_keys.add(text_key)
            self.segments.append(segment)
            added += 1
        self.no_new_count = 0 if added else self.no_new_count + 1
        return added
