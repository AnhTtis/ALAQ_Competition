from __future__ import annotations

from dataclasses import dataclass, field

from ...core.schema import CaseSegment


@dataclass
class EvidenceMemory:
    case_id: str
    queries: list[str] = field(default_factory=list)
    segments: list[CaseSegment] = field(default_factory=list)
    seen_chunk_ids: set[str] = field(default_factory=set)
    no_new_count: int = 0
    api_calls: int = 0
    decision_found: bool = False

    def add_query(self, query: str) -> None:
        self.queries.append(query)

    def add_segments(self, segments: list[CaseSegment]) -> int:
        added = 0
        for segment in segments:
            if segment.chunk_id in self.seen_chunk_ids:
                continue
            self.seen_chunk_ids.add(segment.chunk_id)
            self.segments.append(segment)
            added += 1
        self.no_new_count = 0 if added else self.no_new_count + 1
        return added
