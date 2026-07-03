from __future__ import annotations

import json
from pathlib import Path

from ..core.schema import PredictionRecord


def write_trace(path: Path, records: list[PredictionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        rows.append(
            {
                "case_id": record.case_id,
                "prediction": record.prediction,
                "api_calls": record.api_calls,
                "law_evidence": [
                    {"law_id": a.law_id, "aid": a.aid, "score": a.score, "source": a.source}
                    for a in record.law_evidence
                ],
                "case_evidence": [
                    {"chunk_id": s.chunk_id, "score": s.score, "query": s.query}
                    for s in record.case_evidence
                ],
            }
        )
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
