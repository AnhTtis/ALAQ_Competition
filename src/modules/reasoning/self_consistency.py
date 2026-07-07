from __future__ import annotations

import json
from collections import Counter, defaultdict

from ...core.config import Settings
from ...core.schema import CaseInput, CaseSegment, LawArticle, PredictionRecord
from .heuristics import decision_rule_prediction
from .reasoner import LegalReasoner


class SelfConsistencyReasoner:
    def __init__(self, reasoner: LegalReasoner, settings: Settings):
        self.reasoner = reasoner
        self.settings = settings

    def predict(
        self,
        *,
        case: CaseInput,
        case_segments: list[CaseSegment],
        law_articles: list[LawArticle],
        api_calls: int,
        runs: int | None = None,
        temperature: float | None = None,
    ) -> PredictionRecord:
        if not self.settings.enable_self_consistency:
            return self.reasoner.predict_once(
                case=case,
                case_segments=case_segments,
                law_articles=law_articles,
                api_calls=api_calls,
                temperature=self.settings.llm_temperature,
            )
        total_runs = max(1, runs if runs is not None else self.settings.self_consistency_runs)
        temp = temperature if temperature is not None else self.settings.self_consistency_temperature
        records = [
            self.reasoner.predict_once(case=case, case_segments=case_segments, law_articles=law_articles, api_calls=api_calls, temperature=temp)
            for _ in range(total_runs)
        ]
        winner = self._winner(records, case_segments)
        winning_records = [record for record in records if record.prediction == winner]
        selected = winning_records[0] if winning_records else records[0]
        selected.prediction = winner
        selected.confidence = _average_confidence(winning_records or records)
        selected.case_evidence = _merge_case_evidence(winning_records or records, case_segments)
        selected.law_evidence = _merge_law_evidence(winning_records or records, law_articles, self.settings.final_law_output_max)
        selected.raw_model_output = json.dumps(
            [
                {
                    "prediction": record.prediction,
                    "confidence": record.confidence,
                    "fallback_used": record.fallback_used,
                    "reasoning_summary": record.reasoning_summary,
                    "raw_model_output": record.raw_model_output,
                }
                for record in records
            ],
            ensure_ascii=False,
        )
        selected.fallback_used = any(record.fallback_used for record in records)
        return selected

    def _winner(self, records: list[PredictionRecord], case_segments: list[CaseSegment]) -> str:
        counts = Counter(record.prediction for record in records)
        most_common = counts.most_common()
        if most_common and (len(most_common) == 1 or most_common[0][1] > most_common[1][1]):
            return most_common[0][0]
        by_confidence: dict[str, list[float]] = defaultdict(list)
        for record in records:
            if record.confidence is not None:
                by_confidence[record.prediction].append(record.confidence)
        if by_confidence:
            return max(by_confidence.items(), key=lambda item: sum(item[1]) / max(len(item[1]), 1))[0]
        rule_prediction = decision_rule_prediction(case_segments)
        return rule_prediction or records[0].prediction


def _average_confidence(records: list[PredictionRecord]) -> float | None:
    values = [record.confidence for record in records if record.confidence is not None]
    return sum(values) / len(values) if values else None


def _merge_case_evidence(records: list[PredictionRecord], fallback: list[CaseSegment]) -> list[CaseSegment]:
    seen: set[str] = set()
    out: list[CaseSegment] = []
    for record in records:
        for segment in record.case_evidence:
            if segment.chunk_id not in seen:
                seen.add(segment.chunk_id)
                out.append(segment)
    for segment in fallback:
        if segment.chunk_id not in seen:
            seen.add(segment.chunk_id)
            out.append(segment)
    return out[:16]


def _merge_law_evidence(records: list[PredictionRecord], fallback: list[LawArticle], max_items: int) -> list[LawArticle]:
    seen: set[str] = set()
    out: list[LawArticle] = []
    for record in records:
        for article in record.law_evidence:
            if article.evidence_id not in seen:
                seen.add(article.evidence_id)
                out.append(article)
    for article in fallback:
        if article.evidence_id not in seen:
            seen.add(article.evidence_id)
            out.append(article)
    return out[:max_items]
