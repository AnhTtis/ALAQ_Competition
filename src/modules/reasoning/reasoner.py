from __future__ import annotations

import json

from ...core.config import Settings
from ...core.schema import OUTCOME_LABELS, CaseInput, CaseSegment, LawArticle, PredictionRecord
from ...core.text_utils import truncate
from ...llm.base import LLMClient
from .heuristics import decision_rule_prediction, heuristic_prediction
from .parsing import float_or_none, parse_prediction_payload, selected_case_evidence, selected_law_evidence
from .prompts import FINAL_REASONING_SYSTEM_PROMPT


class LegalReasoner:
    def __init__(self, llm: LLMClient, settings: Settings):
        self.llm = llm
        self.settings = settings

    def predict_once(
        self,
        *,
        case: CaseInput,
        case_segments: list[CaseSegment],
        law_articles: list[LawArticle],
        api_calls: int,
        temperature: float | None = None,
    ) -> PredictionRecord:
        payload = {
            "case_query": case.case_query,
            "case_evidence": [
                {"chunk_id": s.chunk_id, "text": truncate(s.text, self.settings.max_case_text_chars), "score": s.score}
                for s in case_segments[: self.settings.case_evidence_for_prompt]
            ],
            "law_evidence": [
                {
                    "law_id": a.law_id,
                    "aid": a.aid,
                    "article_no": a.article_no,
                    "text": truncate(a.text, self.settings.max_law_text_chars),
                    "score": a.score,
                }
                for a in law_articles[: self.settings.law_evidence_for_prompt]
            ],
        }
        try:
            raw = self.llm.generate(
                [
                    {"role": "system", "content": FINAL_REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                json_mode=True,
                max_new_tokens=self.settings.llm_max_new_tokens,
                temperature=self.settings.llm_temperature if temperature is None else temperature,
            )
            parsed = parse_prediction_payload(raw)
            fallback_used = False
        except Exception as exc:
            raw = f"heuristic fallback after {type(exc).__name__}: {exc}"
            parsed = {}
            fallback_used = True
        prediction = parsed.get("prediction") if isinstance(parsed.get("prediction"), str) else None
        llm_prediction = prediction
        rule_prediction = None
        override_reason = ""
        if self.settings.enable_decision_rule_override:
            rule_prediction = decision_rule_prediction(case_segments)
            if rule_prediction:
                if prediction != rule_prediction:
                    fallback_used = True
                    override_reason = f"decision_rule:{prediction}->{rule_prediction}"
                prediction = rule_prediction
        if prediction not in OUTCOME_LABELS:
            fallback_prediction = heuristic_prediction(case.case_query, case_segments)
            override_reason = override_reason or f"heuristic_fallback:{prediction}->{fallback_prediction}"
            prediction = fallback_prediction
            fallback_used = True
        law_evidence = _backfill_law_evidence(
            selected_law_evidence(parsed, law_articles),
            law_articles,
            min_items=self.settings.final_law_output_min,
            max_items=self.settings.final_law_output_max,
        )
        return PredictionRecord(
            case_id=case.case_id,
            prediction=prediction,
            law_evidence=law_evidence,
            case_evidence=selected_case_evidence(parsed, case_segments),
            api_calls=api_calls,
            confidence=float_or_none(parsed.get("confidence")),
            reasoning_summary=str(parsed.get("reasoning_summary") or parsed.get("rationale") or ""),
            raw_model_output=raw,
            fallback_used=fallback_used,
            llm_prediction=llm_prediction,
            decision_rule_prediction=rule_prediction,
            override_reason=override_reason,
        )


def _backfill_law_evidence(
    selected: list[LawArticle],
    fallback: list[LawArticle],
    *,
    min_items: int,
    max_items: int,
) -> list[LawArticle]:
    seen: set[str] = set()
    out: list[LawArticle] = []
    for article in selected:
        if article.evidence_id not in seen:
            seen.add(article.evidence_id)
            out.append(article)
        if len(out) >= max_items:
            return out[:max_items]
    for article in fallback:
        if len(out) >= max(min_items, 0):
            break
        if article.evidence_id not in seen:
            seen.add(article.evidence_id)
            out.append(article)
    return out[:max_items]
