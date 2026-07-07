from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import Settings
from .data_loader import parse_related_law_articles
from .schema import OUTCOME_LABELS, CaseGold, PredictionRecord


def evaluate(records: list[PredictionRecord], gold: dict[str, CaseGold], settings: Settings | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {"records": len(records)}
    if settings:
        metrics["runtime"] = _runtime_metrics(settings, records)
    labeled = [record for record in records if gold.get(record.case_id) and gold[record.case_id].verdict_label]
    if labeled:
        correct = sum(1 for record in labeled if record.prediction == gold[record.case_id].verdict_label)
        metrics["outcome_accuracy"] = correct / len(labeled)
        metrics["outcome_correct"] = correct
        metrics["outcome_total"] = len(labeled)
        metrics["confusion_matrix"] = _confusion_matrix(labeled, gold)
        metrics["per_label_accuracy"] = _per_label_accuracy(labeled, gold)
        metrics["per_label_prf"] = _per_label_prf(labeled, gold)
    metrics["prediction_counts"] = dict(Counter(record.prediction for record in records))
    metrics["api_calls_total"] = sum(record.api_calls for record in records)
    metrics["api_calls_by_case"] = {record.case_id: record.api_calls for record in records}
    metrics["case_query_attempts_total"] = sum(len(record.retrieval_queries) for record in records)
    metrics["case_query_attempts_by_case"] = {record.case_id: len(record.retrieval_queries) for record in records}
    metrics["case_segments_retrieved_total"] = sum(record.case_segments_retrieved for record in records)
    metrics["case_segments_retrieved_by_case"] = {record.case_id: record.case_segments_retrieved for record in records}
    metrics["case_evidence_counts"] = {record.case_id: len(record.case_evidence) for record in records}
    metrics["case_evidence_nonempty_rate"] = (
        sum(1 for record in records if record.case_evidence) / len(records) if records else 0.0
    )
    metrics["empty_case_evidence"] = [record.case_id for record in records if not record.case_evidence]
    metrics["no_new_case_queries_by_case"] = {record.case_id: record.no_new_case_queries for record in records}
    metrics["case_query_family_yield"] = _case_query_family_yield(records)
    metrics["fallback_count"] = sum(record.fallback_used for record in records)
    metrics["decision_rule_override_count"] = sum(1 for record in records if record.decision_rule_prediction and record.decision_rule_prediction != record.llm_prediction)
    metrics["heuristic_fallback_count"] = sum(1 for record in records if record.override_reason.startswith("heuristic_fallback"))
    law_overlap = _law_overlap_approx(records, gold)
    metrics["law_overlap_approx"] = law_overlap
    if "outcome_accuracy" in metrics:
        metrics["official_score_estimate"] = {
            "score_without_case_recall": 0.70 * metrics["outcome_accuracy"] + 0.10 * law_overlap["f1"],
            "penalized_case_recall": None,
            "note": "Local data does not include gold case chunk ids, so the 20% case-recall component is not estimated.",
        }
    metrics["submission_validation"] = _submission_validation(records, gold)
    return metrics


def _runtime_metrics(settings: Settings, records: list[PredictionRecord]) -> dict[str, Any]:
    is_full_run = len(records) > 0 and len(records) == 50
    is_mock = settings.llm_backend.lower() == "mock"
    return {
        "llm_backend": settings.llm_backend,
        "llm_model_id": settings.llm_model_id,
        "enable_dense_retrieval": settings.enable_dense_retrieval,
        "enable_cross_encoder_rerank": settings.enable_cross_encoder_rerank,
        "enable_self_consistency": settings.enable_self_consistency,
        "self_consistency_runs": settings.self_consistency_runs,
        "self_consistency_temperature": settings.self_consistency_temperature,
        "min_case_api_calls": settings.min_case_api_calls,
        "max_case_api_calls": settings.max_case_api_calls,
        "case_api_calls_per_round": settings.case_api_calls_per_round,
        "max_rag_rounds": settings.max_rag_rounds,
        "round_law_top_k": settings.round_law_top_k,
        "final_evidence_top_k": settings.final_evidence_top_k,
        "final_law_output_min": settings.final_law_output_min,
        "final_law_output_max": settings.final_law_output_max,
        "mock_benchmark_warning": is_full_run and is_mock,
    }


def write_outputs(*, output_dir: Path, records: list[PredictionRecord], metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = [record.to_submission() for record in records]
    (output_dir / "submission.json").write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    debug_rows = [
        {
            "case_id": record.case_id,
            "prediction": record.prediction,
            "llm_prediction": record.llm_prediction,
            "decision_rule_prediction": record.decision_rule_prediction,
            "override_reason": record.override_reason,
            "confidence": record.confidence,
            "fallback_used": record.fallback_used,
            "retrieval_queries": record.retrieval_queries,
            "retrieval_query_families": record.retrieval_query_families,
            "retrieval_query_new_segments": record.retrieval_query_new_segments,
            "retrieval_query_result_chunks": record.retrieval_query_result_chunks,
            "case_segments_retrieved": record.case_segments_retrieved,
            "no_new_case_queries": record.no_new_case_queries,
            "reasoning_summary": record.reasoning_summary,
            "raw_model_output": record.raw_model_output,
        }
        for record in records
    ]
    (output_dir / "model_debug_outputs.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in debug_rows),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def _case_query_family_yield(records: list[PredictionRecord]) -> dict[str, dict[str, float]]:
    attempts: Counter[str] = Counter()
    new_segments: Counter[str] = Counter()
    for record in records:
        for family, added in zip(record.retrieval_query_families, record.retrieval_query_new_segments):
            attempts[family] += 1
            new_segments[family] += added
    return {
        family: {
            "attempts": attempts[family],
            "new_segments": new_segments[family],
            "new_segments_per_attempt": new_segments[family] / attempts[family] if attempts[family] else 0.0,
        }
        for family in sorted(attempts)
    }


def _confusion_matrix(records: list[PredictionRecord], gold: dict[str, CaseGold]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        expected = gold[record.case_id].verdict_label or ""
        matrix[expected][record.prediction] += 1
    return {label: dict(row) for label, row in matrix.items()}


def _per_label_accuracy(records: list[PredictionRecord], gold: dict[str, CaseGold]) -> dict[str, float]:
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for record in records:
        expected = gold[record.case_id].verdict_label or ""
        totals[expected] += 1
        correct[expected] += int(record.prediction == expected)
    return {label: correct[label] / totals[label] for label in totals if totals[label]}


def _per_label_prf(records: list[PredictionRecord], gold: dict[str, CaseGold]) -> dict[str, dict[str, float]]:
    labels = sorted(OUTCOME_LABELS)
    result: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for record in records if record.prediction == label and gold[record.case_id].verdict_label == label)
        fp = sum(1 for record in records if record.prediction == label and gold[record.case_id].verdict_label != label)
        fn = sum(1 for record in records if record.prediction != label and gold[record.case_id].verdict_label == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {"precision": precision, "recall": recall, "f1": f1}
    return result


def _law_overlap_approx(records: list[PredictionRecord], gold: dict[str, CaseGold]) -> dict[str, Any]:
    total_gold = total_pred = overlap = 0
    for record in records:
        gold_articles = parse_related_law_articles(gold.get(record.case_id).related_law_provisions if gold.get(record.case_id) else None)
        pred_articles = {str(article.article_no or article.aid) for article in record.law_evidence}
        total_gold += len(gold_articles)
        total_pred += len(pred_articles)
        overlap += len(gold_articles & pred_articles)
    precision = overlap / total_pred if total_pred else 0.0
    recall = overlap / total_gold if total_gold else 0.0
    f1 = 2 * overlap / (total_pred + total_gold) if (total_pred + total_gold) else 0.0
    return {"overlap": overlap, "predicted": total_pred, "gold": total_gold, "precision": precision, "recall": recall, "f1": f1}


def _submission_validation(records: list[PredictionRecord], gold: dict[str, CaseGold]) -> dict[str, Any]:
    case_ids = [record.case_id for record in records]
    counts = Counter(case_ids)
    duplicate_case_ids = sorted(case_id for case_id, count in counts.items() if count > 1)
    expected_case_ids = set(gold) if len(records) == len(gold) else set()
    missing_case_ids = sorted(expected_case_ids - set(case_ids)) if expected_case_ids else []
    invalid_predictions = [record.case_id for record in records if record.prediction not in OUTCOME_LABELS]
    empty_case_evidence = [record.case_id for record in records if not record.case_evidence]
    invalid_law_evidence = [
        record.case_id
        for record in records
        if any(not article.law_id or not article.aid for article in record.law_evidence)
    ]
    return {
        "ok": not (duplicate_case_ids or missing_case_ids or invalid_predictions or invalid_law_evidence),
        "duplicate_case_ids": duplicate_case_ids,
        "missing_case_ids": missing_case_ids,
        "invalid_predictions": invalid_predictions,
        "empty_case_evidence": empty_case_evidence,
        "invalid_law_evidence": invalid_law_evidence,
    }
