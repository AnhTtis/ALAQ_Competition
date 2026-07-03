from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data_loader import parse_related_law_articles
from .schema import CaseGold, PredictionRecord


def evaluate(records: list[PredictionRecord], gold: dict[str, CaseGold]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"records": len(records)}
    labeled = [record for record in records if gold.get(record.case_id) and gold[record.case_id].verdict_label]
    if labeled:
        correct = sum(1 for record in labeled if record.prediction == gold[record.case_id].verdict_label)
        metrics["outcome_accuracy"] = correct / len(labeled)
        metrics["outcome_correct"] = correct
        metrics["outcome_total"] = len(labeled)
        metrics["confusion_matrix"] = _confusion_matrix(labeled, gold)
        metrics["per_label_accuracy"] = _per_label_accuracy(labeled, gold)
    metrics["prediction_counts"] = dict(Counter(record.prediction for record in records))
    metrics["api_calls_total"] = sum(record.api_calls for record in records)
    metrics["api_calls_by_case"] = {record.case_id: record.api_calls for record in records}
    metrics["fallback_count"] = sum(record.fallback_used for record in records)
    metrics["law_overlap_approx"] = _law_overlap_approx(records, gold)
    return metrics


def write_outputs(*, output_dir: Path, records: list[PredictionRecord], metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = [record.to_submission() for record in records]
    (output_dir / "submission.json").write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    debug_rows = [
        {
            "case_id": record.case_id,
            "prediction": record.prediction,
            "confidence": record.confidence,
            "fallback_used": record.fallback_used,
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
