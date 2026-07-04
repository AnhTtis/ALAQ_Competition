from __future__ import annotations

import gc
from collections.abc import Callable

from ..core.config import Settings
from ..core.data_loader import load_cases, load_law_articles
from ..core.schema import CaseInput, CaseSegment, LawArticle, PredictionRecord
from ..core.text_utils import compact_text, truncate
from ..llm.base import build_llm_client
from ..modules.case_agent import CaseApiAgent, EvidenceMemory
from ..modules.case_agent.api_client import CaseRetrievalClient
from ..modules.case_agent.query_strategy import build_case_queries, clean_case_query_for_retrieval
from ..modules.law_retrieval import LawRetriever
from ..modules.query_understanding import CaseQueryUnderstanding
from ..modules.reasoning import LegalReasoner, SelfConsistencyReasoner


class ModularRagPipeline:
    def __init__(self, settings: Settings, *, dry_run_cache_only: bool = False, use_api: bool = True):
        self.settings = settings
        self.settings.ensure_runtime_dirs()
        self.cases, self.gold = load_cases(settings.public_test_path)
        self.law_articles = load_law_articles(settings.law_corpus_path)
        self.llm = build_llm_client(settings)
        self.understanding = CaseQueryUnderstanding(self.llm, max_new_tokens=settings.llm_plan_max_new_tokens)
        self.case_client = CaseRetrievalClient(
            api_key=settings.alqac_api_key if use_api else "",
            base_url=settings.alqac_api_base,
            api_mode=settings.alqac_api_mode,
            cache_path=settings.cache_dir / "case_api_cache.jsonl",
            min_seconds_between_requests=settings.min_seconds_between_requests,
            api_rate_limit_safety_seconds=settings.api_rate_limit_safety_seconds,
            retry_after_default_seconds=settings.retry_after_default_seconds,
            max_api_retries_per_query=settings.max_api_retries_per_query,
            dry_run_cache_only=dry_run_cache_only or not use_api,
        )
        self.case_agent = CaseApiAgent(self.case_client, settings)
        self.law_retriever = LawRetriever(self.law_articles, settings)
        self.reasoner = SelfConsistencyReasoner(LegalReasoner(self.llm, settings), settings)

    def run(
        self,
        *,
        limit: int | None = None,
        on_record: Callable[[int, int, PredictionRecord, list[PredictionRecord]], None] | None = None,
        self_consistency_runs: int | None = None,
        self_consistency_temperature: float | None = None,
    ) -> list[PredictionRecord]:
        records: list[PredictionRecord] = []
        selected_cases = self.cases if limit is None else self.cases[:limit]
        total = len(selected_cases)
        for index, case in enumerate(selected_cases, start=1):
            try:
                record = self.run_case(
                    case,
                    self_consistency_runs=self_consistency_runs,
                    self_consistency_temperature=self_consistency_temperature,
                )
                records.append(record)
                if on_record:
                    on_record(index, total, record, records)
            finally:
                _cleanup_after_case()
        return records

    def run_case(
        self,
        case: CaseInput,
        *,
        self_consistency_runs: int | None = None,
        self_consistency_temperature: float | None = None,
    ) -> PredictionRecord:
        understanding = self.understanding.analyze(case.case_id, case.case_query)
        memory = EvidenceMemory(case_id=case.case_id)
        law_articles: list[LawArticle] = []

        initial_query_budget = min(
            self._remaining_case_api_budget(memory),
            max(2, self.settings.case_api_calls_per_round),
        )
        if initial_query_budget > 0:
            initial_case_queries = build_case_queries(understanding, max_queries=initial_query_budget)
            self.case_agent.run_queries(case, memory, initial_case_queries, max_attempts=initial_query_budget)

        for round_id in range(1, self.settings.max_rag_rounds + 1):
            if self._remaining_case_api_budget(memory) <= 0:
                break
            law_queries = self.understanding.generate_round_law_queries(
                case_id=case.case_id,
                case_query=case.case_query,
                understanding=understanding,
                round_id=round_id,
                previous_laws=law_articles,
                case_segments=memory.segments,
                max_queries=max(self.settings.round_law_top_k, 1),
            )
            round_laws = self._retrieve_round_laws(law_queries)
            law_articles = self._dedupe_laws(law_articles + round_laws)
            case_queries = self.understanding.generate_round_case_queries(
                case_id=case.case_id,
                case_query=case.case_query,
                understanding=understanding,
                round_id=round_id,
                law_articles=round_laws or law_articles[: self.settings.round_law_top_k],
                case_segments=memory.segments,
                max_queries=self.settings.case_api_calls_per_round,
            )
            if not case_queries:
                break
            max_attempts = min(self.settings.case_api_calls_per_round, self._remaining_case_api_budget(memory))
            attempted = self.case_agent.run_queries(case, memory, case_queries, max_attempts=max_attempts)
            if attempted == 0:
                break

        final_laws = self._retrieve_final_laws(case, memory.segments)
        law_articles = self._dedupe_laws(law_articles + final_laws)

        record = self.reasoner.predict(
            case=case,
            case_segments=memory.segments,
            law_articles=law_articles[: self.settings.final_evidence_top_k],
            api_calls=memory.api_calls,
            runs=self_consistency_runs,
            temperature=self_consistency_temperature,
        )
        record.retrieval_queries = list(memory.queries)
        record.retrieval_query_families = list(memory.query_families)
        record.retrieval_query_new_segments = list(memory.query_new_segments)
        record.retrieval_query_result_chunks = list(memory.query_result_chunks)
        record.case_segments_retrieved = len(memory.segments)
        record.no_new_case_queries = memory.no_new_count
        return record

    def _retrieve_laws(self, queries: list[str]) -> list[LawArticle]:
        return self.law_retriever.search(queries, top_k=self.settings.final_evidence_top_k)

    def _retrieve_round_laws(self, queries: list[str]) -> list[LawArticle]:
        return self.law_retriever.search(queries, top_k=self.settings.round_law_top_k)

    def _retrieve_final_laws(self, case: CaseInput, case_segments: list[CaseSegment]) -> list[LawArticle]:
        clean_query = clean_case_query_for_retrieval(case.case_query)
        queries = [
            compact_text(f"{clean_query} căn cứ pháp lý điều luật áp dụng"),
            compact_text(f"{clean_query} trách nhiệm nghĩa vụ điều kiện chấp nhận yêu cầu"),
        ]
        for segment in case_segments[: self.settings.case_segments_for_law_query]:
            queries.append(compact_text(f"{truncate(segment.text, 360)} điều luật áp dụng căn cứ pháp lý"))
        return self._retrieve_laws(queries)

    def _dedupe_laws(self, articles: list[LawArticle]) -> list[LawArticle]:
        seen: set[str] = set()
        out: list[LawArticle] = []
        for article in articles:
            key = article.evidence_id
            if key not in seen:
                seen.add(key)
                out.append(article)
        return out

    def _remaining_case_api_budget(self, memory: EvidenceMemory) -> int:
        return max(0, self.settings.max_case_api_calls - memory.api_calls)


RagPipeline = ModularRagPipeline


def _cleanup_after_case() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
