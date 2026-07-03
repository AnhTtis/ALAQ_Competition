from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKEN_DIR = PROJECT_ROOT / "token"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip().lstrip("﻿"), value.strip().strip('"').strip("'"))


_load_dotenv(TOKEN_DIR / ".env")
_load_dotenv(PROJECT_ROOT / ".env")


def _read_secret_file(path: Path) -> str:
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    return ""


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    token_dir: Path = TOKEN_DIR
    data_dir: Path = PROJECT_ROOT / "data"
    runs_dir: Path = PROJECT_ROOT / "runs"
    cache_dir: Path = PROJECT_ROOT / "runs" / "cache"
    outputs_dir: Path = PROJECT_ROOT / "runs" / "outputs"
    hf_cache_dir: Path = PROJECT_ROOT / "hf_cache"
    public_test_path: Path = PROJECT_ROOT / "data" / "ALQAC2026_public_test.json"
    law_corpus_path: Path = PROJECT_ROOT / "data" / "corpus_law_pub.json"

    alqac_api_key: str = os.getenv("ALQAC_API_KEY", _read_secret_file(TOKEN_DIR / "alqac.txt"))
    alqac_api_base: str = os.getenv("ALQAC_API_BASE", "https://alqac-api.ngrok.pro")
    alqac_api_mode: str = os.getenv("ALQAC_API_MODE", "x_api_key_retrieve")
    min_seconds_between_requests: float = _get_float("MIN_SECONDS_BETWEEN_REQUESTS", 5.0)
    api_rate_limit_safety_seconds: float = _get_float("API_RATE_LIMIT_SAFETY_SECONDS", 0.5)
    retry_after_default_seconds: float = _get_float("RETRY_AFTER_DEFAULT_SECONDS", 60.0)
    max_api_retries_per_query: int = _get_int("MAX_API_RETRIES_PER_QUERY", 1)

    llm_backend: str = os.getenv("LLM_BACKEND", "mock")
    llm_model_id: str = os.getenv("LLM_MODEL_ID", "Qwen/Qwen3.5-9B")
    llm_model_path: str = os.getenv("LLM_MODEL_PATH", "")
    llm_device: str = os.getenv("LLM_DEVICE", "auto")
    require_gpu: bool = _get_bool("REQUIRE_GPU", False)
    llm_plan_max_new_tokens: int = _get_int("LLM_PLAN_MAX_NEW_TOKENS", 1024)
    llm_max_new_tokens: int = _get_int("LLM_MAX_NEW_TOKENS", 2048)
    llm_temperature: float = _get_float("LLM_TEMPERATURE", 0.0)

    hf_token_file: str = os.getenv("HF_TOKEN_FILE", "hf.txt")
    embedding_model_id: str = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3")
    enable_dense_retrieval: bool = _get_bool("ENABLE_DENSE_RETRIEVAL", True)
    cross_encoder_model_id: str = os.getenv("CROSS_ENCODER_MODEL_ID", "BAAI/bge-reranker-v2-m3")
    enable_cross_encoder_rerank: bool = _get_bool("ENABLE_CROSS_ENCODER_RERANK", True)

    min_case_api_calls: int = _get_int("MIN_CASE_API_CALLS", 5)
    max_case_api_calls: int = _get_int("MAX_CASE_API_CALLS", 16)
    case_api_calls_per_round: int = _get_int("CASE_API_CALLS_PER_ROUND", 4)
    max_no_new_segment_queries: int = _get_int("MAX_NO_NEW_SEGMENT_QUERIES", 4)
    max_rag_rounds: int = _get_int("MAX_RAG_ROUNDS", 4)
    round_law_top_k: int = _get_int("ROUND_LAW_TOP_K", 3)
    law_top_k: int = _get_int("LAW_TOP_K", 32)
    law_bm25_candidates: int = _get_int("LAW_BM25_CANDIDATES", 80)
    law_dense_candidates: int = _get_int("LAW_DENSE_CANDIDATES", 80)
    law_rerank_top_k: int = _get_int("LAW_RERANK_TOP_K", 32)
    final_evidence_top_k: int = _get_int("FINAL_EVIDENCE_TOP_K", 16)
    case_evidence_for_prompt: int = _get_int("CASE_EVIDENCE_FOR_PROMPT", 24)
    law_evidence_for_prompt: int = _get_int("LAW_EVIDENCE_FOR_PROMPT", 24)
    case_segments_for_law_query: int = _get_int("CASE_SEGMENTS_FOR_LAW_QUERY", 10)
    max_case_text_chars: int = _get_int("MAX_CASE_TEXT_CHARS", 2400)
    max_law_text_chars: int = _get_int("MAX_LAW_TEXT_CHARS", 1300)
    enable_decision_rule_override: bool = _get_bool("ENABLE_DECISION_RULE_OVERRIDE", False)
    enable_self_consistency: bool = _get_bool("ENABLE_SELF_CONSISTENCY", True)
    self_consistency_runs: int = _get_int("SELF_CONSISTENCY_RUNS", 3)
    self_consistency_temperature: float = _get_float("SELF_CONSISTENCY_TEMPERATURE", 0.4)

    def ensure_runtime_dirs(self) -> None:
        for path in (
            self.token_dir,
            self.runs_dir,
            self.cache_dir,
            self.outputs_dir,
            self.hf_cache_dir,
            self.hf_cache_dir / "hub",
            self.hf_cache_dir / "transformers",
            self.hf_cache_dir / "datasets",
            self.hf_cache_dir / "sentence_transformers",
            self.hf_cache_dir / "torch",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def configure_hf_cache_env(self) -> None:
        os.environ["HF_HOME"] = str(self.hf_cache_dir)
        os.environ["HF_HUB_CACHE"] = str(self.hf_cache_dir / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(self.hf_cache_dir / "transformers")
        os.environ["HF_DATASETS_CACHE"] = str(self.hf_cache_dir / "datasets")
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(self.hf_cache_dir / "sentence_transformers")
        os.environ["TORCH_HOME"] = str(self.hf_cache_dir / "torch")

    def read_hf_token(self) -> str:
        candidates = [
            self.token_dir / self.hf_token_file,
            self.token_dir / "hf.txt",
            self.project_root / self.hf_token_file,
            self.project_root / "hf.txt",
        ]
        for path in dict.fromkeys(candidates):
            if path.exists():
                token = path.read_text(encoding="utf-8").strip()
                if token:
                    return token
        return ""

    def clean_runtime_cache(self, *, include_rebuildable: bool = False) -> list[Path]:
        removed: list[Path] = []
        for directory in (self.cache_dir, self.outputs_dir):
            if not directory.exists():
                continue
            for path in directory.glob("*.tmp"):
                if path.is_file():
                    path.unlink()
                    removed.append(path)
        if include_rebuildable:
            for filename in ("law_embeddings.npz", "bge_law_embeddings.npz"):
                embeddings_cache = self.cache_dir / filename
                if embeddings_cache.exists():
                    embeddings_cache.unlink()
                    removed.append(embeddings_cache)
        return removed

    def disk_free_gb(self, path: Path | None = None) -> float:
        usage = shutil.disk_usage(path or self.project_root)
        return usage.free / (1024 ** 3)


def load_settings() -> Settings:
    settings = Settings()
    settings.ensure_runtime_dirs()
    settings.configure_hf_cache_env()
    return settings
