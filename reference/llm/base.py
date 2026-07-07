from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError


def build_llm_client(settings: Any) -> LLMClient:
    backend = settings.llm_backend.lower().strip()
    if backend == "hf_transformers":
        from .hf_transformers import HFTransformersClient

        return HFTransformersClient(settings)
    if backend == "llamacpp":
        from .llamacpp import LlamaCppClient

        return LlamaCppClient(settings)
    if backend == "vllm":
        from .vllm_client import VLLMClient

        return VLLMClient(settings)
    if backend == "mock":
        from .mock import MockLLMClient

        return MockLLMClient()
    raise ValueError(f"Unsupported LLM_BACKEND={settings.llm_backend!r}")
