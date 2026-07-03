from __future__ import annotations

from .base import LLMClient


class LlamaCppClient(LLMClient):
    def __init__(self, settings):
        if not settings.llm_model_path:
            raise ValueError("LLM_MODEL_PATH is required for llamacpp backend")
        from llama_cpp import Llama

        self.settings = settings
        self.llm = Llama(model_path=settings.llm_model_path, n_ctx=8192, verbose=False)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant:"
        output = self.llm(prompt, max_tokens=max_new_tokens or self.settings.llm_max_new_tokens, temperature=temperature)
        return output["choices"][0]["text"]
