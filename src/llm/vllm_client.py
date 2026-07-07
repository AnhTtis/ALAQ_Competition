from __future__ import annotations

from .base import LLMClient


def _strip_thinking(text: str) -> str:
    start = text.lower().find("<think>")
    end = text.lower().find("</think>")
    if start != -1 and end != -1 and end >= start:
        cleaned = (text[:start] + text[end + len("</think>"):]).strip()
        if cleaned:
            return cleaned
    return text.strip()


class VLLMClient(LLMClient):
    def __init__(self, settings):
        self.settings = settings
        self.llm = None
        self.tokenizer = None
        self.settings.configure_hf_cache_env()

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        self._load()
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not json_mode,
        )
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens or self.settings.llm_max_new_tokens,
        )
        outputs = self.llm.generate([prompt], params)
        return _strip_thinking(outputs[0].outputs[0].text)

    def _load(self) -> None:
        if self.llm is not None and self.tokenizer is not None:
            return
        from transformers import AutoTokenizer
        from vllm import LLM

        model_ref = self.settings.llm_model_path or self.settings.llm_model_id
        token = self.settings.read_hf_token() or None
        self.tokenizer = AutoTokenizer.from_pretrained(model_ref, token=token, trust_remote_code=True)
        self.llm = LLM(model=model_ref, trust_remote_code=True)
