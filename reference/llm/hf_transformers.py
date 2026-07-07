from __future__ import annotations

from .base import LLMClient


class HFTransformersClient(LLMClient):
    def __init__(self, settings):
        self.settings = settings
        self.model = None
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
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_device = self._input_device()
        if input_device is not None:
            inputs = inputs.to(input_device)
        import torch

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.settings.llm_max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _input_device(self):
        import torch

        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for device in device_map.values():
                if isinstance(device, int):
                    return torch.device(f"cuda:{device}")
                if isinstance(device, str) and device not in {"cpu", "disk"}:
                    return torch.device(device)
            return torch.device("cpu")
        return getattr(self.model, "device", None)

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.settings.require_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "GPU/CUDA is required for --gpu Qwen mode, but torch.cuda.is_available() is false. "
                "Install a CUDA-enabled PyTorch build on this machine or run without --gpu."
            )

        model_ref = self.settings.llm_model_path or self.settings.llm_model_id
        token = self.settings.read_hf_token() or None
        self.tokenizer = AutoTokenizer.from_pretrained(model_ref, token=token, trust_remote_code=True)
        kwargs = {"device_map": self.settings.llm_device, "trust_remote_code": True, "token": token}
        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:
                kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
        self.model.eval()
