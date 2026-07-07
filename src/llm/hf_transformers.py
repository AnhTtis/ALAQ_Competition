from __future__ import annotations

import json

from .base import LLMClient

VI_QWEN_RAG_SYSTEM = (
    "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. "
    "Hãy luôn trả lời một cách hữu ích nhất có thể."
)

_VI_QWEN_RAG_IDS = {"AITeamVN/Vi-Qwen2-7B-RAG"}


def _is_vi_qwen_rag(model_ref: str) -> bool:
    return any(rid in model_ref for rid in _VI_QWEN_RAG_IDS)


def _build_vi_qwen_prompt(messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
    """Chuyển messages thành prompt format của Vi-Qwen2-7B-RAG.

    Vi-Qwen2-7B-RAG không dùng chat template ổn định như các instruct model khác,
    nên adapter này giữ lại system prompt, payload JSON và context truy xuất trong
    một prompt RAG rõ ràng thay vì bỏ qua hướng dẫn hệ thống.
    """
    system_parts: list[str] = []
    payload_parts: list[str] = []
    context_parts: list[str] = []
    question = ""

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            if content.strip():
                system_parts.append(content.strip())
            continue
        if role != "user":
            continue

        payload_parts.append(content)
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            if not question:
                question = content
            continue

        if not isinstance(data, dict):
            if not question:
                question = content
            continue

        question = str(data.get("case_query") or data.get("query") or question or content)
        for key in ("case_evidence", "law_evidence"):
            for ev in data.get(key) or []:
                if not isinstance(ev, dict):
                    continue
                text = str(ev.get("text") or "").strip()
                if text:
                    context_parts.append(text)

    system_text = "\n\n".join(system_parts) or VI_QWEN_RAG_SYSTEM
    payload_text = "\n\n".join(payload_parts) or "{}"
    context_text = "\n\n".join(context_parts) or "(không có ngữ cảnh)"
    json_instruction = (
        "\n\n### Ràng buộc định dạng :\n"
        "Chỉ trả về đúng một JSON object hợp lệ, không markdown, không giải thích ngoài JSON."
        if json_mode
        else ""
    )
    return (
        f"{VI_QWEN_RAG_SYSTEM}\n\n"
        f"### Hướng dẫn hệ thống :\n{system_text}\n\n"
        f"### Dữ liệu đầu vào :\n{payload_text}\n\n"
        f"### Ngữ cảnh :\n{context_text}\n\n"
        f"### Câu hỏi :\n{question}{json_instruction}\n\n"
        f"### Trả lời :\n"
    )


class HFTransformersClient(LLMClient):
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.settings.configure_hf_cache_env()
        self._vi_qwen_mode = _is_vi_qwen_rag(
            settings.llm_model_path or settings.llm_model_id
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        self._load()
        if self._vi_qwen_mode:
            prompt = _build_vi_qwen_prompt(messages, json_mode=json_mode)
            inputs = self.tokenizer(prompt, return_tensors="pt")
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")

        input_device = self._input_device()
        if input_device is not None:
            inputs = inputs.to(input_device)
        import torch

        temp = temperature if temperature > 0 else self.settings.llm_temperature
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.settings.llm_max_new_tokens,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
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

        # Dọn artifacts trước khi tải (lock files từ lần tải trước bị interrupt)
        removed = self.settings.clean_hf_download_artifacts()
        if removed:
            print(f"[HF] Cleaned {len(removed)} stale artifact(s) before download.", flush=True)

        if self.settings.require_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "GPU/CUDA is required but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or set REQUIRE_GPU=false."
            )

        model_ref = self.settings.llm_model_path or self.settings.llm_model_id
        token = self.settings.read_hf_token() or None
        print(f"[HF] Loading: {model_ref} (token={'set' if token else 'none'})", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_ref, token=token, trust_remote_code=True)

        # Vi-Qwen2-7B-RAG dùng bfloat16; các model khác theo llm_torch_dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.settings.llm_torch_dtype, torch.bfloat16)

        kwargs: dict = {
            "device_map": self.settings.llm_device,
            "trust_remote_code": True,
            "token": token,
            "torch_dtype": torch_dtype,
        }

        # 4-bit quantization khi CUDA available
        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                kwargs.pop("torch_dtype", None)  # BnB config quản dtype
            except Exception:
                pass  # giữ torch_dtype fallback

        self.model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
        self.model.eval()

        # Dọn lock files còn sót sau khi tải xong
        removed_after = self.settings.clean_hf_download_artifacts()
        if removed_after:
            print(f"[HF] Cleaned {len(removed_after)} artifact(s) after download.", flush=True)
        print(f"[HF] Ready: {model_ref}", flush=True)
