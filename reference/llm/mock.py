from __future__ import annotations

import json

from .base import LLMClient


class MockLLMClient(LLMClient):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        text = "\n".join(message.get("content", "") for message in messages).lower()
        if "case_queries" in text or "missing_facts" in text:
            return json.dumps(
                {
                    "legal_issues": ["tranh chấp dân sự"],
                    "known_facts": [],
                    "missing_facts": ["quyết định tuyên xử", "nhận định của tòa án"],
                    "case_queries": [
                        "quyết định tuyên xử chấp nhận không chấp nhận chấp nhận một phần",
                        "nhận định của tòa án xét thấy có căn cứ không có căn cứ",
                    ],
                    "law_queries": ["căn cứ pháp lý nghĩa vụ trách nhiệm dân sự"],
                    "should_continue": False,
                },
                ensure_ascii=False,
            )
        prediction = "PARTIAL_A_WIN"
        if "không chấp nhận" in text or "bác yêu cầu" in text:
            prediction = "B_WIN"
        elif "chấp nhận yêu cầu" in text:
            prediction = "A_WIN"
        return json.dumps(
            {
                "prediction": prediction,
                "confidence": 0.2,
                "law_evidence": [],
                "case_evidence": [],
                "reasoning_summary": "mock backend dùng cho smoke test, không phải dự đoán cuối.",
            },
            ensure_ascii=False,
        )
