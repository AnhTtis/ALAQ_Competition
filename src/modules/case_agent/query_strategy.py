from __future__ import annotations

from ...core.schema import CaseUnderstanding
from ...core.text_utils import compact_text, normalize_text

PRIORITY_SUFFIXES = [
    "phần quyết định tuyên xử của bản án",
    "không chấp nhận yêu cầu khởi kiện bác yêu cầu của nguyên đơn",
    "chấp nhận yêu cầu khởi kiện của nguyên đơn",
    "chấp nhận một phần yêu cầu khởi kiện nghĩa vụ cụ thể",
    "nhận định của tòa án hội đồng xét xử xét thấy có căn cứ",
    "căn cứ pháp lý điều luật áp dụng để giải quyết vụ án",
    "yêu cầu phản tố yêu cầu độc lập của bị đơn",
    "buộc bị đơn thanh toán bồi thường giao trả hoàn trả",
    "án phí chi phí tố tụng nghĩa vụ chịu án phí",
    "kết luận giải quyết vụ án quyền và nghĩa vụ các bên",
    "quan điểm viện kiểm sát đề nghị chấp nhận không chấp nhận",
    "kháng cáo sửa án giữ nguyên án sơ thẩm",
    "lời khai chứng cứ tài liệu hồ sơ vụ án",
    "trách nhiệm dân sự nghĩa vụ chứng minh thiệt hại",
    "hợp đồng chuyển nhượng đặt cọc phạt cọc hoàn trả tiền",
]


def build_case_queries(understanding: CaseUnderstanding, *, max_queries: int) -> list[str]:
    base_parts = [understanding.original_query, understanding.dispute_type, " ".join(understanding.legal_keywords[:8])]
    base = compact_text(" ".join(part for part in base_parts if part))
    candidates = list(understanding.case_search_queries)
    for suffix in PRIORITY_SUFFIXES:
        candidates.append(compact_text(f"{base} {suffix}"))
    for claim in understanding.plaintiff_requests:
        candidates.append(compact_text(f"{claim} tòa án chấp nhận không chấp nhận yêu cầu"))
    for position in understanding.defendant_positions:
        candidates.append(compact_text(f"{position} phản tố ý kiến bị đơn tòa án nhận định"))
    return _dedupe(candidates)[:max_queries]


def _dedupe(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for query in queries:
        text = compact_text(query)
        key = normalize_text(text)
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out
