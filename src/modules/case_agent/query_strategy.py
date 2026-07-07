from __future__ import annotations

import re

from ...core.schema import CaseUnderstanding
from ...core.text_utils import compact_text, normalize_text

PRIORITY_QUERIES = [
    "quyết định tuyên xử chấp nhận không chấp nhận chấp nhận một phần nghĩa vụ án phí",
    "nhận định của tòa án xét thấy hội đồng xét xử lập luận có căn cứ không có căn cứ",
    "yêu cầu khởi kiện yêu cầu phản tố nội dung tranh chấp",
    "quyết định tuyên xử chấp nhận không chấp nhận chấp nhận một phần",
    "phần quyết định tuyên xử của bản án",
    "nhận định của tòa án hội đồng xét xử xét thấy có căn cứ",
    "không chấp nhận yêu cầu khởi kiện bác yêu cầu của nguyên đơn",
    "chấp nhận một phần yêu cầu khởi kiện nghĩa vụ cụ thể",
    "chấp nhận yêu cầu khởi kiện của nguyên đơn",
    "bị đơn ý kiến phản đối không chấp nhận",
    "quan hệ pháp luật căn cứ pháp lý",
    "các bên liên quan nguyên đơn bị đơn",
]

META_TAIL_RE = re.compile(
    r"\b(?:agent\s+du\s+doan|hay\s+du\s+doan|theo\s+ban|can\s+xac\s+dinh|tra\s+loi|nguyen\s+don\s+thang|bi\s+don\s+thang|ai\s+duoc\s+chap\s+nhan|a_win|b_win|partial_a_win|partial_b_win)\b.*$",
    re.IGNORECASE,
)


def clean_case_query_for_retrieval(text: str) -> str:
    cleaned = compact_text(text)
    normalized = normalize_text(cleaned)
    match = META_TAIL_RE.search(normalized)
    if match:
        normalized_prefix = normalized[: match.start()].strip(" .;:?!")
        original_words = cleaned.split()
        normalized_words = normalize_text(" ".join(original_words)).split()
        keep = len(normalized_prefix.split())
        cleaned = " ".join(original_words[:keep]) if keep and keep <= len(normalized_words) else cleaned
    return _trim_query(cleaned, max_words=42)


def classify_case_query(query: str) -> str:
    normalized = normalize_text(query)
    if "nhan dinh" in normalized or "xet thay" in normalized:
        return "reasoning"
    if "quyet dinh" in normalized or "tuyen xu" in normalized:
        return "decision"
    if "yeu cau khoi kien" in normalized or "phan to" in normalized:
        return "claim"
    if "buoc" in normalized or "thanh toan" in normalized or "giao tra" in normalized or "boi thuong" in normalized:
        return "obligation"
    if "can cu phap ly" in normalized or "dieu" in normalized:
        return "legal_basis"
    return "general"


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
    clean_base = clean_case_query_for_retrieval(understanding.original_query)
    dispute = compact_text(understanding.dispute_type)
    keyword_text = compact_text(" ".join(understanding.legal_keywords[:5]))
    main_claim = _trim_query(" ".join(understanding.plaintiff_requests[:2]), max_words=28)
    main_position = _trim_query(" ".join(understanding.defendant_positions[:2]), max_words=28)
    anchor = compact_text(" ".join(part for part in (dispute, keyword_text, clean_base) if part))
    anchor = _trim_query(anchor, max_words=48)

    canonical_base = clean_base or _trim_query(compact_text(understanding.original_query), max_words=42)
    candidates = [
        compact_text(f"{canonical_base} {PRIORITY_QUERIES[0]}"),
        compact_text(f"{canonical_base} {PRIORITY_QUERIES[1]}"),
        compact_text(f"{canonical_base} {PRIORITY_QUERIES[2]}"),
        compact_text(f"{canonical_base} {PRIORITY_QUERIES[9]}"),
        compact_text(f"{canonical_base} {PRIORITY_QUERIES[10]}"),
        compact_text(f"{canonical_base} {PRIORITY_QUERIES[11]}"),
        compact_text(f"{anchor} phần quyết định tuyên xử chấp nhận không chấp nhận một phần"),
        compact_text(f"{anchor} nhận định của tòa án xét thấy có căn cứ không có căn cứ"),
        compact_text(f"{main_claim or anchor} yêu cầu khởi kiện chấp nhận không chấp nhận"),
        compact_text(f"{anchor} buộc thanh toán giao trả hoàn trả bồi thường nghĩa vụ"),
        compact_text(f"{anchor} căn cứ pháp lý điều luật áp dụng"),
    ]
    if main_position:
        candidates.append(compact_text(f"{main_position} ý kiến bị đơn phản tố nhận định của tòa"))

    for query in PRIORITY_QUERIES[2:6]:
        candidates.append(compact_text(f"{canonical_base} {query}"))
    candidates.extend(PRIORITY_QUERIES[:4])
    candidates.extend(understanding.case_search_queries[:6])
    for suffix in PRIORITY_SUFFIXES[:6]:
        candidates.append(compact_text(f"{anchor} {suffix}"))
    return _dedupe(candidates)[:max_queries]


def _trim_query(text: str, *, max_words: int) -> str:
    words = compact_text(text).split()
    return " ".join(words[:max_words])


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
