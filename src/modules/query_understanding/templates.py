from __future__ import annotations

import re

from ...core.text_utils import compact_text, normalize_text

ARTICLE_REF_RE = re.compile(r"\b(?:đi[eê]u|dieu)\s*(\d{1,4})\b", re.IGNORECASE)
KEYWORD_GROUPS = {
    "đất đai": ["đất", "quyền sử dụng đất", "thửa", "giấy chứng nhận", "chuyển nhượng"],
    "hợp đồng": ["hợp đồng", "đặt cọc", "chuyển nhượng", "mua bán", "vay", "thuê"],
    "bồi thường": ["bồi thường", "thiệt hại", "tổn thất", "chi phí điều trị"],
    "hôn nhân gia đình": ["ly hôn", "nuôi con", "cấp dưỡng", "tài sản chung"],
    "thừa kế": ["thừa kế", "di sản", "hàng thừa kế", "chia di sản"],
    "lao động": ["lao động", "tiền lương", "sa thải", "bảo hiểm"],
}


def detect_dispute_type(case_query: str) -> str:
    normalized = normalize_text(case_query)
    scores = []
    for label, terms in KEYWORD_GROUPS.items():
        score = sum(1 for term in terms if normalize_text(term) in normalized)
        if score:
            scores.append((score, label))
    if scores:
        return max(scores)[1]
    return "tranh chấp dân sự"


def extract_article_refs(case_query: str) -> list[str]:
    return sorted(set(ARTICLE_REF_RE.findall(case_query)), key=lambda item: int(item))


def extract_legal_keywords(case_query: str) -> list[str]:
    normalized = normalize_text(case_query)
    keywords: list[str] = []
    for terms in KEYWORD_GROUPS.values():
        for term in terms:
            if normalize_text(term) in normalized:
                keywords.append(term)
    for phrase in ["nguyên đơn", "bị đơn", "phản tố", "án phí", "nghĩa vụ", "trách nhiệm", "căn cứ pháp lý"]:
        if normalize_text(phrase) in normalized:
            keywords.append(phrase)
    return _dedupe(keywords)


def initial_case_queries(case_query: str, legal_keywords: list[str] | None = None) -> list[str]:
    q = compact_text(case_query)
    keyword_text = " ".join(legal_keywords or [])
    anchors = [compact_text(f"{q} {keyword_text}"), q]
    queries = []
    for anchor in anchors:
        if not anchor:
            continue
        queries.extend(
            [
                f"{anchor} quyết định tuyên xử phần quyết định bản án",
                f"{anchor} tòa án chấp nhận yêu cầu khởi kiện của nguyên đơn",
                f"{anchor} tòa án không chấp nhận bác yêu cầu khởi kiện",
                f"{anchor} chấp nhận một phần yêu cầu nghĩa vụ cụ thể",
                f"{anchor} nhận định của tòa án hội đồng xét xử có căn cứ",
                f"{anchor} căn cứ pháp lý điều luật áp dụng giải quyết vụ án",
                f"{anchor} bị đơn phản tố yêu cầu độc lập ý kiến bị đơn",
                f"{anchor} buộc thanh toán bồi thường giao trả hoàn trả nghĩa vụ",
                f"{anchor} án phí chi phí tố tụng nghĩa vụ chịu án phí",
                f"{anchor} kết luận giải quyết vụ án quyền và nghĩa vụ các bên",
            ]
        )
    return _dedupe(queries)[:24]


def initial_law_queries(case_query: str, legal_keywords: list[str] | None = None, article_refs: list[str] | None = None) -> list[str]:
    q = compact_text(case_query)
    keyword_text = " ".join(legal_keywords or [])
    article_text = " ".join(f"Điều {item}" for item in article_refs or [])
    return _dedupe(
        [
            compact_text(f"{q} {keyword_text} căn cứ pháp lý điều luật áp dụng"),
            compact_text(f"{q} Bộ luật Dân sự Bộ luật Tố tụng dân sự Luật Đất đai nghĩa vụ trách nhiệm"),
            compact_text(f"{article_text} {q}"),
        ]
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = compact_text(item)
        key = normalize_text(text)
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out
