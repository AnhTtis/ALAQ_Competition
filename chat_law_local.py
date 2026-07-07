from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "corpus_law_pub.json"
EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit", "q"}


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat thử với model HuggingFace local trên corpus luật ALQAC bằng BM25 context."
    )
    parser.add_argument("--model-id", default=None, help="HuggingFace model id. Mặc định dùng LLM_MODEL_ID/settings.")
    parser.add_argument("--model-path", default=None, help="Đường dẫn model local; nếu có sẽ ưu tiên hơn --model-id.")
    parser.add_argument("--cache-dir", default=None, help="HF cache dir. Mặc định theo settings, hiện là D:/hf_cache.")
    parser.add_argument("--gpu", action="store_true", help="Ưu tiên chạy bằng CUDA với device_map=auto.")
    parser.add_argument("--gpu-id", type=int, default=None, help="CUDA GPU index vật lý; tự động bật --gpu.")
    parser.add_argument("--require-gpu", action="store_true", help="Thoát nếu CUDA không khả dụng.")
    parser.add_argument("--device", default=None, help="Override LLM_DEVICE, ví dụ auto, cuda:0, cpu.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH, help="File corpus luật JSON.")
    parser.add_argument("--top-k", type=int, default=5, help="Số điều luật BM25 đưa vào context.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Giới hạn token sinh ra để tiết kiệm VRAM/RAM.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--max-article-chars", type=int, default=1200, help="Cắt mỗi điều luật trong prompt còn tối đa N ký tự.")
    parser.add_argument("--show-sources", action="store_true", help="In các điều luật BM25 đã đưa vào context.")
    parser.add_argument("--once", default=None, help="Hỏi một câu rồi thoát, tiện để smoke test.")
    return parser.parse_args()


def apply_env_overrides(args: argparse.Namespace) -> None:
    use_gpu = args.gpu or args.gpu_id is not None
    os.environ["LLM_BACKEND"] = "hf_transformers"
    os.environ["ENABLE_DENSE_RETRIEVAL"] = "false"
    os.environ["ENABLE_CROSS_ENCODER_RERANK"] = "false"
    os.environ["LLM_MAX_NEW_TOKENS"] = str(max(args.max_new_tokens, 1))
    os.environ["LLM_TEMPERATURE"] = str(args.temperature)
    os.environ["REQUIRE_GPU"] = "true" if (use_gpu or args.require_gpu) else "false"

    if args.cache_dir:
        os.environ["HF_CACHE_DIR"] = str(args.cache_dir)
    if args.model_id:
        os.environ["LLM_MODEL_ID"] = args.model_id
    if args.model_path:
        os.environ["LLM_MODEL_PATH"] = args.model_path
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.device:
        os.environ["LLM_DEVICE"] = args.device
    elif use_gpu:
        os.environ["LLM_DEVICE"] = "auto"


def compact_text(value: object) -> str:
    return " ".join(str(value or "").split())


def truncate(value: object, limit: int) -> str:
    text = compact_text(value)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def cuda_status() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "torch chưa được cài đặt"
    except (ImportError, OSError) as exc:
        return f"torch import lỗi: {exc}"

    if not torch.cuda.is_available():
        return "CUDA không khả dụng"
    current = torch.cuda.current_device()
    return f"CUDA OK: devices={torch.cuda.device_count()} active={current} name={torch.cuda.get_device_name(current)}"


def print_startup(settings: Any, corpus_path: Path, article_count: int) -> None:
    model_ref = settings.llm_model_path or settings.llm_model_id
    print("=== Local Law Chat ===", flush=True)
    print(f"Corpus: {corpus_path}", flush=True)
    print(f"Law articles: {article_count}", flush=True)
    print(f"Model: {model_ref}", flush=True)
    print(f"HF cache: {settings.hf_cache_dir}", flush=True)
    print(f"Device: {settings.llm_device} | require_gpu={settings.require_gpu}", flush=True)
    print(f"Retrieval: BM25 only (dense=false, rerank=false)", flush=True)
    print(f"CUDA: {cuda_status()}", flush=True)
    print("Model sẽ được tải vào HF cache nếu chưa có sẵn.", flush=True)
    if "7B" in model_ref.upper() and "CUDA OK" not in cuda_status():
        print("Cảnh báo: model 7B chạy CPU có thể rất chậm hoặc thiếu RAM.", flush=True)


def print_sources(hits: list[Any]) -> None:
    print("\nNguồn BM25:", flush=True)
    for index, article in enumerate(hits, start=1):
        preview = truncate(article.text, 220)
        print(
            f"[{index}] law_id={article.law_id} aid={article.aid} "
            f"article_no={article.article_no} score={article.score:.4f}\n    {preview}",
            flush=True,
        )


def build_messages(query: str, hits: list[Any], max_article_chars: int) -> list[dict[str, str]]:
    payload = {
        "query": query,
        "law_evidence": [
            {
                "law_id": article.law_id,
                "aid": article.aid,
                "article_no": article.article_no,
                "score": article.score,
                "text": truncate(article.text, max_article_chars),
            }
            for article in hits
        ],
    }
    return [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def answer_once(query: str, *, retriever: Any, client: Any, args: argparse.Namespace) -> None:
    hits = retriever.search(query, top_k=max(args.top_k, 1))
    if args.show_sources:
        print_sources(hits)
    messages = build_messages(query, hits, max(args.max_article_chars, 1))
    print("\nĐang sinh câu trả lời...", flush=True)
    answer = client.generate(
        messages,
        max_new_tokens=max(args.max_new_tokens, 1),
        temperature=args.temperature,
    )
    print("\nTrả lời:", flush=True)
    print(answer.strip(), flush=True)


def chat_loop(*, retriever: Any, client: Any, args: argparse.Namespace) -> None:
    show_sources = args.show_sources
    print("\nNhập câu hỏi pháp luật. Lệnh: /exit để thoát, /sources bật/tắt nguồn, /help xem trợ giúp.", flush=True)
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát.", flush=True)
            return

        if not query:
            continue
        if query.lower() in EXIT_COMMANDS:
            print("Thoát.", flush=True)
            return
        if query.lower() == "/help":
            print("/sources: bật/tắt in nguồn BM25 | /exit: thoát", flush=True)
            continue
        if query.lower() == "/sources":
            show_sources = not show_sources
            args.show_sources = show_sources
            print(f"show_sources={show_sources}", flush=True)
            continue

        try:
            answer_once(query, retriever=retriever, client=client, args=args)
        except RuntimeError as exc:
            print(f"Lỗi runtime: {exc}", file=sys.stderr, flush=True)
        except KeyboardInterrupt:
            print("\nĐã huỷ lượt sinh. Gõ /exit để thoát.", flush=True)


def main() -> None:
    configure_stdio()
    args = parse_args()
    apply_env_overrides(args)

    from src.core.config import load_settings
    from src.core.data_loader import load_law_articles
    from src.llm.hf_transformers import HFTransformersClient
    from src.modules.law_retrieval.bm25 import BM25LawRetriever

    settings = load_settings()
    corpus_path = args.corpus.resolve()
    articles = load_law_articles(corpus_path)
    retriever = BM25LawRetriever(articles)
    client = HFTransformersClient(settings)

    print_startup(settings, corpus_path, len(articles))
    if args.once:
        answer_once(args.once, retriever=retriever, client=client, args=args)
    else:
        chat_loop(retriever=retriever, client=client, args=args)


if __name__ == "__main__":
    main()
