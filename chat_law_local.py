from __future__ import annotations

import argparse
import json
import os
import subprocess
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
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="CUDA GPU index vật lý. Sau khi set CUDA_VISIBLE_DEVICES, PyTorch sẽ thấy nó là cuda:0.",
    )
    parser.add_argument("--require-gpu", action="store_true", help="Thoát nếu CUDA không khả dụng.")
    parser.add_argument("--device", default=None, help="Override LLM_DEVICE, ví dụ auto, cuda:0, cpu.")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Kiểu dtype khi load model. auto sẽ dùng float16 trên GPU cũ không hỗ trợ bfloat16.",
    )
    parser.add_argument(
        "--cuda-alloc-conf",
        default="expandable_segments:True",
        help="Set PYTORCH_CUDA_ALLOC_CONF trước khi import torch để giảm fragmentation.",
    )
    parser.add_argument(
        "--min-free-vram-gb",
        type=float,
        default=0.0,
        help="Nếu >0, kiểm tra GPU được chọn còn ít nhất N GiB VRAM trống trước khi load model.",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH, help="File corpus luật JSON.")
    parser.add_argument("--top-k", type=int, default=5, help="Số điều luật BM25 đưa vào context.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Giới hạn token sinh ra để tiết kiệm VRAM/RAM.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--max-article-chars", type=int, default=1200, help="Cắt mỗi điều luật trong prompt còn tối đa N ký tự.")
    parser.add_argument("--show-sources", action="store_true", help="In các điều luật BM25 đã đưa vào context.")
    parser.add_argument("--once", default=None, help="Hỏi một câu rồi thoát, tiện để smoke test.")
    parser.add_argument("--list-gpus", action="store_true", help="Liệt kê GPU NVIDIA theo nvidia-smi rồi thoát.")
    return parser.parse_args()


def nvidia_smi_lines() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def nvidia_smi_gpu_ids() -> list[int]:
    ids: list[int] = []
    for line in nvidia_smi_lines():
        prefix = line.split(":", 1)[0].strip()
        if prefix.lower().startswith("gpu "):
            value = prefix.split(None, 1)[1]
            if value.isdigit():
                ids.append(int(value))
    return ids


def print_available_gpus() -> None:
    lines = nvidia_smi_lines()
    if not lines:
        print("Không đọc được danh sách GPU từ nvidia-smi.", flush=True)
        return
    print("GPU NVIDIA khả dụng theo nvidia-smi:", flush=True)
    for line in lines:
        print(f"- {line}", flush=True)


def query_gpu_memory(gpu_id: int) -> tuple[str, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_id),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return "nvidia-smi không khả dụng", None
    if result.returncode != 0:
        return (result.stderr or result.stdout or "nvidia-smi query lỗi").strip(), None
    line = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    free_mib: float | None = None
    if len(parts) >= 5:
        try:
            free_mib = float(parts[4])
        except ValueError:
            free_mib = None
    return line, free_mib


def validate_gpu_id_or_exit(args: argparse.Namespace) -> None:
    if args.gpu_id is None:
        return
    ids = nvidia_smi_gpu_ids()
    if ids and args.gpu_id not in ids:
        available = ", ".join(str(item) for item in ids)
        raise SystemExit(
            f"Không có GPU NVIDIA vật lý số {args.gpu_id}. GPU khả dụng: {available}.\n"
            f"Hãy chạy với --gpu-id {ids[0]} hoặc dùng --list-gpus để kiểm tra."
        )


def validate_device_arg_or_exit(args: argparse.Namespace) -> None:
    if args.gpu_id is None or not args.device:
        return
    normalized = args.device.strip().lower()
    if normalized.startswith("cuda:") and normalized != "cuda:0":
        raise SystemExit(
            f"Bạn đã chọn GPU vật lý {args.gpu_id}, script sẽ set CUDA_VISIBLE_DEVICES={args.gpu_id}.\n"
            "Sau khi mask như vậy, PyTorch chỉ nhìn thấy GPU đó dưới tên logic cuda:0.\n"
            "Vì vậy hãy bỏ --device hoặc dùng --device cuda:0, không dùng --device cuda:3."
        )


def validate_free_vram_or_exit(args: argparse.Namespace) -> None:
    if args.gpu_id is None or args.min_free_vram_gb <= 0:
        return
    status, free_mib = query_gpu_memory(args.gpu_id)
    if free_mib is None:
        print(f"Không kiểm tra được VRAM GPU {args.gpu_id}: {status}", flush=True)
        return
    required_mib = args.min_free_vram_gb * 1024
    if free_mib < required_mib:
        raise SystemExit(
            f"GPU vật lý {args.gpu_id} không đủ VRAM trống.\n"
            f"nvidia-smi: {status}\n"
            f"Cần >= {args.min_free_vram_gb:.2f} GiB trống nhưng chỉ còn {free_mib / 1024:.2f} GiB.\n"
            "Hãy chọn GPU khác, giảm model/context, hoặc tắt process đang chiếm VRAM."
        )


def auto_torch_dtype(args: argparse.Namespace) -> str | None:
    if args.torch_dtype != "auto":
        return args.torch_dtype
    if os.getenv("LLM_TORCH_DTYPE"):
        return None
    if not (args.gpu or args.gpu_id is not None):
        return None
    try:
        import torch
    except Exception:
        return "float16"
    if not torch.cuda.is_available():
        return "float16"
    major, _minor = torch.cuda.get_device_capability(0)
    return "bfloat16" if major >= 8 else "float16"


def apply_env_overrides(args: argparse.Namespace) -> None:
    use_gpu = args.gpu or args.gpu_id is not None
    os.environ["LLM_BACKEND"] = "hf_transformers"
    os.environ["ENABLE_DENSE_RETRIEVAL"] = "false"
    os.environ["ENABLE_CROSS_ENCODER_RERANK"] = "false"
    os.environ["LLM_MAX_NEW_TOKENS"] = str(max(args.max_new_tokens, 1))
    os.environ["LLM_TEMPERATURE"] = str(args.temperature)
    os.environ["REQUIRE_GPU"] = "true" if (use_gpu or args.require_gpu) else "false"
    if args.cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", args.cuda_alloc_conf)

    if args.cache_dir:
        os.environ["HF_CACHE_DIR"] = str(args.cache_dir)
    if args.model_id:
        os.environ["LLM_MODEL_ID"] = args.model_id
    if args.model_path:
        os.environ["LLM_MODEL_PATH"] = args.model_path
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch_dtype = auto_torch_dtype(args)
    if torch_dtype:
        os.environ["LLM_TORCH_DTYPE"] = torch_dtype
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
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    visible_note = f" physical_visible={visible}" if visible else " physical_visible=all"
    try:
        import torch
    except ModuleNotFoundError:
        return f"torch chưa được cài đặt |{visible_note}"
    except (ImportError, OSError) as exc:
        return f"torch import lỗi: {exc} |{visible_note}"

    if not torch.cuda.is_available():
        return f"CUDA không khả dụng |{visible_note}"
    current = torch.cuda.current_device()
    return (
        f"CUDA OK:{visible_note} logical_devices={torch.cuda.device_count()} "
        f"active_logical={current} name={torch.cuda.get_device_name(current)}"
    )


def print_startup(settings: Any, corpus_path: Path, article_count: int) -> None:
    model_ref = settings.llm_model_path or settings.llm_model_id
    print("=== Local Law Chat ===", flush=True)
    print(f"Corpus: {corpus_path}", flush=True)
    print(f"Law articles: {article_count}", flush=True)
    print(f"Model: {model_ref}", flush=True)
    print(f"HF cache: {settings.hf_cache_dir}", flush=True)
    print(
        f"Device: {settings.llm_device} | require_gpu={settings.require_gpu} | torch_dtype={settings.llm_torch_dtype}",
        flush=True,
    )
    print(f"Retrieval: BM25 only (dense=false, rerank=false)", flush=True)
    print(f"CUDA: {cuda_status()}", flush=True)
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if visible.isdigit():
        memory_status, _ = query_gpu_memory(int(visible))
        print(f"Selected physical GPU {visible}: {memory_status}", flush=True)
        print("Ghi chú: trong PyTorch, GPU vật lý này sẽ hiện là cuda:0 do CUDA_VISIBLE_DEVICES mask.", flush=True)
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
    if args.list_gpus:
        print_available_gpus()
        return
    validate_gpu_id_or_exit(args)
    validate_device_arg_or_exit(args)
    validate_free_vram_or_exit(args)
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
