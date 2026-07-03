from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the modular ALQAC 5-module Legal RAG pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases for debugging.")
    parser.add_argument("--no-api", action="store_true", help="Do not call the Case Content API.")
    parser.add_argument("--dry-run-cache-only", action="store_true", help="Use cached API responses only.")
    parser.add_argument("--print-metrics", action="store_true", help="Print metrics JSON after the run.")
    parser.add_argument("--quiet", action="store_true", help="Disable per-case progress logs.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write partial outputs every N completed cases.")
    parser.add_argument("--gpu", action="store_true", help="Run local LLM with Transformers on CUDA using device_map=auto.")
    parser.add_argument("--gpu-id", type=int, default=None, help="Physical CUDA GPU index to use; implies --gpu.")
    parser.add_argument("--clean-cache", action="store_true", help="Remove only safe/rebuildable cache files before running.")
    parser.add_argument("--self-consistency-runs", type=int, default=None, help="Override number of Module E reasoning runs.")
    parser.add_argument("--temperature", type=float, default=None, help="Override self-consistency sampling temperature.")
    parser.add_argument("--no-dense", action="store_true", help="Disable dense BGE-M3 law retrieval for this run.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder law reranking for this run.")
    args = parser.parse_args()

    _configure_runtime(args)

    from src.core.config import load_settings
    from src.core.evaluation import evaluate, write_outputs
    from src.pipeline import RagPipeline, write_trace

    settings = load_settings()
    if args.gpu or args.gpu_id is not None:
        _validate_cuda_or_exit(args.gpu_id)
    if args.clean_cache:
        removed = settings.clean_runtime_cache(include_rebuildable=True)
        if removed and not args.quiet:
            print("cleaned cache:")
            for path in removed:
                print(f"- {path}")
    pipeline = RagPipeline(settings, dry_run_cache_only=args.dry_run_cache_only, use_api=not args.no_api)

    def checkpoint(index, total, record, partial_records):
        if not args.quiet:
            print(
                f"[{index}/{total}] {record.case_id} prediction={record.prediction} "
                f"api_calls={record.api_calls} laws={len(record.law_evidence)} cases={len(record.case_evidence)}",
                flush=True,
            )
        if index % max(args.checkpoint_every, 1) == 0 or index == total:
            partial_metrics = evaluate(partial_records, pipeline.gold)
            write_outputs(output_dir=settings.outputs_dir, records=partial_records, metrics=partial_metrics)
            write_trace(settings.outputs_dir / "retrieval_trace.jsonl", partial_records)
            if not args.quiet:
                print(f"checkpoint -> {settings.outputs_dir / 'submission.json'}", flush=True)

    records = pipeline.run(
        limit=args.limit,
        on_record=checkpoint,
        self_consistency_runs=args.self_consistency_runs,
        self_consistency_temperature=args.temperature,
    )
    metrics = evaluate(records, pipeline.gold)
    write_outputs(output_dir=settings.outputs_dir, records=records, metrics=metrics)
    write_trace(settings.outputs_dir / "retrieval_trace.jsonl", records)

    if args.print_metrics:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print(f"Wrote outputs to {settings.outputs_dir}")
        print(f"Records: {len(records)}")
        if "outcome_accuracy" in metrics:
            print(f"Outcome accuracy: {metrics['outcome_accuracy']:.4f} ({metrics['outcome_correct']}/{metrics['outcome_total']})")


def _configure_runtime(args) -> None:
    use_gpu = args.gpu or args.gpu_id is not None
    if use_gpu:
        if args.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ.setdefault("LLM_BACKEND", "hf_transformers")
        os.environ["LLM_DEVICE"] = "auto"
        os.environ["REQUIRE_GPU"] = "true"
    else:
        os.environ.setdefault("LLM_BACKEND", "mock")
        os.environ["REQUIRE_GPU"] = "false"
        os.environ.setdefault("ENABLE_DENSE_RETRIEVAL", "false")
        os.environ.setdefault("ENABLE_CROSS_ENCODER_RERANK", "false")
    if args.no_dense:
        os.environ["ENABLE_DENSE_RETRIEVAL"] = "false"
    if args.no_rerank:
        os.environ["ENABLE_CROSS_ENCODER_RERANK"] = "false"
    if args.self_consistency_runs is not None:
        os.environ["SELF_CONSISTENCY_RUNS"] = str(args.self_consistency_runs)
    if args.temperature is not None:
        os.environ["SELF_CONSISTENCY_TEMPERATURE"] = str(args.temperature)


def _validate_cuda_or_exit(gpu_id: int | None) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is not installed. Install a CUDA-enabled torch build before using --gpu/--gpu-id.") from exc
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "PyTorch CUDA failed to import, so --gpu/--gpu-id cannot run Qwen.\n"
            f"Import error: {exc}\n"
            "Reinstall a matching CUDA-enabled torch build in this environment."
        ) from exc

    if not torch.cuda.is_available():
        selected = "auto" if gpu_id is None else str(gpu_id)
        raise SystemExit(
            "CUDA is not available to PyTorch, so --gpu/--gpu-id cannot run Qwen.\n"
            f"Selected server GPU: {selected}\n"
            "Check that the server has an NVIDIA driver and CUDA-enabled PyTorch."
        )

    print(
        f"CUDA ready: visible_devices={torch.cuda.device_count()} "
        f"active_device={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
