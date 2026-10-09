"""Benchmark hierarchical branching vs BEV on the SAME JEV-Bench test examples.

The script loads trained ``best_val`` checkpoints, rebuilds each architecture with its own
saved configuration, uses the shared processed JEV-Bench test cache, and reports:

* test accuracy (sanity check),
* total GPU model time,
* mean / p50 / p95 / p99 batch model latency,
* milliseconds per example,
* examples / second,
* end-to-end wall time (DataLoader collation + H2D + model),
* peak CUDA allocated memory,
* the actual hierarchical inference backend (Triton or eager fallback).

The hierarchical model is warmed up before timing so one-time Triton JIT compilation is not
included in the latency numbers.

Examples
--------
For isolated evaluation, run these commands sequentially in separate processes:

python benchmark_hierarchical.py --checkpoint checkpoints_jev_hierarchical/best_val
python benchmark_bev.py --checkpoint checkpoints_jev_bev/best_val
python compare_latency.py hierarchical_latency.json bev_latency.json

python benchmark_test_latency.py \
    --hierarchical-checkpoint checkpoints_jev_hierarchical/best_val \
    --bev-checkpoint checkpoints_jev_bev/best_val \
    --processed-data-dir jev_bench_processed \
    --batch-size 1

# Also measure the old eager branching path as an extra baseline:
python benchmark_test_latency.py ... --include-hierarchical-eager

# Fail rather than silently benchmark eager branching if Triton cannot be used:
python benchmark_test_latency.py ... --require-branch-kernel
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import statistics
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from jevbench.data.collators import BEVJevCollator, BranchingVariableChoiceCollator
from jevbench.data.constants import DATASET_NAME_DEFAULT
from jevbench.data.dataset import JevBenchDecisionDataset, _read_jsonl, prepare_jev_bench_cache
from jevbench.models import build_model, forward_batch
from jevbench.training.metrics import move_tensor_batch
from jevbench.utils.params import print_parameter_summary
from jevbench.utils.runtime import resolve_model_dtype, seed_everything


def parse_args(architecture=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.set_defaults(hierarchical_checkpoint=None, bev_checkpoint=None, include_hierarchical_eager=False, eager=False)
    if architecture is None:
        p.add_argument("--hierarchical-checkpoint", default=None)
        p.add_argument("--bev-checkpoint", default=None)
        p.add_argument("--include-hierarchical-eager", action="store_true")
    else:
        p.description = f"Benchmark only {architecture} in this process and save raw results."
        p.add_argument("--checkpoint", f"--{architecture}-checkpoint", dest=f"{architecture}_checkpoint", required=True,
                       help="Local checkpoint directory or Hugging Face repo ID")
        if architecture == "hierarchical":
            p.add_argument("--eager", action="store_true", help="Run eager branching instead of the fast backend")
    p.add_argument("--require-branch-kernel", action="store_true")

    p.add_argument("--processed-data-dir", type=Path, default=None)
    p.add_argument("--max-test-examples", type=int, default=None, help="Default: entire test split")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--warmup-iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    p.add_argument("--use-human-soft-labels", action="store_true", default=False)

    p.add_argument("--verify-hierarchical-kernel", action="store_true", default=True)
    p.add_argument("--no-verify-hierarchical-kernel", dest="verify_hierarchical_kernel", action="store_false")
    p.add_argument("--strict-kernel-check", action="store_true")
    p.add_argument("--kernel-atol", type=float, default=5e-2)
    p.add_argument("--kernel-rtol", type=float, default=5e-2)

    output_name = f"{architecture}_latency.json" if architecture else "latency_comparison.json"
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()
    args.output_json = args.output_json or Path("hierarchical_eager_latency.json" if args.eager else output_name)
    if args.hierarchical_checkpoint is None and args.bev_checkpoint is None:
        p.error("Provide at least one of --hierarchical-checkpoint or --bev-checkpoint")
    if args.batch_size < 1 or args.num_workers < 0 or args.warmup_iters < 1:
        p.error("batch-size and warmup-iters must be positive; num-workers must be nonnegative")
    if args.max_test_examples is not None and args.max_test_examples < 1:
        p.error("max-test-examples must be positive")
    if args.eager and args.require_branch_kernel:
        p.error("--eager cannot be combined with --require-branch-kernel")
    return args


def resolve_checkpoint(path: Path) -> Path:
    path = Path(path)
    if (path / "model.pt").exists():
        return path
    if (path / "best_val" / "model.pt").exists():
        return path / "best_val"
    if not path.exists():
        from huggingface_hub import snapshot_download

        snapshot_path = Path(snapshot_download(repo_id=str(path)))
        if (snapshot_path / "model.pt").exists():
            return snapshot_path
        if (snapshot_path / "best_val" / "model.pt").exists():
            return snapshot_path / "best_val"
    raise FileNotFoundError(f"Could not find model.pt in {path} or {path / 'best_val'}")


def load_checkpoint_config(path: Path, architecture: str) -> dict:
    path = resolve_checkpoint(path)
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint has no config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    saved_arch = config.get("architecture")
    if saved_arch is not None and saved_arch != architecture:
        raise ValueError(f"{path} says architecture={saved_arch!r}, expected {architecture!r}")
    config["architecture"] = architecture
    return config


def config_get(config: dict, key: str, default):
    value = config.get(key, default)
    return default if value is None else value


def shared_data_namespace(config: dict, cli) -> SimpleNamespace:
    processed_dir = cli.processed_data_dir or Path(config_get(config, "processed_data_dir", "jev_bench_processed"))
    return SimpleNamespace(
        processed_data_dir=Path(processed_dir),
        configs=config.get("configs"),
        hf_dataset=config_get(config, "hf_dataset", DATASET_NAME_DEFAULT),
        dataset_revision=config_get(config, "dataset_revision", "main"),
        hf_cache_dir=None,
        force_reprocess=False,
        surface_seed=int(config_get(config, "surface_seed", 42)),
    )


def load_test_dataset(reference_config: dict, cli):
    data_args = shared_data_namespace(reference_config, cli)
    processed_dir = prepare_jev_bench_cache(data_args)
    rows = _read_jsonl(processed_dir / "test.jsonl")
    random.Random(cli.seed).shuffle(rows)
    dataset = JevBenchDecisionDataset(
        rows,
        "test",
        max_examples=cli.max_test_examples,
        seed=cli.seed + 3,
        use_human_soft_labels=cli.use_human_soft_labels,
    )
    return processed_dir, dataset


def make_eval_collator(architecture: str, config: dict, tokenizer, seed: int):
    if architecture == "hierarchical":
        return BranchingVariableChoiceCollator(
            tokenizer=tokenizer,
            max_length=int(config_get(config, "max_length", 512)),
            max_label_length=int(config_get(config, "max_label_length", 96)),
            training=False,
            randomize_order=True,
            seed=seed + 2,
        )
    if architecture == "bev":
        return BEVJevCollator(
            tokenizer=tokenizer,
            max_state_tokens=int(config_get(config, "max_state_tokens", 512)),
            max_choice_tokens=int(config_get(config, "max_choice_tokens", 96)),
            max_packed_tokens=int(config_get(config, "max_packed_tokens", 32768)),
            training=False,
            randomize_order=True,
            seed=seed + 2,
        )
    raise ValueError(architecture)


def model_namespace(config: dict, architecture: str) -> SimpleNamespace:
    # build_model only consumes the architecture-specific keys below.
    if architecture == "hierarchical":
        return SimpleNamespace(
            architecture="hierarchical",
            model_name=config_get(config, "model_name", "Qwen/Qwen3-0.6B-Base"),
            max_label_length=int(config_get(config, "max_label_length", 96)),
            label_layers=int(config_get(config, "label_layers", 2)),
            label_heads=int(config_get(config, "label_heads", 8)),
            label_ff_mult=int(config_get(config, "label_ff_mult", 4)),
            candidate_layers=int(config_get(config, "candidate_layers", 2)),
            candidate_heads=int(config_get(config, "candidate_heads", 8)),
            candidate_ff_mult=int(config_get(config, "candidate_ff_mult", 4)),
            dropout=float(config_get(config, "dropout", 0.10)),
        )
    return SimpleNamespace(
        architecture="bev",
        model_name=config_get(config, "model_name", "Qwen/Qwen3-0.6B-Base"),
        head_dim=int(config_get(config, "head_dim", 512)),
        head_layers=int(config_get(config, "head_layers", 2)),
    )


def load_model(checkpoint: Path, config: dict, architecture: str, device, dtype):
    ns = model_namespace(config, architecture)
    model = build_model(ns, dtype).to(device)
    ckpt = resolve_checkpoint(checkpoint)
    try:
        state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt / "model.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/state mismatch for {architecture}: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    model.eval()
    model.requires_grad_(False)
    return model, ns


def autocast_context(device, dtype):
    if device.type != "cuda" or dtype == torch.float32:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(xs, q):
    if not xs:
        return float("nan")
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def verify_hierarchical_kernel(model, batch, device, amp_dtype, cli):
    if not hasattr(model, "set_inference_kernel"):
        print("Kernel verification skipped: hierarchical model exposes no kernel toggle.")
        return None

    print("\nVerifying hierarchical Triton output against eager branching on one batch...")
    b = move_tensor_batch(batch, device)
    with torch.inference_mode():
        model.set_inference_kernel(True)
        with autocast_context(device, amp_dtype):
            fast = forward_batch(model, b, "hierarchical")["logits"].float()
        cuda_sync(device)
        backend = getattr(model, "inference_backend", "unknown")

        model.set_inference_kernel(False)
        with autocast_context(device, amp_dtype):
            eager = forward_batch(model, b, "hierarchical")["logits"].float()
        cuda_sync(device)
        model.set_inference_kernel(True)

    mask = b["candidate_mask"]
    diff = (fast - eager).abs()[mask]
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    same_argmax = float((fast.argmax(-1) == eager.argmax(-1)).float().mean().item())
    close = bool(torch.allclose(fast[mask], eager[mask], atol=cli.kernel_atol, rtol=cli.kernel_rtol))
    result = {
        "backend": backend,
        "allclose": close,
        "max_abs_logit_diff": max_abs,
        "mean_abs_logit_diff": mean_abs,
        "argmax_agreement": same_argmax,
        "atol": cli.kernel_atol,
        "rtol": cli.kernel_rtol,
    }
    print(
        f"  backend={backend} allclose={close} max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} argmax_agreement={same_argmax:.4f}"
    )
    if cli.strict_kernel_check and not close:
        raise RuntimeError("Hierarchical Triton kernel failed the requested strict numerical check")
    return result


def benchmark_model(
    name: str,
    architecture: str,
    checkpoint: Path,
    config: dict,
    dataset,
    cli,
    device,
    model_dtype,
    force_hierarchical_eager: bool = False,
):
    print("\n" + "=" * 88)
    print(f"Benchmark: {name}")
    print("=" * 88)

    ckpt = resolve_checkpoint(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    collator = make_eval_collator(architecture, config, tokenizer, cli.seed)
    loader = DataLoader(
        dataset,
        batch_size=cli.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cli.num_workers,
        pin_memory=device.type == "cuda",
    )

    model, _ = load_model(checkpoint, config, architecture, device, model_dtype)
    print_parameter_summary(model, architecture)
    if architecture == "hierarchical" and hasattr(model, "set_inference_kernel"):
        model.set_inference_kernel(not force_hierarchical_eager)

    # Pull one batch for kernel compilation / allocator warmup.  The measured pass below starts
    # with a fresh DataLoader and therefore still evaluates the complete test set.
    first_batch = next(iter(loader))
    first_on_device = move_tensor_batch(first_batch, device)
    with torch.inference_mode():
        for _ in range(max(1, cli.warmup_iters)):
            with autocast_context(device, model_dtype):
                _ = forward_batch(model, first_on_device, architecture)
        cuda_sync(device)

    kernel_check = None
    if (
        architecture == "hierarchical"
        and not force_hierarchical_eager
        and cli.verify_hierarchical_kernel
    ):
        kernel_check = verify_hierarchical_kernel(model, first_batch, device, model_dtype, cli)
        # Verification leaves the fast kernel enabled.

    if cli.require_branch_kernel and architecture == "hierarchical" and not force_hierarchical_eager:
        backend = getattr(model, "inference_backend", "unknown")
        if backend != "triton_shared_prefix":
            raise RuntimeError(
                f"--require-branch-kernel was set, but hierarchical backend is {backend!r}. "
                "Install a CUDA-enabled PyTorch/Triton environment and check the warning above."
            )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model_ms = []
    batch_examples = []
    batch_correct = []
    batch_candidates = []
    total_examples = 0
    correct = 0
    candidates_seen = 0

    wall_start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch_wall_start = time.perf_counter()
            b = move_tensor_batch(batch, device)

            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with autocast_context(device, model_dtype):
                    out = forward_batch(model, b, architecture)
                end_event.record()
                end_event.synchronize()
                elapsed_ms = float(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                with autocast_context(device, model_dtype):
                    out = forward_batch(model, b, architecture)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Synchronization above makes this wall timestamp include collation/H2D/model for this batch.
            _ = time.perf_counter() - batch_wall_start
            model_ms.append(elapsed_ms)

            pred = out["logits"].argmax(-1)
            gold = b["target_probs"].argmax(-1)
            batch_hits = int((pred == gold).sum().item())
            correct += batch_hits
            batch_n = int(pred.shape[0])
            total_examples += batch_n
            batch_choices = int(b["candidate_mask"].sum().item())
            candidates_seen += batch_choices
            batch_examples.append(batch_n)
            batch_correct.append(batch_hits)
            batch_candidates.append(batch_choices)

    cuda_sync(device)
    wall_seconds = time.perf_counter() - wall_start

    total_model_ms = float(sum(model_ms))
    result = {
        "name": name,
        "architecture": architecture,
        "checkpoint": str(ckpt),
        "examples": total_examples,
        "batches": len(model_ms),
        "batch_size": cli.batch_size,
        "raw": {
            "batch_model_ms": model_ms,
            "batch_examples": batch_examples,
            "batch_correct": batch_correct,
            "batch_candidates": batch_candidates,
        },
        "accuracy": correct / max(total_examples, 1),
        "avg_candidates_per_example": candidates_seen / max(total_examples, 1),
        "model_total_seconds": total_model_ms / 1000.0,
        "model_ms_per_example": total_model_ms / max(total_examples, 1),
        "model_examples_per_second": total_examples / max(total_model_ms / 1000.0, 1e-12),
        "batch_latency_ms_mean": float(statistics.mean(model_ms)) if model_ms else float("nan"),
        "batch_latency_ms_p50": percentile(model_ms, 50),
        "batch_latency_ms_p95": percentile(model_ms, 95),
        "batch_latency_ms_p99": percentile(model_ms, 99),
        "end_to_end_seconds": wall_seconds,
        "end_to_end_ms_per_example": wall_seconds * 1000.0 / max(total_examples, 1),
        "end_to_end_examples_per_second": total_examples / max(wall_seconds, 1e-12),
        "peak_cuda_allocated_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None
        ),
        "inference_backend": getattr(model, "inference_backend", "standard_qwen_bev"),
        "kernel_check": kernel_check,
    }

    print(
        f"accuracy={result['accuracy']:.4f} | backend={result['inference_backend']} | "
        f"model={result['model_ms_per_example']:.3f} ms/ex "
        f"({result['model_examples_per_second']:.2f} ex/s) | "
        f"e2e={result['end_to_end_ms_per_example']:.3f} ms/ex | "
        f"p50={result['batch_latency_ms_p50']:.3f} ms p95={result['batch_latency_ms_p95']:.3f} ms"
    )

    del model, loader, first_batch, first_on_device
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    return result


def print_comparison(results):
    if not results:
        return
    print("\n" + "=" * 120)
    print("LATENCY COMPARISON")
    print("=" * 120)
    header = (
        f"{'name':30s} {'backend':24s} {'acc':>7s} {'ms/ex':>10s} "
        f"{'ex/s':>10s} {'p50 ms':>10s} {'p95 ms':>10s} {'e2e ms/ex':>12s} {'peak MB':>10s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        peak = "n/a" if r["peak_cuda_allocated_mb"] is None else f"{r['peak_cuda_allocated_mb']:.0f}"
        print(
            f"{r['name'][:30]:30s} {str(r['inference_backend'])[:24]:24s} "
            f"{r['accuracy']:7.4f} {r['model_ms_per_example']:10.3f} "
            f"{r['model_examples_per_second']:10.2f} {r['batch_latency_ms_p50']:10.3f} "
            f"{r['batch_latency_ms_p95']:10.3f} {r['end_to_end_ms_per_example']:12.3f} {peak:>10s}"
        )

    # Pairwise speedup against BEV when both are present.
    by_name = {r["name"]: r for r in results}
    fast = by_name.get("hierarchical_triton")
    bev = by_name.get("bev")
    if fast is not None and bev is not None:
        model_speedup = bev["model_ms_per_example"] / max(fast["model_ms_per_example"], 1e-12)
        e2e_speedup = bev["end_to_end_ms_per_example"] / max(fast["end_to_end_ms_per_example"], 1e-12)
        print(
            f"\nHierarchical Triton vs BEV: {model_speedup:.3f}x model-time speed ratio "
            f"and {e2e_speedup:.3f}x end-to-end speed ratio "
            "(>1 means hierarchical is faster)."
        )

    eager = by_name.get("hierarchical_eager")
    if fast is not None and eager is not None:
        kernel_speedup = eager["model_ms_per_example"] / max(fast["model_ms_per_example"], 1e-12)
        print(f"Triton branch kernel vs eager branching: {kernel_speedup:.3f}x model-time speedup.")


def main(architecture=None):
    cli = parse_args(architecture)
    seed_everything(cli.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    else:
        warnings.warn("CUDA is unavailable; the Triton branch kernel cannot run and latency is CPU-only.")

    if cli.require_branch_kernel:
        os.environ["JEV_REQUIRE_BRANCH_TRITON"] = "1"

    configs: Dict[str, dict] = {}
    if cli.hierarchical_checkpoint is not None:
        configs["hierarchical"] = load_checkpoint_config(cli.hierarchical_checkpoint, "hierarchical")
    if cli.bev_checkpoint is not None:
        configs["bev"] = load_checkpoint_config(cli.bev_checkpoint, "bev")

    # Ensure the two checkpoints really refer to the same processed benchmark definition.
    if len(configs) == 2:
        for key in ("hf_dataset", "dataset_revision", "surface_seed"):
            hv = config_get(configs["hierarchical"], key, None)
            bv = config_get(configs["bev"], key, None)
            if hv != bv:
                warnings.warn(f"Checkpoint data setting differs for {key}: hierarchical={hv!r}, bev={bv!r}")

    reference_config = configs.get("hierarchical") or configs["bev"]
    processed_dir, test_dataset = load_test_dataset(reference_config, cli)
    if not len(test_dataset):
        raise ValueError("The selected test dataset is empty")
    print(f"Processed data: {processed_dir.resolve()}")
    print(f"Test examples: {len(test_dataset):,}")
    if len(test_dataset):
        avg_candidates = sum(x["candidate_count"] for x in test_dataset.rows) / len(test_dataset)
        print(f"Average candidates/example: {avg_candidates:.2f}")

    dtype_name = cli.dtype
    if dtype_name == "auto":
        # Use the same automatic policy as training rather than trusting a stale saved device choice.
        dtype_name = "auto"
    model_dtype = resolve_model_dtype(device, dtype_name)
    print("model dtype:", model_dtype)

    metadata = {
        "dataset_sha256": hashlib.sha256(
            json.dumps(test_dataset.rows, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest(),
        "processed_data_dir": str(processed_dir.resolve()),
        "seed": cli.seed,
        "batch_size": cli.batch_size,
        "num_workers": cli.num_workers,
        "warmup_iters": cli.warmup_iters,
        "use_human_soft_labels": cli.use_human_soft_labels,
        "dtype": str(model_dtype),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }

    results = []
    if cli.hierarchical_checkpoint is not None:
        results.append(
            benchmark_model(
                "hierarchical_eager" if cli.eager else "hierarchical_triton",
                "hierarchical",
                cli.hierarchical_checkpoint,
                configs["hierarchical"],
                test_dataset,
                cli,
                device,
                model_dtype,
                force_hierarchical_eager=cli.eager,
            )
        )
        if cli.include_hierarchical_eager:
            results.append(
                benchmark_model(
                    "hierarchical_eager",
                    "hierarchical",
                    cli.hierarchical_checkpoint,
                    configs["hierarchical"],
                    test_dataset,
                    cli,
                    device,
                    model_dtype,
                    force_hierarchical_eager=True,
                )
            )

    if cli.bev_checkpoint is not None:
        results.append(
            benchmark_model(
                "bev",
                "bev",
                cli.bev_checkpoint,
                configs["bev"],
                test_dataset,
                cli,
                device,
                model_dtype,
            )
        )

    cli.output_json.parent.mkdir(parents=True, exist_ok=True)
    cli.output_json.write_text(
        json.dumps({"schema_version": 1, "metadata": metadata, "results": results}, indent=2),
        encoding="utf-8",
    )
    if architecture is None:
        print_comparison(results)
    print(f"\nSaved JSON report: {cli.output_json.resolve()}")


if __name__ == "__main__":
    main()

# sample usage: python benchmark_test_latency.py --hierarchical-checkpoint checkpoints_jev_hierarchical  --processed-data-dir jev_bench_processed --max-test-examples 1000 --batch-size 4 --bev-checkpoint checkpoints_jev_bev/best_val --include-hierarchical-eager