"""Compare saved benchmark reports without importing models or touching CUDA."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


COMPARISON_KEYS = (
    "dataset_sha256", "seed", "batch_size", "num_workers", "warmup_iters",
    "use_human_soft_labels", "dtype", "device", "gpu", "torch_version", "cuda_version",
)


def summarize(result):
    raw = result["raw"]
    timings = raw["batch_model_ms"]
    sizes = raw["batch_examples"]
    hits = raw["batch_correct"]
    candidates = raw["batch_candidates"]
    if not timings or not (len(timings) == len(sizes) == len(hits) == len(candidates)):
        raise ValueError("Raw batch arrays must be nonempty and have equal lengths")
    if any(not math.isfinite(value) or value < 0 for value in timings):
        raise ValueError("Raw timings must be finite and nonnegative")
    for size, correct, choices in zip(sizes, hits, candidates):
        if any(type(value) is not int for value in (size, correct, choices)):
            raise ValueError("Raw counts must be integers")
        if size < 1 or not 0 <= correct <= size or choices < size:
            raise ValueError("Invalid raw example, correct, or candidate count")
    wall_seconds = result["end_to_end_seconds"]
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("End-to-end seconds must be finite and positive")
    examples = sum(sizes)
    total_ms = sum(timings)
    percentiles = statistics.quantiles(timings, n=100, method="inclusive") if len(timings) > 1 else timings * 99
    return {
        **result,
        "examples": examples,
        "batches": len(timings),
        "accuracy": sum(hits) / examples,
        "avg_candidates_per_example": sum(candidates) / examples,
        "model_total_seconds": total_ms / 1000,
        "model_ms_per_example": total_ms / examples,
        "model_examples_per_second": examples / max(total_ms / 1000, 1e-12),
        "batch_latency_ms_mean": statistics.mean(timings),
        "batch_latency_ms_p50": percentiles[49],
        "batch_latency_ms_p95": percentiles[94],
        "batch_latency_ms_p99": percentiles[98],
        "end_to_end_ms_per_example": wall_seconds * 1000 / examples,
        "end_to_end_examples_per_second": examples / wall_seconds,
    }


def load_reports(paths, allow_mismatch=False):
    results = []
    reference = None
    mismatches = []
    names = set()
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema_version") != 1:
            raise ValueError(f"{path}: expected a raw report from the separate benchmark scripts")
        metadata = report["metadata"]
        if any(key not in metadata for key in COMPARISON_KEYS):
            raise ValueError(f"{path}: missing benchmark metadata")
        if reference is None:
            reference = metadata
        for key in COMPARISON_KEYS:
            if metadata[key] != reference[key]:
                mismatches.append(f"{path}: {key}={metadata[key]!r}, first report={reference[key]!r}")
        if not report["results"]:
            raise ValueError(f"{path}: no benchmark results")
        for result in report["results"]:
            if result["name"] in names:
                raise ValueError(f"Duplicate benchmark name: {result['name']}")
            names.add(result["name"])
            results.append(summarize(result))
    if mismatches and not allow_mismatch:
        raise ValueError("Benchmark settings differ; use --allow-mismatch to compare anyway:\n" + "\n".join(mismatches))
    if len({result["examples"] for result in results}) > 1:
        mismatches.append("Reports contain different example counts")
        if not allow_mismatch:
            raise ValueError(mismatches[-1])
    return results, mismatches


def print_comparison(results):
    print("LATENCY COMPARISON")
    print(f"{'name':24s} {'backend':24s} {'acc':>7s} {'ms/ex':>10s} {'ex/s':>10s} "
          f"{'p50 ms':>10s} {'p95 ms':>10s} {'p99 ms':>10s} {'e2e ms/ex':>12s} {'peak MB':>10s}")
    for result in results:
        peak = result["peak_cuda_allocated_mb"]
        peak_text = "n/a" if peak is None else f"{peak:.0f}"
        print(f"{result['name']:24s} {result['inference_backend']:24s} {result['accuracy']:7.4f} "
              f"{result['model_ms_per_example']:10.3f} {result['model_examples_per_second']:10.2f} "
              f"{result['batch_latency_ms_p50']:10.3f} {result['batch_latency_ms_p95']:10.3f} "
              f"{result['batch_latency_ms_p99']:10.3f} {result['end_to_end_ms_per_example']:12.3f} {peak_text:>10s}")
    bev = next((result for result in results if result["architecture"] == "bev"), None)
    if bev is not None:
        for result in results:
            if result["architecture"] == "hierarchical":
                model_ratio = bev["model_ms_per_example"] / max(result["model_ms_per_example"], 1e-12)
                wall_ratio = bev["end_to_end_ms_per_example"] / max(result["end_to_end_ms_per_example"], 1e-12)
                print(f"\n{result['name']} ({result['inference_backend']}) vs BEV: "
                      f"{model_ratio:.3f}x model-time, {wall_ratio:.3f}x end-to-end "
                      "(>1 means hierarchical is faster).")
    by_name = {result["name"]: result for result in results}
    fast = by_name.get("hierarchical_triton")
    eager = by_name.get("hierarchical_eager")
    if fast is not None and eager is not None:
        ratio = eager["model_ms_per_example"] / max(fast["model_ms_per_example"], 1e-12)
        print(f"\nFast backend ({fast['inference_backend']}) vs eager: {ratio:.3f}x model-time speedup.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+", help="JSON files saved by the benchmark scripts")
    parser.add_argument("--output-json", type=Path, default=Path("latency_comparison.json"))
    parser.add_argument("--allow-mismatch", action="store_true")
    args = parser.parse_args()
    if len(args.reports) < 2:
        parser.error("Provide at least two reports to compare")
    if args.output_json.resolve() in {path.resolve() for path in args.reports}:
        parser.error("The comparison output must not overwrite an input report")
    try:
        results, mismatches = load_reports(args.reports, args.allow_mismatch)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    for mismatch in mismatches:
        print(f"WARNING: {mismatch}")
    print_comparison(results)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({
        "sources": [str(path) for path in args.reports],
        "mismatches": mismatches,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved comparison: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()