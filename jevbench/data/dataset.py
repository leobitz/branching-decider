"""Hugging Face loading, preprocessing cache, and PyTorch dataset wrappers."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

try:
    from datasets import get_dataset_config_names, load_dataset
except Exception:  # cache-only operation remains available
    get_dataset_config_names = None
    load_dataset = None

from .constants import (
    PROCESSING_VERSION, PROCESSED_SPLITS, NOUL_SURFACE_PAIRS, SCORE_SURFACE_STYLES
)
from .processing import process_jev_row

def _requested_configs(args):
    if args.configs:
        out = [x.strip() for x in str(args.configs).split(",") if x.strip()]
        if not out:
            raise ValueError("--configs was supplied but no config names were parsed")
        return out
    if get_dataset_config_names is None:
        raise ImportError("datasets is required to discover jev-bench configs. Install with: pip install datasets")
    return list(get_dataset_config_names(
        args.hf_dataset,
        revision=args.dataset_revision,
        cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir else None,
    ))


def _expected_manifest(args, configs):
    return {
        "processing_version": PROCESSING_VERSION,
        "hf_dataset": args.hf_dataset,
        "dataset_revision": args.dataset_revision,
        "configs": list(configs),
        "surface_seed": int(args.surface_seed),
    }


def _cache_matches(processed_dir: Path, expected_manifest):
    processed_dir = Path(processed_dir)
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    if not all((processed_dir / f"{s}.jsonl").exists() for s in PROCESSED_SPLITS):
        return False
    try:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for key, value in expected_manifest.items():
        if current.get(key) != value:
            return False
    return True


def prepare_jev_bench_cache(args):
    processed_dir = Path(args.processed_data_dir)

    # If a cache exists and configs were not explicitly supplied, its config list can be used
    # without contacting the Hub just to discover names.
    cached_manifest = None
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.exists():
        try:
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            cached_manifest = None

    if args.configs:
        configs = _requested_configs(args)
    elif cached_manifest and cached_manifest.get("hf_dataset") == args.hf_dataset and cached_manifest.get("dataset_revision") == args.dataset_revision:
        configs = list(cached_manifest.get("configs") or [])
        if not configs:
            configs = _requested_configs(args)
    else:
        configs = _requested_configs(args)

    expected = _expected_manifest(args, configs)
    if not args.force_reprocess and _cache_matches(processed_dir, expected):
        print(f"Using existing processed jev-bench cache: {processed_dir.resolve()}")
        return processed_dir

    if load_dataset is None:
        raise ImportError(
            "The processed cache is missing/stale and Hugging Face datasets is not installed. "
            "Install with: pip install datasets"
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    tmp_paths = {s: processed_dir / f".{s}.jsonl.tmp" for s in PROCESSED_SPLITS}
    handles = {s: tmp_paths[s].open("w", encoding="utf-8") for s in PROCESSED_SPLITS}
    counts = {s: Counter() for s in PROCESSED_SPLITS}
    primitive_counts = {s: Counter() for s in PROCESSED_SPLITS}
    failures = []

    try:
        print(f"Processing {args.hf_dataset} ({len(configs)} configs) into {processed_dir.resolve()}")
        for ci, config_name in enumerate(configs, start=1):
            print(f"[{ci:02d}/{len(configs):02d}] loading config: {config_name}")
            ds = load_dataset(
                args.hf_dataset,
                config_name,
                revision=args.dataset_revision,
                cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir else None,
            )
            for split_name in PROCESSED_SPLITS:
                if split_name not in ds:
                    print(f"  {split_name:10s}: absent")
                    continue
                written = 0
                for row_idx, row in enumerate(ds[split_name]):
                    try:
                        rec = process_jev_row(row, config_name, split_name, args.surface_seed)
                    except Exception as exc:
                        failures.append({
                            "config": config_name,
                            "split": split_name,
                            "row_index": row_idx,
                            "error": str(exc),
                        })
                        if len(failures) <= 20:
                            print(f"    WARNING: skipped {config_name}/{split_name}[{row_idx}]: {exc}")
                        continue
                    handles[split_name].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counts[split_name][config_name] += 1
                    primitive_counts[split_name][rec["primitive"]] += 1
                    written += 1
                print(f"  {split_name:10s}: {written:,} processed")
    finally:
        for h in handles.values():
            h.close()

    for split_name in PROCESSED_SPLITS:
        final = processed_dir / f"{split_name}.jsonl"
        tmp_paths[split_name].replace(final)

    manifest = dict(expected)
    manifest.update({
        "split_counts": {s: dict(counts[s]) for s in PROCESSED_SPLITS},
        "primitive_counts": {s: dict(primitive_counts[s]) for s in PROCESSED_SPLITS},
        "failures": failures,
        "failure_count": len(failures),
        "noul_surface_pairs": [list(x) for x in NOUL_SURFACE_PAIRS],
        "score_surface_styles": list(SCORE_SURFACE_STYLES),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Processed cache complete:")
    for split_name in PROCESSED_SPLITS:
        print(f"  {split_name:10s}: {sum(counts[split_name].values()):,}")
    print(f"  failures  : {len(failures):,}")
    return processed_dir


def _read_jsonl(path: Path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _balanced_cap_rows(rows, max_examples, group_field="source_config", seed=42):
    if max_examples is None or len(rows) <= max_examples:
        return rows
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_field, "unknown"))].append(row)
    names = sorted(groups)
    rng = random.Random(seed)
    for g in names:
        rng.shuffle(groups[g])
    chosen = []
    cursors = {g: 0 for g in names}
    # Round-robin gives every config representation before taking second examples.
    while len(chosen) < max_examples:
        progressed = False
        for g in names:
            i = cursors[g]
            if i < len(groups[g]):
                chosen.append(groups[g][i])
                cursors[g] += 1
                progressed = True
                if len(chosen) >= max_examples:
                    break
        if not progressed:
            break
    rng.shuffle(chosen)
    return chosen


def normalize_jev_record(row, split_name, row_index, use_human_soft_labels=False):
    labels = row.get("labels")
    if not isinstance(labels, list) or len(labels) < 2:
        raise ValueError(f"{split_name}[{row_index}] has invalid labels")
    labels = [str(x) for x in labels]
    target = int(row["target_index"])
    if not 0 <= target < len(labels):
        raise ValueError(f"{split_name}[{row_index}] target_index={target} out of range")

    probs = None
    if use_human_soft_labels:
        probs = row.get("soft_target_probs")
        if probs is not None:
            probs = np.asarray(probs, dtype=np.float64)
            if probs.ndim != 1 or len(probs) != len(labels) or not np.isfinite(probs).all() or (probs < 0).any() or probs.sum() <= 0:
                probs = None
    if probs is None:
        probs = np.zeros(len(labels), dtype=np.float64)
        probs[target] = 1.0
    else:
        probs = probs / probs.sum()

    return {
        "context": str(row.get("context", "")),
        "query": str(row.get("query", "")),
        "labels": labels,
        "target": target,
        "target_probs": probs.astype(np.float32).tolist(),
        "source": str(row.get("source", row.get("source_config", "unknown"))),
        "source_config": str(row.get("source_config", "unknown")),
        "task_family": str(row.get("task_family", row.get("source_config", "unknown"))),
        "primitive": str(row.get("primitive", "unknown")),
        "domain": str(row.get("domain", "unknown")),
        "label_style": str(row.get("label_style", "unknown")),
        "candidate_count": len(labels),
        "split": split_name,
        "original": bool(row.get("original", True)),
        "context_id": row.get("context_id"),
        "decision_id": row.get("decision_id"),
        "is_soft_target": bool(np.count_nonzero(probs > 1e-8) > 1),
        "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
    }


class JevBenchDecisionDataset(Dataset):
    def __init__(self, rows, split_name, max_examples=None, seed=42, use_human_soft_labels=False):
        rows = _balanced_cap_rows(rows, max_examples, group_field="source_config", seed=seed)
        self.rows = []
        bad = []
        for i, row in enumerate(rows):
            try:
                self.rows.append(normalize_jev_record(row, split_name, i, use_human_soft_labels))
            except Exception as exc:
                bad.append((i, str(exc)))
        if bad:
            preview = "\n".join(f"  row {i}: {msg}" for i, msg in bad[:10])
            raise ValueError(f"Failed to normalize {len(bad)}/{len(rows)} processed rows from {split_name}.\n{preview}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def load_jev_bench_splits(args):
    processed_dir = prepare_jev_bench_cache(args)
    raw = {s: _read_jsonl(processed_dir / f"{s}.jsonl") for s in PROCESSED_SPLITS}
    if not getattr(args, "use_reformulated_data", True):
        raw["train"] = [r for r in raw["train"] if r.get("original", True)]
    limits = {
        "train": args.max_train_examples,
        "validation": args.max_val_examples,
        "test": args.max_test_examples,
    }
    datasets = {
        s: JevBenchDecisionDataset(
            raw[s],
            s,
            max_examples=limits[s],
            seed=args.seed + {"train": 1, "validation": 2, "test": 3}[s],
            use_human_soft_labels=args.use_human_soft_labels,
        )
        for s in PROCESSED_SPLITS
    }
    print(f"Using processed dataset directory: {processed_dir.resolve()}")
    for split_name, ds in datasets.items():
        by_primitive = Counter(x["primitive"] for x in ds.rows)
        by_config = Counter(x["source_config"] for x in ds.rows)
        soft = sum(x["is_soft_target"] for x in ds.rows)
        n_orig = sum(x["original"] for x in ds.rows)
        print(
            f"{split_name:10s}: n={len(ds):7d} | original={n_orig:7d} | generated={len(ds) - n_orig:7d} | "
            f"configs={len(by_config):2d} | primitive={dict(by_primitive)} | soft_targets={soft:,}"
        )
    return processed_dir, datasets
