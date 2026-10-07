"""Command-line configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from jevbench.data.constants import DATASET_NAME_DEFAULT


def _add_common(p):
    p.add_argument("--model-name", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--model-dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")

    p.add_argument("--hf-dataset", default=DATASET_NAME_DEFAULT)
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--hf-cache-dir", type=Path, default=None)
    p.add_argument("--processed-data-dir", type=Path, default=Path("jev_bench_processed"))
    p.add_argument("--force-reprocess", action="store_true")
    p.add_argument("--surface-seed", type=int, default=42)
    p.add_argument("--configs", default=None, help="Comma-separated JEV-Bench configs; default loads all.")
    p.add_argument("--use-human-soft-labels", action="store_true", default=False)

    p.add_argument("--quick-run", action="store_true", default=True)
    p.add_argument("--full-run", dest="quick_run", action="store_false")
    p.add_argument("--balanced-train-sampling", action="store_true", default=True)
    p.add_argument("--no-balanced-train-sampling", dest="balanced_train_sampling", action="store_false")
    p.add_argument("--balance-field", default="source_config")
    p.add_argument("--train-samples-per-epoch", type=int, default=None)
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--max-test-examples", type=int, default=None)

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--randomize-candidate-order", action="store_true", default=True)
    p.add_argument("--no-randomize-candidate-order", dest="randomize_candidate_order", action="store_false")

    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-max-batches", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--wandb-project", default="variable-decision-jev-bench")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-wandb", dest="use_wandb", action="store_false", default=True)


def _add_hierarchical(p):
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-label-length", type=int, default=96)
    p.add_argument("--label-layers", type=int, default=2)
    p.add_argument("--label-heads", type=int, default=8)
    p.add_argument("--label-ff-mult", type=int, default=4)
    p.add_argument("--candidate-layers", type=int, default=2)
    p.add_argument("--candidate-heads", type=int, default=8)
    p.add_argument("--candidate-ff-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)


def _add_bev(p):
    p.add_argument("--max-state-tokens", type=int, default=512)
    p.add_argument("--max-choice-tokens", type=int, default=96)
    p.add_argument("--max-packed-tokens", type=int, default=32768)
    p.add_argument("--head-dim", type=int, default=512)
    p.add_argument("--head-layers", type=int, default=2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train hierarchical branched-Qwen or BEV on the same JEV-Bench pipeline."
    )
    sub = parser.add_subparsers(dest="architecture", required=True)

    h = sub.add_parser("hierarchical", help="Shared-trunk branched Qwen + label/candidate Transformer hierarchy")
    _add_common(h)
    _add_hierarchical(h)

    b = sub.add_parser("bev", help="BEV packed masking/position trick + BEV ChoiceHead")
    _add_common(b)
    _add_bev(b)

    args = parser.parse_args(argv)
    if args.epochs is None:
        args.epochs = 1 if args.quick_run else 5
    if args.quick_run:
        if args.train_samples_per_epoch is None:
            args.train_samples_per_epoch = 16_000 if args.architecture == "hierarchical" else 16_000
        args.max_val_examples = args.max_val_examples or 5_000
        args.max_test_examples = args.max_test_examples or 5_000
    if args.output_dir is None:
        args.output_dir = Path(f"checkpoints_jev_{args.architecture}")
    return args
