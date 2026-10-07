#!/usr/bin/env python3
"""Train either decision architecture on the same JEV-Bench data pipeline.

Examples:
    python train.py hierarchical --full-run --epochs 1 --batch-size 16 --grad-accum-steps 8 --label-layers 2 --candidate-layers 0
    python train.py bev --full-run --epochs 3 --batch-size 1 --grad-accum-steps 16
    python train.py hierarchical --epochs 1 --batch-size 16 --grad-accum-steps 1 --run-name datset-v2-qwen --no-wandb --label-layers 2 --candidate-layers 0 --full-run
    python train.py bev --epochs 1 --batch-size 8 --grad-accum-steps 1 --run-name datset-v2-qwen --no-wandb
"""

from jevbench.cli import parse_args


def main():
    args = parse_args()
    # Lazy import keeps CLI/help lightweight and makes the package organization explicit.
    from jevbench.training.engine import train
    train(args)


if __name__ == "__main__":
    main()

# python train-qwen-jev-bench-hierarchical.py  --full-run --epochs 1 --batch-size 16 --grad-accum-steps 1 --run-name dataset-v2-qwen-branch-cls --eval-every 1000 --set-layers 0 --label-layers 2 --candidate-layers 0
# python train-qwen-jev-bench-hierarchical.py  --epochs 1 --batch-size 8 --grad-accum-steps 1 --run-name datset-v2-qwen --no-wandb --label-layers 2 --candidate-layers 2