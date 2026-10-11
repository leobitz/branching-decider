# python train.py hierarchical --epochs 1 --batch-size 16 --grad-accum-steps 1 --run-name jev-bench-branching-00 --label-layers 0 --candidate-layers 0 --full-run --output-dir checkpoints_jev_hierarchical-00
# python train.py hierarchical --epochs 1 --batch-size 16 --grad-accum-steps 1 --run-name jev-bench-branching-20 --label-layers 2 --candidate-layers 0 --full-run --output-dir checkpoints_jev_hierarchical-20 --no-wandb
# python train.py bev --epochs 1 --batch-size 16 --grad-accum-steps 1 --run-name set-transformer --full-run --output-dir checkpoints_jev_bev
# python train.py hierarchical --epochs 1 --batch-size 16 --grad-accum-steps 1 --run-name jev-bench-branching-21 --label-layers 2 --candidate-layers 1 --full-run --output-dir checkpoints_jev_hierarchical-21

# python train.py hierarchical --epochs 1 --batch-size 8 --grad-accum-steps 2 --run-name jev-bench-branching-1.7b --label-layers 2 --candidate-layers 0 --full-run --output-dir checkpoints_jev_hierarchical-1.7B --model-name Qwen/Qwen3-1.7B-Base --max-length 1024 --max-label-length 256


python train.py hierarchical --epochs 1 --batch-size 4 --grad-accum-steps 4 --run-name jev-bench-branching-1.7b-reform --label-layers 2 --candidate-layers 0 --full-run --output-dir checkpoints_jev_hierarchical-1.7B-reform --model-name Qwen/Qwen3-1.7B-Base --max-length 1024 --max-label-length 256 --no-wandb --processed-data-dir jev_bench_processed_reformulated

python train.py hierarchical --epochs 2 --batch-size 8 --grad-accum-steps 2 --run-name jev-bench-branching-1.7b-2e --label-layers 2 --candidate-layers 0 --full-run --output-dir checkpoints_jev_hierarchical-1.7B-2e --model-name Qwen/Qwen3-1.7B-Base --max-length 1024 --max-label-length 256