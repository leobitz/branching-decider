#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

conda run --no-capture-output -n py11 python benchmark_hierarchical.py \
  --checkpoint leobitz/decision-master-base-label-encode-0.1-1.7b \
  --processed-data-dir jev_bench_processed \
  --max-test-examples 1000 \
  --batch-size 2 \
  --output-json hierarchical_latency.json

conda run --no-capture-output -n py11 python benchmark_bev.py \
  --checkpoint leobitz/bev-decider-1.7b \
  --processed-data-dir jev_bench_processed \
  --max-test-examples 1000 \
  --batch-size 2 \
  --output-json bev_latency.json

conda run --no-capture-output -n py11 python compare_latency.py hierarchical_latency.json bev_latency.json