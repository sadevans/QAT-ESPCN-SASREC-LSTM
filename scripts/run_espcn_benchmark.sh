#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH}"

echo "Running ESPCN benchmark (quality + CPU latency)..."
python espcn/benchmark_quant.py \
  --base-config configs/espcn/base.yaml \
  --checkpoint-dir checkpoints/espcn_run \
  --results-out results/espcn_quant_benchmark.json


