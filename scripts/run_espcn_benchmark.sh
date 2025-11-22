#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH}"

CHECKPOINT_DIR="./checkpoints/espcn_run"

echo "Running ESPCN benchmark (quality + CPU latency)..."
python espcn/benchmark_quant.py \
  --base-config configs/espcn/base.yaml \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --results-out results/espcn_quant_benchmark.json


