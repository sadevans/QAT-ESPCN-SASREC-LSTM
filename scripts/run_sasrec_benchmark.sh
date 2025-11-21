#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH}"

echo "Running SASRec benchmark (quality + CPU latency)..."
python sasrec/benchmark_quant.py \
  --base-config configs/sasrec/base.yaml \
  --results-out results/sasrec_quant_benchmark.json


