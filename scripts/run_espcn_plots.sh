#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH}"

echo "Generating ESPCN plots (tradeoffs + distributions)..."
python espcn/plot_quant_analysis.py \
  --benchmark-json results/espcn_quant_benchmark.json \
  --base-config configs/espcn/base.yaml \
  --checkpoint-dir checkpoints/espcn_run \
  --out-dir results/plots_espcn


