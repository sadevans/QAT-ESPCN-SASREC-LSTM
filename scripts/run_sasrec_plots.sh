#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)"
export PYTHONPATH=\"$ROOT_DIR:${PYTHONPATH}\"

echo \"Generating SASRec plots (tradeoffs + activations)...\"
python sasrec/plot_quant_analysis.py \\
  --benchmark-json results/sasrec_quant_benchmark.json \\
  --base-config configs/sasrec/base.yaml \\
  --out-dir results/plots_sasrec


