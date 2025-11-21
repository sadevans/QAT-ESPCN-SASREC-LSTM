#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH}"

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

echo "Training SASRec FP32..."
python sasrec/train.py --config configs/sasrec/sasrec_fp32.yaml

echo "Training SASRec LSQ..."
python sasrec/train.py --config configs/sasrec/sasrec_lsq.yaml

echo "Training SASRec APoT..."
python sasrec/train.py --config configs/sasrec/sasrec_apot.yaml

echo "Training SASRec QDrop..."
python sasrec/train.py --config configs/sasrec/sasrec_qdrop.yaml

echo "Training SASRec AdaRound..."
python sasrec/train.py --config configs/sasrec/sasrec_adaround.yaml


