#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH}"

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

echo "Training ESPCN FP32..."
python espcn/train.py --config configs/espcn/espcn_fp32.yaml

echo "Training ESPCN LSQ..."
python espcn/train.py --config configs/espcn/espcn_lsq.yaml

echo "Training ESPCN APoT..."
python espcn/train.py --config configs/espcn/espcn_apot.yaml

echo "Training ESPCN QDrop..."
python espcn/train.py --config configs/espcn/espcn_qdrop.yaml

echo "Training ESPCN AdaRound..."
python espcn/train.py --config configs/espcn/espcn_adaround.yaml


