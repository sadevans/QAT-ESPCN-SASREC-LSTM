#!/usr/bin/env python3
"""
Benchmark exported ONNX models (FP32 + INT8) on CPU:

* measure file size (MB)
* evaluate PSNR on the validation Set14 split
* measure latency / throughput on CPU
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from espcn.data.dataloaders import get_val_loaders
from espcn.train import calculate_psnr
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX ESPCN models on CPU.")
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/espcn/base.yaml",
        help="Base ESPCN config for dataset paths (DIV2K / Set14).",
    )
    parser.add_argument(
        "--onnx-fp32",
        type=str,
        required=True,
        help="Path to the FP32 ONNX model.",
    )
    parser.add_argument(
        "--onnx-int8",
        type=str,
        required=True,
        help="Path to the INT8 ONNX model.",
    )
    parser.add_argument(
        "--results-out",
        type=str,
        default="results/espcn_onnx_benchmark.json",
        help="Where to store the benchmark summary.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations before measuring latency.",
    )
    return parser.parse_args()


def create_session(path: Path) -> ort.InferenceSession:
    """
    Create ONNX Runtime inference session for CPU execution.
    
    Args:
        path: Path to ONNX model file.
        
    Returns:
        ONNX Runtime inference session configured for CPU.
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def eval_psnr(session: ort.InferenceSession, loader: DataLoader) -> float:
    """
    Evaluate PSNR metric using ONNX model inference.
    
    Args:
        session: ONNX Runtime inference session.
        loader: DataLoader with validation samples.
        
    Returns:
        Average PSNR across all validation samples.
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    total_psnr = 0.0
    count = 0

    for batch in tqdm(loader, desc="PSNR eval", leave=False):
        lr = batch["lr"].numpy()
        hr = batch["hr"]
        sr = session.run([output_name], {input_name: lr})[0]
        sr_tensor = torch.from_numpy(sr)
        total_psnr += calculate_psnr(sr_tensor, hr)
        count += 1

    if count == 0:
        return 0.0
    return total_psnr / count


def benchmark_latency(session: ort.InferenceSession, loader: DataLoader, warmup: int) -> Dict[str, float]:
    """
    Benchmark ONNX model latency and throughput on CPU.
    
    Args:
        session: ONNX Runtime inference session.
        loader: DataLoader with input samples.
        warmup: Number of warmup iterations before timing.
        
    Returns:
        Dictionary with keys:
            - 'avg_latency_ms': Average latency in milliseconds.
            - 'median_latency_ms': Median latency in milliseconds.
            - 'throughput_fps': Throughput in frames per second.
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    samples = [batch["lr"].numpy() for batch in loader]
    if not samples:
        return {"avg_latency_ms": 0.0, "median_latency_ms": 0.0, "throughput_fps": 0.0}

    for _ in range(warmup):
        for data in samples[:2]:
            session.run([output_name], {input_name: data})

    timings: list[float] = []
    for data in tqdm(samples, desc="Latency benchmark", leave=False):
        start = time.perf_counter()
        session.run([output_name], {input_name: data})
        end = time.perf_counter()
        timings.append(end - start)

    if not timings:
        return {"avg_latency_ms": 0.0, "median_latency_ms": 0.0, "throughput_fps": 0.0}

    avg = float(np.mean(timings))
    median = float(np.median(timings))
    throughput = 1.0 / avg if avg > 0 else 0.0
    return {
        "avg_latency_ms": avg * 1000.0,
        "median_latency_ms": median * 1000.0,
        "throughput_fps": throughput,
    }


def model_size_mb(path: Path) -> float:
    """
    Get ONNX model file size in megabytes.
    
    Args:
        path: Path to ONNX model file.
        
    Returns:
        File size in megabytes, or 0.0 if file doesn't exist.
    """
    if not path.is_file():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def run_single(session: ort.InferenceSession, loader: DataLoader, warmup: int) -> Dict[str, float]:
    """
    Run complete benchmark for a single ONNX model (latency and PSNR).
    
    Args:
        session: ONNX Runtime inference session.
        loader: DataLoader with validation samples.
        warmup: Number of warmup iterations before timing.
        
    Returns:
        Dictionary with benchmark metrics (latency, throughput, PSNR).
    """
    metrics = benchmark_latency(session, loader, warmup)
    metrics["psnr"] = eval_psnr(session, loader)
    return metrics


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.base_config)

    val_loaders = get_val_loaders(base_cfg)
    if len(val_loaders) < 2:
        raise SystemExit("Expected Set14 loader (index 1). Check configs/espcn/base.yaml data paths.")
    val_loader = val_loaders[1]

    fp32_path = Path(args.onnx_fp32)
    int8_path = Path(args.onnx_int8)
    if not fp32_path.exists():
        raise FileNotFoundError(f"FP32 ONNX not found: {fp32_path}")
    if not int8_path.exists():
        raise FileNotFoundError(f"INT8 ONNX not found: {int8_path}")

    print(f"[load] FP32 session: {fp32_path}")
    fp32_sess = create_session(fp32_path)
    print(f"[load] INT8 session: {int8_path}")
    int8_sess = create_session(int8_path)

    print("[benchmark] FP32 ONNX")
    fp32_metrics = run_single(fp32_sess, val_loader, args.warmup)
    fp32_metrics["model_size_mb"] = model_size_mb(fp32_path)

    print("[benchmark] INT8 ONNX")
    int8_metrics = run_single(int8_sess, val_loader, args.warmup)
    int8_metrics["model_size_mb"] = model_size_mb(int8_path)

    results = {
        "fp32": fp32_metrics,
        "int8": int8_metrics,
    }

    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[done] Saved ONNX benchmark results to {out_path}")


if __name__ == "__main__":
    main()

