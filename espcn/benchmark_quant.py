from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from espcn.data.dataloaders import get_val_loaders
from espcn.model.quant import QuantESPCN
from utils import load_config, ensure_dir


def build_model(config: Dict[str, Any]) -> nn.Module:
    model_cfg = config["model"].copy()
    return QuantESPCN(**model_cfg)


@torch.no_grad()
def eval_psnr_ssim(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    from espcn.train import calculate_psnr, calculate_ssim

    model.eval().to(device)
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    for batch in loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        sr = model(lr)
        total_psnr += calculate_psnr(sr, hr)
        total_ssim += calculate_ssim(sr, hr)
        n += 1

    if n == 0:
        return 0.0, 0.0
    return total_psnr / n, total_ssim / n


@torch.no_grad()
def benchmark_cpu_latency(
    model: nn.Module,
    loader: DataLoader,
    warmup: int = 5,
    iters: int = 50,
) -> Dict[str, float]:
    device = torch.device("cpu")
    model.eval().to(device)

    inputs: List[torch.Tensor] = []
    for batch in loader:
        inputs.append(batch["lr"].to(device))
    if not inputs:
        return {"throughput_samples_per_sec": 0.0, "avg_latency_ms": 0.0, "median_latency_ms": 0.0}

    for _ in range(warmup):
        for x in inputs[:2]:
            _ = model(x)

    times: List[float] = []
    for x in tqdm(inputs, desc="CPU benchmark", leave=False):
        start = time.perf_counter()
        _ = model(x)
        end = time.perf_counter()
        times.append(end - start)

    if not times:
        return {"throughput_samples_per_sec": 0.0, "avg_latency_ms": 0.0, "median_latency_ms": 0.0}

    avg_latency = sum(times) / len(times)
    median_latency = sorted(times)[len(times) // 2]
    throughput = 1.0 / avg_latency if avg_latency > 0 else 0.0

    return {
        "throughput_samples_per_sec": throughput,
        "avg_latency_ms": avg_latency * 1000.0,
        "median_latency_ms": median_latency * 1000.0,
    }


def model_size_mb(checkpoint_path: Path) -> float:
    if not checkpoint_path.is_file():
        return 0.0
    size_bytes = checkpoint_path.stat().st_size
    return size_bytes / (1024 * 1024)


def load_model_from_checkpoint(
    config: Dict[str, Any],
    ckpt_path: Path,
    device: torch.device,
) -> nn.Module:
    model = build_model(config)
    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ESPCN quantization methods vs FP32.")
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/espcn/base.yaml",
        help="Base ESPCN config (defines data paths, model, etc).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/espcn_run",
        help="Directory with trained ESPCN checkpoints.",
    )
    parser.add_argument(
        "--results-out",
        type=str,
        default="results/espcn_quant_benchmark.json",
        help="Where to store benchmark results JSON.",
    )
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    device = torch.device("cpu")

    val_loaders = get_val_loaders(base_config)
    if not val_loaders:
        raise SystemExit("No validation loaders found. Check data paths in config.")
    val_loader = val_loaders[1]

    ckpt_dir = Path(args.checkpoint_dir)

    methods = {
        "fp32": ckpt_dir / "espcn_fp32.pth",
        "lsq": ckpt_dir / "espcn_lsq.pth",
        "apot": ckpt_dir / "espcn_apot.pth",
        "qdrop": ckpt_dir / "espcn_qdrop.pth",
        "adaround": ckpt_dir / "espcn_adaround.pth",
    }

    records: List[Dict[str, Any]] = []

    for name, ckpt_path in methods.items():
        if not ckpt_path.exists():
            print(f"[warn] checkpoint for {name} not found: {ckpt_path}, skipping.")
            continue

        print(f"\n=== Benchmarking {name.upper()} ===")
        model = load_model_from_checkpoint(base_config, ckpt_path, device)

        psnr, ssim = eval_psnr_ssim(model, val_loader, device)
        print(f"{name}: PSNR={psnr:.4f} dB | SSIM={ssim:.4f}")

        cpu_metrics = benchmark_cpu_latency(model, val_loader)
        size_mb = model_size_mb(ckpt_path)

        rec = {
            "model": f"espcn_{name}",
            "quant_method": name,
            "psnr_y": psnr,
            "ssim": ssim,
            "throughput_samples_per_sec": cpu_metrics["throughput_samples_per_sec"],
            "avg_latency_ms": cpu_metrics["avg_latency_ms"],
            "median_latency_ms": cpu_metrics["median_latency_ms"],
            "checkpoint_path": str(ckpt_path),
            "model_size_mb": size_mb,
        }
        records.append(rec)

    out_path = Path(args.results_out)
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(records, indent=2))
    print(f"\nSaved benchmark results to {out_path}")


if __name__ == "__main__":
    main()


