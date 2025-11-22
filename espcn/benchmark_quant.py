from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from espcn.data.dataloaders import get_val_loaders
from espcn.model.quant import QuantESPCN
from utils import load_config, ensure_dir, get_espcn_method_configs


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
    method: str = "fp32",
    quant_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    model = build_model(config)
    raw_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in raw_state:
        state_dict = raw_state["model_state_dict"]
    else:
        state_dict = raw_state

    method = method.lower()

    if method in ("lsq", "apot", "qdrop"):
        if quant_config is None:
            raise ValueError(f"quant_config must be provided for QAT method '{method}'.")
        model.prepare_quant(method, quant_config)

        print("Modules after attach:")
        for name, _ in model.named_modules():
            if "quantizer" in name:
                print("  ", name)

        raw_state = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in raw_state:
            state_dict = raw_state["model_state_dict"]
        else:
            state_dict = raw_state

        print("\nKeys in checkpoint with 'quantizer':")
        for k in state_dict.keys():
            if "quantizer" in k:
                print("  ", k)
        
    elif method == "adaround":
        if quant_config is None and isinstance(raw_state, dict):
            quant_config = raw_state.get("config", {})
        if quant_config is None:
            raise ValueError("AdaRound checkpoint does not contain 'config' field for quantization.")
        model.prepare_quant("adaround", quant_config)
        from quant.adaround import AdaRoundModule
        adapted_state_dict = {}
        for k, v in state_dict.items():
            if ("running_min" in k or "running_max" in k) and v.dim() == 0:
                adapted_state_dict[k] = v.reshape(1)
            else:
                adapted_state_dict[k] = v
        for name, module in model.named_modules():
            if isinstance(module, AdaRoundModule) and getattr(module, "alpha", None) is None:
                alpha_key = f"{name}.alpha"
                if alpha_key in adapted_state_dict:
                    loaded_alpha = adapted_state_dict[alpha_key]
                    module.alpha = nn.Parameter(loaded_alpha.clone().detach())
                    module.register_parameter("alpha", module.alpha)
                    module.alpha_init.fill_(True)
                else:
                    module._init_alpha(module.weight)

        model.load_state_dict(adapted_state_dict, strict=True)
        model.to(device)
        return model

    adapted_state_dict = {}
    for k, v in state_dict.items():
        if ("running_min" in k or "running_max" in k) and v.dim() == 0:
            # преобразуем скаляр ([]) → вектор длины 1 ([1])
            adapted_state_dict[k] = v.reshape(1)
        else:
            adapted_state_dict[k] = v

    model.load_state_dict(adapted_state_dict, strict=True)
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

    method_configs = get_espcn_method_configs()

    qat_configs: Dict[str, Dict[str, Any]] = {}
    for m in ("lsq", "apot", "qdrop"):
        cfg_path = Path(f"configs/espcn/espcn_{m}.yaml")
        if cfg_path.exists():
            qat_configs[m] = load_config(str(cfg_path))

    records: List[Dict[str, Any]] = []

    for method_name, (method_config, ckpt_name) in method_configs.items():
        method_cfg_path = Path(method_config)
        if not method_cfg_path.exists():
            print(f"[warn] config for {method_name} not found: {method_cfg_path}, skipping.")
            continue
        method_full_cfg = load_config(str(method_cfg_path))
        ckpt_path = ckpt_dir / ckpt_name

        if not ckpt_path.exists():
            print(f"[warn] checkpoint for {method_name} not found: {ckpt_path}, skipping.")
            continue

        print(f"\n=== Benchmarking {method_name.upper()} ===")
        quant_cfg = None
        if method_name in ("lsq", "apot", "qdrop"):
            quant_cfg = method_full_cfg.get("quantization", {})

        model = load_model_from_checkpoint(
            base_config,
            ckpt_path,
            device,
            method=method_name,
            quant_config=quant_cfg,
        )

        psnr, ssim = eval_psnr_ssim(model, val_loader, device)
        print(f"{method_name}: PSNR={psnr:.4f} dB | SSIM={ssim:.4f}")

        cpu_metrics = benchmark_cpu_latency(model, val_loader)
        size_mb = model_size_mb(ckpt_path)

        rec = {
            "model": method_cfg_path.stem,
            "quant_method": method_name,
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


