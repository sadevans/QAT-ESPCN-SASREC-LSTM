from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from espcn.model.quant import QuantESPCN
from espcn.data.dataloaders import get_val_loaders
from utils import load_config


def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text())


def plot_tradeoffs(records: List[Dict[str, Any]], out_dir: Path) -> None:
    methods = [r["quant_method"] for r in records]
    psnr = np.array([r["psnr_y"] for r in records])
    ssim = np.array([r.get("ssim", 0.0) for r in records])
    latency = np.array([r["avg_latency_ms"] for r in records])
    size_mb = np.array([r.get("model_size_mb", 0.0) for r in records])

    out_dir.mkdir(parents=True, exist_ok=True)

    # PSNR vs latency
    plt.figure(figsize=(6, 4))
    for i, m in enumerate(methods):
        plt.scatter(latency[i], psnr[i], label=m.upper())
        plt.text(latency[i] * 1.01, psnr[i], m.upper())
    plt.xlabel("Avg latency (ms / image)")
    plt.ylabel("PSNR-Y (dB)")
    plt.title("PSNR vs latency (ESPCN)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "espcn_psnr_vs_latency.png", dpi=200)
    plt.close()

    # PSNR vs model size
    plt.figure(figsize=(6, 4))
    for i, m in enumerate(methods):
        plt.scatter(size_mb[i], psnr[i], label=m.upper())
        plt.text(size_mb[i] * 1.01, psnr[i], m.upper())
    plt.xlabel("Model checkpoint size (MB)")
    plt.ylabel("PSNR-Y (dB)")
    plt.title("PSNR vs model size (ESPCN)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "espcn_psnr_vs_model_size.png", dpi=200)
    plt.close()

    # SSIM vs latency
    plt.figure(figsize=(6, 4))
    for i, m in enumerate(methods):
        plt.scatter(latency[i], ssim[i], label=m.upper())
        plt.text(latency[i] * 1.01, ssim[i], m.upper())
    plt.xlabel("Avg latency (ms / image)")
    plt.ylabel("SSIM")
    plt.title("SSIM vs latency (ESPCN)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "espcn_ssim_vs_latency.png", dpi=200)
    plt.close()


def build_model(config: Dict[str, Any]) -> nn.Module:
    model_cfg = config["model"].copy()
    return QuantESPCN(**model_cfg)


def load_state(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def plot_weight_distributions(
    fp32_state: Dict[str, Any],
    quant_state: Dict[str, Any],
    out_dir: Path,
    method_name: str,
) -> None:
    fp32_weights = []
    quant_weights = []
    for k, v in fp32_state.items():
        if "weight" in k and v.dtype.is_floating_point:
            fp32_weights.append(v.flatten().cpu().numpy())
    for k, v in quant_state.items():
        if "weight" in k and v.dtype.is_floating_point:
            quant_weights.append(v.flatten().cpu().numpy())

    if not fp32_weights or not quant_weights:
        return

    fp32_all = np.concatenate(fp32_weights)
    quant_all = np.concatenate(quant_weights)

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    bins = np.linspace(
        np.percentile(fp32_all, 0.1),
        np.percentile(fp32_all, 99.9),
        100,
    )
    plt.hist(fp32_all, bins=bins, alpha=0.5, label="FP32", density=True)
    plt.hist(quant_all, bins=bins, alpha=0.5, label=method_name.upper(), density=True)
    plt.xlabel("Weight value")
    plt.ylabel("Density")
    plt.title(f"Weight distribution: FP32 vs {method_name.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"weights_fp32_vs_{method_name}.png", dpi=200)
    plt.close()


@torch.no_grad()
def plot_activation_distributions(
    fp32_model: nn.Module,
    quant_model: nn.Module,
    config: Dict[str, Any],
    out_dir: Path,
    method_name: str,
) -> None:
    from espcn.data.dataloaders import get_val_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp32_model.eval().to(device)
    quant_model.eval().to(device)

    # Grab first validation sample
    val_loaders = get_val_loaders(config)
    if not val_loaders:
        return
    sample_batch = next(iter(val_loaders[0]))
    lr = sample_batch["lr"].to(device)

    fp32_acts: List[torch.Tensor] = []
    quant_acts: List[torch.Tensor] = []

    def make_hook(storage: List[torch.Tensor]):
        def _hook(_, __, output):
            if isinstance(output, torch.Tensor):
                storage.append(output.detach().flatten().cpu())
        return _hook

    # Attach hooks to convolutional layers
    fp32_handles = []
    quant_handles = []
    for m in fp32_model.modules():
        if isinstance(m, nn.Conv2d):
            fp32_handles.append(m.register_forward_hook(make_hook(fp32_acts)))
    for m in quant_model.modules():
        if isinstance(m, nn.Conv2d):
            quant_handles.append(m.register_forward_hook(make_hook(quant_acts)))

    _ = fp32_model(lr)
    _ = quant_model(lr)

    for h in fp32_handles:
        h.remove()
    for h in quant_handles:
        h.remove()

    if not fp32_acts or not quant_acts:
        return

    fp32_all = torch.cat(fp32_acts).numpy()
    quant_all = torch.cat(quant_acts).numpy()

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    bins = np.linspace(
        np.percentile(fp32_all, 0.1),
        np.percentile(fp32_all, 99.9),
        100,
    )
    plt.hist(fp32_all, bins=bins, alpha=0.5, label="FP32", density=True)
    plt.hist(quant_all, bins=bins, alpha=0.5, label=method_name.upper(), density=True)
    plt.xlabel("Activation value")
    plt.ylabel("Density")
    plt.title(f"Activation distribution: FP32 vs {method_name.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"activations_fp32_vs_{method_name}.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ESPCN quantization analysis.")
    parser.add_argument(
        "--benchmark-json",
        type=str,
        default="results/espcn_quant_benchmark.json",
        help="Benchmark JSON produced by espcn/benchmark_quant.py",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/espcn/base.yaml",
        help="Base ESPCN config.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/espcn_run",
        help="Directory with ESPCN checkpoints.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/plots_espcn",
        help="Directory to save plots.",
    )
    args = parser.parse_args()

    records = load_benchmark(Path(args.benchmark_json))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if records:
        plot_tradeoffs(records, out_dir)

    # Weight / activation comparisons for each quant method vs FP32
    base_config = load_config(args.base_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(args.checkpoint_dir)
    fp32_ckpt = ckpt_dir / "espcn_fp32.pth"
    if not fp32_ckpt.exists():
        print(f"[warn] FP32 checkpoint not found at {fp32_ckpt}, skipping weight/activation plots.")
        return

    fp32_state = load_state(fp32_ckpt, device)
    fp32_model = build_model(base_config)
    fp32_model.load_state_dict(fp32_state, strict=True)

    methods = ["lsq", "apot", "qdrop", "adaround"]
    for m in methods:
        ckpt_path = ckpt_dir / f"espcn_{m}.pth"
        if not ckpt_path.exists():
            print(f"[warn] checkpoint for {m} not found: {ckpt_path}, skipping.")
            continue

        quant_state = load_state(ckpt_path, device)
        plot_weight_distributions(fp32_state, quant_state, out_dir, m)

        quant_model = build_model(base_config)
        quant_model.load_state_dict(quant_state, strict=True)
        plot_activation_distributions(fp32_model, quant_model, base_config, out_dir, m)


if __name__ == "__main__":
    main()


