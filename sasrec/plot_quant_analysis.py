from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from sasrec.model.quant import QuantSASRec
from sasrec.data.dataloaders import create_dataloaders
from utils import load_config


class Args:
    def __init__(self, config: Dict[str, Any]):
        self.hidden_units = config["model"]["hidden_units"]
        self.num_blocks = config["model"]["num_blocks"]
        self.num_heads = config["model"]["num_heads"]
        self.dropout_rate = config["model"]["dropout_rate"]
        self.maxlen = config["model"]["maxlen"]
        self.device = torch.device(config["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.norm_first = config["model"].get("norm_first", False)


def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text())


def plot_tradeoffs(records: List[Dict[str, Any]], out_dir: Path) -> None:
    methods = [r["quant_method"] for r in records]
    ndcg = np.array([r["ndcg"] for r in records])
    hit = np.array([r["hit"] for r in records])
    latency = np.array([r["avg_latency_ms"] for r in records])
    size_mb = np.array([r.get("model_size_mb", 0.0) for r in records])

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    for i, m in enumerate(methods):
        plt.scatter(latency[i], ndcg[i], label=m.upper())
        plt.text(latency[i] * 1.01, ndcg[i], m.upper())
    plt.xlabel("Avg latency (ms / batch)")
    plt.ylabel("NDCG@10")
    plt.title("NDCG vs latency (SASRec)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sasrec_ndcg_vs_latency.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    for i, m in enumerate(methods):
        plt.scatter(size_mb[i], ndcg[i], label=m.upper())
        plt.text(size_mb[i] * 1.01, ndcg[i], m.upper())
    plt.xlabel("Model checkpoint size (MB)")
    plt.ylabel("NDCG@10")
    plt.title("NDCG vs model size (SASRec)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sasrec_ndcg_vs_model_size.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    for i, m in enumerate(methods):
        plt.scatter(latency[i], hit[i], label=m.upper())
        plt.text(latency[i] * 1.01, hit[i], m.upper())
    plt.xlabel("Avg latency (ms / batch)")
    plt.ylabel("Hit@10")
    plt.title("Hit@10 vs latency (SASRec)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sasrec_hit_vs_latency.png", dpi=200)
    plt.close()


def build_model(config: Dict[str, Any], usernum: int, itemnum: int, args: Args) -> nn.Module:
    return QuantSASRec(usernum, itemnum, args)


def load_state(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        return state["model_state_dict"]
    return state


@torch.no_grad()
def collect_activations(
    model: nn.Module,
    train_loader,
    device: torch.device,
    max_batches: int = 10,
) -> np.ndarray:
    model.eval().to(device)
    acts: List[torch.Tensor] = []

    def hook(_, __, output):
        if isinstance(output, torch.Tensor):
            acts.append(output.detach().flatten().cpu())

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(hook))

    for i, batch in enumerate(train_loader):
        u, seq, pos, neg = [x.to(device) for x in batch]
        _ = model(u, seq, pos, neg)
        if i + 1 >= max_batches:
            break

    for h in handles:
        h.remove()

    if not acts:
        return np.array([])
    return torch.cat(acts).numpy()


def plot_activation_distributions(
    base_config: Dict[str, Any],
    out_dir: Path,
    checkpoints: Dict[str, Path],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        base_config, seed=base_config["experiment"].get("seed", 42)
    )
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    args = Args(base_config)
    fp32_ckpt = checkpoints.get("fp32")
    if fp32_ckpt is None or not fp32_ckpt.exists():
        print(f"[warn] FP32 checkpoint not found at {fp32_ckpt}, skipping activation plots.")
        return

    fp32_state = load_state(fp32_ckpt, device)
    fp32_model = build_model(base_config, usernum, itemnum, args)
    fp32_model.load_state_dict(fp32_state, strict=False)

    fp32_acts = collect_activations(fp32_model, train_loader, device)
    if fp32_acts.size == 0:
        print("[warn] No activations collected for FP32 model.")
        return

    methods = ["lsq", "apot", "qdrop", "adaround"]
    for m in methods:
        ckpt = checkpoints.get(m)
        if ckpt is None or not ckpt.exists():
            print(f"[warn] checkpoint for {m} not found: {ckpt}, skipping activation plots.")
            continue

        quant_state = load_state(ckpt, device)
        quant_model = build_model(base_config, usernum, itemnum, args)
        quant_model.load_state_dict(quant_state, strict=False)
        quant_acts = collect_activations(quant_model, train_loader, device)
        if quant_acts.size == 0:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 4))
        bins = np.linspace(
            np.percentile(fp32_acts, 1.0),
            np.percentile(fp32_acts, 99.0),
            100,
        )
        plt.hist(fp32_acts, bins=bins, alpha=0.5, label="FP32", density=True)
        plt.hist(quant_acts, bins=bins, alpha=0.5, label=m.upper(), density=True)
        plt.xlabel("Activation value")
        plt.ylabel("Density")
        plt.title(f"Activation distribution: FP32 vs {m.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"sasrec_activations_fp32_vs_{m}.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SASRec quantization analysis.")
    parser.add_argument(
        "--benchmark-json",
        type=str,
        default="results/sasrec_quant_benchmark.json",
        help="Benchmark JSON produced by sasrec/benchmark_quant.py",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/sasrec/base.yaml",
        help="Base SASRec config.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/plots_sasrec",
        help="Directory to save plots.",
    )
    args = parser.parse_args()

    records = load_benchmark(Path(args.benchmark_json))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if records:
        plot_tradeoffs(records, out_dir)

    base_config = load_config(args.base_config)
    checkpoints = {
        "fp32": Path("checkpoints/sasrec_fp32/sasrec_fp32.pth"),
        "lsq": Path("checkpoints/sasrec_lsq/sasrec_lsq.pth"),
        "apot": Path("checkpoints/sasrec_apot/sasrec_apot.pth"),
        "qdrop": Path("checkpoints/sasrec_qdrop/sasrec_qdrop.pth"),
        "adaround": Path("checkpoints/sasrec_adaround/sasrec_adaround.pth"),
    }
    plot_activation_distributions(base_config, out_dir, checkpoints)


if __name__ == "__main__":
    main()


