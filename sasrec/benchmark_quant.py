from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sasrec.data.dataloaders import create_dataloaders
from sasrec.model.quant import QuantSASRec
from sasrec.metrics import ndcg_k, hit_k
from utils import load_config, ensure_dir


class Args:
    def __init__(self, config: Dict[str, Any]):
        self.hidden_units = config["model"]["hidden_units"]
        self.num_blocks = config["model"]["num_blocks"]
        self.num_heads = config["model"]["num_heads"]
        self.dropout_rate = config["model"]["dropout_rate"]
        self.maxlen = config["model"]["maxlen"]
        self.device = torch.device(config["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.norm_first = config["model"].get("norm_first", False)


def build_model(config: Dict[str, Any], usernum: int, itemnum: int, args: Args) -> nn.Module:
    return QuantSASRec(usernum, itemnum, args)


@torch.no_grad()
def evaluate(model: nn.Module, dataset: List, args: Args) -> Tuple[float, float]:
    model.eval()
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    ndcg = 0.0
    ht = 0.0
    valid_user = 0.0
    users = range(1, usernum + 1)

    for u in tqdm(users, desc="Eval", leave=False):
        if len(user_train[u]) < 1 or len(user_valid[u]) < 1:
            continue

        seq = user_train[u]
        target_item = user_valid[u][0]

        seq_len = len(seq)
        if seq_len >= args.maxlen:
            seq = seq[-args.maxlen:]
        else:
            seq = [0] * (args.maxlen - seq_len) + seq

        log_seqs = torch.tensor([seq], dtype=torch.long, device=args.device)
        item_indices = list(range(1, itemnum + 1))

        predictions = model.predict(
            user_ids=None,
            log_seqs=log_seqs,
            item_indices=item_indices,
        )
        predictions = -predictions[0]
        _, indices = torch.topk(predictions, 10, largest=False)
        rank_list = [item_indices[i] for i in indices.cpu().numpy()]

        ndcg += ndcg_k([target_item], rank_list, k=10)
        ht += hit_k([target_item], rank_list, k=10)
        valid_user += 1

    if valid_user == 0:
        return 0.0, 0.0
    return ndcg / valid_user, ht / valid_user


@torch.no_grad()
def benchmark_cpu_latency(
    model: nn.Module,
    train_loader: DataLoader,
    warmup: int = 5,
    iters: int = 50,
) -> Dict[str, float]:
    device = torch.device("cpu")
    model.eval().to(device)

    batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for batch in train_loader:
        u, seq, pos, neg = [x.to(device) for x in batch]
        batches.append((u, seq, pos, neg))
        if len(batches) >= iters:
            break
    if not batches:
        return {"throughput_samples_per_sec": 0.0, "avg_latency_ms": 0.0, "median_latency_ms": 0.0}

    for _ in range(warmup):
        for u, seq, pos, neg in batches[:2]:
            _ = model(u, seq, pos, neg)

    times: List[float] = []
    for u, seq, pos, neg in tqdm(batches, desc="CPU benchmark", leave=False):
        start = time.perf_counter()
        _ = model(u, seq, pos, neg)
        end = time.perf_counter()
        times.append(end - start)

    if not times:
        return {"throughput_samples_per_sec": 0.0, "avg_latency_ms": 0.0, "median_latency_ms": 0.0}

    avg_latency = float(np.mean(times))
    median_latency = float(np.median(times))
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
    usernum: int,
    itemnum: int,
) -> nn.Module:
    args = Args(config)
    model = build_model(config, usernum, itemnum, args)
    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SASRec quantization methods vs FP32.")
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/sasrec/base.yaml",
        help="Base SASRec config.",
    )
    parser.add_argument(
        "--results-out",
        type=str,
        default="results/sasrec_quant_benchmark.json",
        help="Where to store benchmark results JSON.",
    )
    args = parser.parse_args()

    config = load_config(args.base_config)
    device = torch.device("cpu")

    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        config, seed=config["experiment"].get("seed", 42)
    )
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    root_ckpt = Path(config["paths"]["checkpoints_dir"])
    checkpoints = {
        "fp32": root_ckpt / "sasrec_fp32" / "sasrec_fp32.pth",
        "lsq": root_ckpt / "sasrec_lsq" / "sasrec_lsq.pth",
        "apot": root_ckpt / "sasrec_apot" / "sasrec_apot.pth",
        "qdrop": root_ckpt / "sasrec_qdrop" / "sasrec_qdrop.pth",
        "adaround": root_ckpt / "sasrec_adaround" / "sasrec_adaround.pth",
    }

    records: List[Dict[str, Any]] = []

    for name, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            print(f"[warn] checkpoint for {name} not found: {ckpt_path}, skipping.")
            continue

        print(f"\n=== Benchmarking {name.upper()} ===")
        model = load_model_from_checkpoint(config, ckpt_path, device, usernum, itemnum)

        ndcg, ht = evaluate(model, dataset, Args(config))
        print(f"{name}: NDCG@10={ndcg:.4f} | Hit@10={ht:.4f}")

        cpu_metrics = benchmark_cpu_latency(model, train_loader)
        size_mb = model_size_mb(ckpt_path)

        rec = {
            "model": f"sasrec_{name}",
            "quant_method": name,
            "ndcg": ndcg,
            "hit": ht,
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


