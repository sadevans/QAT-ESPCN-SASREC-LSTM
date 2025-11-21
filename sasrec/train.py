from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from sasrec.data.dataloaders import create_dataloaders
from sasrec.model.quant import QuantSASRec
from sasrec.metrics import ndcg_k, hit_k
from utils import (
    configure_logging,
    load_config,
    set_random_seeds,
    init_clearml_task,
    ensure_dir
)


class Args:
    """Helper to pass config parameters to SASRec model."""
    def __init__(self, config: Dict[str, Any]):
        self.hidden_units = config["model"]["hidden_units"]
        self.num_blocks = config["model"]["num_blocks"]
        self.num_heads = config["model"]["num_heads"]
        self.dropout_rate = config["model"]["dropout_rate"]
        self.maxlen = config["model"]["maxlen"]
        self.device = torch.device(config["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.norm_first = config["model"].get("norm_first", False)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    checkpoint_dir: str,
    filename: str,
    metrics: Optional[Dict[str, float]] = None
):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if metrics is not None:
        state.update(metrics)
        
    torch.save(state, path)


def evaluate(
    model: nn.Module,
    dataset: List,
    args: Args,
    logger=None,
    epoch=0
) -> Tuple[float, float]:
    """Evaluate model (NDCG@10 and Hit@10)."""
    model.eval()
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    users = range(1, usernum + 1)
    
    for u in tqdm(users, desc="Evaluating", leave=False):
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
        
        with torch.no_grad():
            predictions = model.predict(
                user_ids=None, 
                log_seqs=log_seqs,
                item_indices=item_indices
            )
            predictions = -predictions[0] # negative for ascending sort

            _, indices = torch.topk(predictions, 10, largest=False)
            rank_list = [item_indices[i] for i in indices.cpu().numpy()]
            
            NDCG += ndcg_k([target_item], rank_list, k=10)
            HT += hit_k([target_item], rank_list, k=10)
            valid_user += 1

    if valid_user == 0:
        return 0.0, 0.0

    avg_ndcg = NDCG / valid_user
    avg_hit = HT / valid_user

    if logger:
        logger.report_scalar("NDCG@10", "Val", value=avg_ndcg, iteration=epoch)
        logger.report_scalar("Hit@10", "Val", value=avg_hit, iteration=epoch)

    return avg_ndcg, avg_hit


def evaluate_test(
    model: nn.Module,
    dataset: List,
    args: Args,
    logger=None,
    epoch: int = 0,
) -> Tuple[float, float]:
    """Final testing on user_test split (NDCG@10 and Hit@10)."""
    model.eval()
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    users = range(1, usernum + 1)

    for u in tqdm(users, desc="Testing", leave=False):
        if len(user_train[u]) < 1 or len(user_test[u]) < 1:
            continue

        seq = user_train[u]
        target_item = user_test[u][0]

        seq_len = len(seq)
        if seq_len >= args.maxlen:
            seq = seq[-args.maxlen:]
        else:
            seq = [0] * (args.maxlen - seq_len) + seq

        log_seqs = torch.tensor([seq], dtype=torch.long, device=args.device)
        item_indices = list(range(1, itemnum + 1))

        with torch.no_grad():
            predictions = model.predict(
                user_ids=None,
                log_seqs=log_seqs,
                item_indices=item_indices,
            )
            predictions = -predictions[0]  # negative for ascending sort

            _, indices = torch.topk(predictions, 10, largest=False)
            rank_list = [item_indices[i] for i in indices.cpu().numpy()]

            NDCG += ndcg_k([target_item], rank_list, k=10)
            HT += hit_k([target_item], rank_list, k=10)
            valid_user += 1

    if valid_user == 0:
        return 0.0, 0.0

    avg_ndcg = NDCG / valid_user
    avg_hit = HT / valid_user

    if logger:
        logger.report_scalar("NDCG@10", "Test", value=avg_ndcg, iteration=epoch)
        logger.report_scalar("Hit@10", "Test", value=avg_hit, iteration=epoch)

    return avg_ndcg, avg_hit


def train_epoch(
    model: nn.Module, 
    train_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device,
    logger=None,
    epoch=0
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        u, seq, pos, neg = [x.to(device) for x in batch]
        
        pos_logits, neg_logits = model(u, seq, pos, neg)
        
        pos_labels = torch.ones(pos_logits.shape, device=device)
        neg_labels = torch.zeros(neg_logits.shape, device=device)
        
        optimizer.zero_grad()
        indices = np.where(pos.cpu().numpy() != 0) # Ignore padding
        loss = criterion(pos_logits[indices], pos_labels[indices])
        loss += criterion(neg_logits[indices], neg_labels[indices])
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    if logger:
        logger.report_scalar("Loss", "Train", value=avg_loss, iteration=epoch)
        
    return avg_loss


def train_fp32(
    model: nn.Module,
    train_loader: DataLoader,
    dataset: List,
    config: Dict[str, Any],
    criterion: nn.Module,
    args: Args,
    checkpoint_dir: str,
    save_name: str = "sasrec_fp32.pth",
    logger=None
):
    """Train FP32 model."""
    print("Starting FP32 training...")
    
    optimizer = Adam(
        model.parameters(), 
        lr=config["optimization"]["lr"],
        betas=tuple(config["optimization"].get("betas", (0.9, 0.98))),
        weight_decay=config["optimization"].get("weight_decay", 0.0)
    )
    
    epochs = config["training"]["epochs"]
    eval_interval = config["training"].get("eval_interval", 20)
    best_ndcg = 0.0
    
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, args.device, logger=logger, epoch=epoch)
        
        if epoch % eval_interval == 0:
            ndcg, ht = evaluate(model, dataset, args, logger=logger, epoch=epoch)
            print(f"[FP32] Epoch {epoch}/{epochs} | Loss: {loss:.4f} | NDCG@10: {ndcg:.4f} | Hit@10: {ht:.4f}")
            
            results = {"ndcg": ndcg, "hit": ht}
            with open(Path(config["paths"]["results_dir"]) / f"{config['experiment']['run_name']}_results.json", "w") as f:
                json.dump(results, f, indent=2)

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                save_checkpoint(
                    model, optimizer, epoch, checkpoint_dir, save_name,
                    metrics={"ndcg": ndcg, "hit": ht}
                )
                print(f"Saved best FP32 checkpoint (NDCG: {ndcg:.4f})")
        else:
            print(f"[FP32] Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

    print(f"FP32 training done. Best NDCG (Val): {best_ndcg:.4f}")

    ckpt_path = os.path.join(checkpoint_dir, save_name)
    if os.path.exists(ckpt_path):
        print(f"Loading best FP32 checkpoint from {ckpt_path} for final testing...")
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])

    print("Running final test evaluation (user_test split)...")
    test_ndcg, test_hit = evaluate_test(model, dataset, args, logger=None, epoch=0)
    print(f"[FP32][Test] NDCG@10: {test_ndcg:.4f} | Hit@10: {test_hit:.4f}")

    results_path = Path(config["paths"]["results_dir"]) / f"{config['experiment']['run_name']}_results.json"
    results: Dict[str, Any] = {}
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}
    results.update({"test_ndcg": test_ndcg, "test_hit": test_hit})
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def train_qat(
    model: nn.Module,
    train_loader: DataLoader,
    dataset: List,
    config: Dict[str, Any],
    criterion: nn.Module,
    args: Args,
    strategy_name: str,
    quant_config: Dict[str, Any],
    checkpoint_dir: str,
    save_name: str = "sasrec_qat.pth",
    logger=None
):
    """Train QAT model."""
    print(f"Starting QAT with {strategy_name.upper()}...")
    
    model.prepare_quant(strategy_name, quant_config)
    
    print("Running dummy pass to initialize quantization parameters...")
    model.train()
    with torch.no_grad():
        for batch in train_loader:
            u, seq, pos, neg = [x.to(args.device) for x in batch]
            model(u, seq, pos, neg)
            break
            
    optimizer = Adam(
        model.parameters(), 
        lr=config["optimization"]["lr"],
        betas=tuple(config["optimization"].get("betas", (0.9, 0.98))),
        weight_decay=config["optimization"].get("weight_decay", 0.0)
    )
    
    epochs = config["training"]["epochs"]
    eval_interval = config["training"].get("eval_interval", 20)
    best_ndcg = 0.0
    
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, args.device, logger=logger, epoch=epoch)
        
        if epoch % eval_interval == 0:
            ndcg, ht = evaluate(model, dataset, args, logger=logger, epoch=epoch)
            print(f"[QAT {strategy_name}] Epoch {epoch}/{epochs} | Loss: {loss:.4f} | NDCG@10: {ndcg:.4f} | Hit@10: {ht:.4f}")
            
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                save_checkpoint(
                    model, optimizer, epoch, checkpoint_dir, save_name,
                    metrics={"ndcg": ndcg, "hit": ht}
                )
                print(f"Saved best QAT checkpoint (NDCG: {ndcg:.4f})")
        else:
             print(f"[QAT {strategy_name}] Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

    print(f"QAT ({strategy_name}) done. Best NDCG (Val): {best_ndcg:.4f}")

    ckpt_path = os.path.join(checkpoint_dir, save_name)
    if os.path.exists(ckpt_path):
        print(f"Loading best QAT checkpoint from {ckpt_path} for final testing...")
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])

    print("Running final test evaluation (user_test split)...")
    test_ndcg, test_hit = evaluate_test(model, dataset, args, logger=None, epoch=0)
    print(f"[QAT {strategy_name}][Test] NDCG@10: {test_ndcg:.4f} | Hit@10: {test_hit:.4f}")

    results_path = Path(config["paths"]["results_dir"]) / f"{config['experiment']['run_name']}_results.json"
    results: Dict[str, Any] = {}
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}
    results.update({f"test_ndcg_{strategy_name}": test_ndcg, f"test_hit_{strategy_name}": test_hit})
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def apply_adaround(
    model: nn.Module,
    train_loader: DataLoader,
    dataset: List,
    args: Args,
    fp32_checkpoint: str,
    adaround_config: Dict[str, Any],
    checkpoint_dir: str,
    save_name: str = "sasrec_adaround.pth",
    logger=None
):
    """Apply AdaRound PTQ."""
    print("Starting AdaRound PTQ...")
    
    fp32_path = os.path.join(checkpoint_dir, fp32_checkpoint)
    if not os.path.exists(fp32_path):
        raise FileNotFoundError(f"FP32 checkpoint not found: {fp32_path}")
        
    print(f"Loading FP32 checkpoint: {fp32_checkpoint}")
    ckpt = torch.load(fp32_path, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    
    model.prepare_quant("adaround", adaround_config)

    if logger is not None and hasattr(model, "quant_strategy") and hasattr(model.quant_strategy, "set_logger"):
        model.quant_strategy.set_logger(logger)
    
    print("Running AdaRound calibration...")
    model.calibrate(train_loader)
    
    print("Validating AdaRound model on validation split...")
    ndcg_val, ht_val = evaluate(model, dataset, args, logger=logger, epoch=1)
    print(f"[AdaRound][Val] NDCG@10: {ndcg_val:.4f} | Hit@10: {ht_val:.4f}")

    print("Running final test evaluation for AdaRound model (user_test split)...")
    ndcg_test, ht_test = evaluate_test(model, dataset, args, logger=None, epoch=0)
    print(f"[AdaRound][Test] NDCG@10: {ndcg_test:.4f} | Hit@10: {ht_test:.4f}")

    save_checkpoint(
        model,
        None,
        0,
        checkpoint_dir,
        save_name,
        metrics={
            "ndcg_val": ndcg_val,
            "hit_val": ht_val,
            "ndcg_test": ndcg_test,
            "hit_test": ht_test,
            "config": adaround_config,
        },
    )
    print(f"AdaRound done. Model saved as {save_name}")


def main():
    parser = argparse.ArgumentParser(description="Train SASRec with quantization.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args_cli = parser.parse_args()
    
    configure_logging()
    config = load_config(args_cli.config)
    set_random_seeds(config["experiment"].get("seed", 42))
    
    print(f"Experiment: {config['experiment']}")
    
    args = Args(config)
    
    print("Loading data...")
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        config, seed=config["experiment"].get("seed", 42)
    )
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    print("Building model...")
    model = QuantSASRec(usernum, itemnum, args).to(args.device)
    
    quant_cfg = config.get("quantization", {})
    strategy_name = quant_cfg.get("method", "fp32").lower()
    
    checkpoint_dir = Path(config.get("paths", {}).get("checkpoints_dir", "./checkpoints"))
    run_name = config["experiment"].get("run_name", "sasrec_run")
    checkpoint_subdir = checkpoint_dir / run_name
    ensure_dir(checkpoint_subdir)
    
    results_dir = Path(config.get("paths", {}).get("results_dir", "./results"))
    ensure_dir(results_dir)
    
    logging_cfg = config.get("logging", {})
    clearml_task = init_clearml_task(logging_cfg, config)
    logger = clearml_task.get_logger() if clearml_task else None
    
    criterion = nn.BCEWithLogitsLoss()
    
    if strategy_name == "fp32":
        train_fp32(
            model=model,
            train_loader=train_loader,
            dataset=dataset,
            config=config,
            criterion=criterion,
            args=args,
            checkpoint_dir=str(checkpoint_subdir),
            save_name="sasrec_fp32.pth",
            logger=logger
        )
        
    elif strategy_name in ("lsq", "apot", "qdrop"):
        train_qat(
            model=model,
            train_loader=train_loader,
            dataset=dataset,
            config=config,
            criterion=criterion,
            args=args,
            strategy_name=strategy_name,
            quant_config=quant_cfg,
            checkpoint_dir=str(checkpoint_subdir),
            save_name=f"sasrec_{strategy_name}.pth",
            logger=logger
        )
        
    elif strategy_name == "adaround":
        fp32_epochs = config["training"].get("fp32_epochs", 0)
        fp32_ckpt = quant_cfg.get("base_checkpoint", None)
        
        fp32_ckpt_name = "sasrec_fp32_for_adaround.pth"
        
        if fp32_epochs > 0:
             train_fp32(
                model=model,
                train_loader=train_loader,
                dataset=dataset,
                config=config,
                criterion=criterion,
                args=args,
                checkpoint_dir=str(checkpoint_subdir),
                save_name=fp32_ckpt_name,
                logger=logger
            )
        elif fp32_ckpt:
             fp32_ckpt_path = Path(fp32_ckpt)
             if not fp32_ckpt_path.exists():
                 fp32_ckpt_path = checkpoint_subdir / fp32_ckpt
             
             if not fp32_ckpt_path.exists():
                  raise FileNotFoundError(f"Base checkpoint not found: {fp32_ckpt}")
             if fp32_ckpt_path.parent == checkpoint_subdir:
                 fp32_ckpt_name = fp32_ckpt_path.name
             else:
                 fp32_ckpt_name = str(fp32_ckpt_path)

        else:
             raise ValueError("AdaRound requires either fp32_epochs > 0 or base_checkpoint.")
        
        apply_adaround(
            model=model,
            train_loader=train_loader,
            dataset=dataset,
            args=args,
            fp32_checkpoint=fp32_ckpt_name,
            adaround_config=quant_cfg,
            checkpoint_dir=str(checkpoint_subdir),
            save_name="sasrec_adaround.pth",
            logger=logger
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

if __name__ == "__main__":
    main()
