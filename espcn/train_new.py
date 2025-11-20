"""Simplified training script for ESPCN models (FP32 and quantization-aware training)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from espcn.data.dataloaders import create_dataloaders
from espcn.model.quant import QuantESPCN
from utils import (
    configure_logging,
    load_config,
    set_random_seeds,
    init_clearml_task
)


def build_model(config: Dict[str, Any]) -> QuantESPCN:
    """Build QuantESPCN model (supports FP32 and all quantization modes)."""
    model_cfg = config["model"].copy()
    return QuantESPCN(**model_cfg)


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate PSNR between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate SSIM between two images."""
    C1 = (0.01 * 1.0) ** 2  # max_val = 1.0
    C2 = (0.03 * 1.0) ** 2

    mu1 = torch.mean(img1, dim=[-2, -1], keepdim=True)
    mu2 = torch.mean(img2, dim=[-2, -1], keepdim=True)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.mean((img1 - mu1) ** 2, dim=[-2, -1], keepdim=True)
    sigma2_sq = torch.mean((img2 - mu2) ** 2, dim=[-2, -1], keepdim=True)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2), dim=[-2, -1], keepdim=True)

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return torch.mean(numerator / denominator).item()


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Train"):
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)

        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()

        # Calculate PSNR for training
        psnr = calculate_psnr(sr, hr)

        total_loss += loss.item()
        total_psnr += psnr
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches

    return avg_loss, avg_psnr


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate model and return average PSNR and SSIM."""
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    for batch in val_loader:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        sr = model(lr)

        # Calculate PSNR
        psnr = calculate_psnr(sr, hr)
        total_psnr += psnr

        # Calculate SSIM
        ssim_val = calculate_ssim(sr, hr)
        total_ssim += ssim_val

        num_batches += 1

    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches

    return avg_psnr, avg_ssim


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_dir: str,
    filename: str,
):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    filename: str,
) -> int:
    """Load model checkpoint and return epoch number."""
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=next(model.parameters()).device)
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0)


def train_fp32(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    checkpoint_dir: str,
    save_name: str = "espcn_fp32.pth",
):
    """Train FP32 model."""
    print("🚀 Starting FP32 training...")
    
    # Setup optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config["optimization"]["lr"],
        weight_decay=config["optimization"].get("weight_decay", 0.0),
        betas=tuple(config["optimization"].get("betas", (0.9, 0.999))),
    )

    best_psnr = 0.0

    for epoch in range(epochs):
        # Train one epoch
        train_loss, train_psnr = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_psnr, val_ssim = validate(model, val_loader, device)

        print(f"[FP32] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f}")

        # Save best checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, save_name)
            print(f"✅ Saved best FP32 checkpoint (PSNR: {val_psnr:.2f} dB)")

    print(f"✅ FP32 training done. Best PSNR: {best_psnr:.2f} dB")


def train_qat(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    criterion: nn.Module,
    device: torch.device,
    strategy_name: str,
    quant_config: Dict[str, Any],
    epochs: int,
    checkpoint_dir: str,
    save_name: str = "espcn_qat.pth",
):
    """Train model with quantization-aware training (QAT)."""
    print(f"🚀 Starting QAT with {strategy_name.upper()}...")

    # Apply quantization strategy
    model.prepare_quant(strategy_name, quant_config)
    
    # Run dummy forward pass to initialize lazy modules (e.g. LSQ activations)
    print("🔧 Running dummy pass to initialize quantization parameters...")
    model.train()
    with torch.no_grad():
        # Fetch one batch
        for batch in train_loader:
            lr = batch["lr"].to(device)
            model(lr)
            break
            
    # Setup optimizer (AFTER quantization parameters are initialized)
    optimizer = Adam(
        model.parameters(),
        lr=config["optimization"]["lr"],
        weight_decay=config["optimization"].get("weight_decay", 0.0),
        betas=tuple(config["optimization"].get("betas", (0.9, 0.999))),
    )

    best_psnr = 0.0

    for epoch in range(epochs):
        # Train one epoch
        train_loss, train_psnr = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_psnr, val_ssim = validate(model, val_loader, device)

        print(f"[QAT {strategy_name}] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f}")

        # Save best checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, save_name)
            print(f"✅ Saved best QAT checkpoint (PSNR: {val_psnr:.2f} dB)")

    print(f"✅ QAT ({strategy_name}) done. Best PSNR: {best_psnr:.2f} dB")


def apply_adaround(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    fp32_checkpoint: str,
    adaround_config: Dict[str, Any],
    checkpoint_dir: str,
    save_name: str = "espcn_adaround.pth",
):
    """Apply AdaRound post-training quantization."""
    print("🚀 Starting AdaRound PTQ...")

    # Load FP32 checkpoint
    fp32_path = os.path.join(checkpoint_dir, fp32_checkpoint)
    if not os.path.exists(fp32_path):
        raise FileNotFoundError(f"FP32 checkpoint not found: {fp32_path}")

    print(f"📦 Loading FP32 checkpoint: {fp32_checkpoint}")
    fp32_state = torch.load(fp32_path, map_location=device)["model_state_dict"]
    model.load_state_dict(fp32_state, strict=True)

    # Apply AdaRound
    model.prepare_quant("adaround", adaround_config)

    # Run calibration
    print("🔧 Running AdaRound calibration (rounding optimization)...")
    model.calibrate(train_loader)

    # Final validation
    print("🔍 Validating AdaRound model...")
    psnr, ssim = validate(model, val_loader, device)
    print(f"[AdaRound] Final PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

    # Save quantized model
    torch.save({
        "model_state_dict": model.state_dict(),
        "strategy": "adaround",
        "config": adaround_config,
    }, os.path.join(checkpoint_dir, save_name))

    print(f"✅ AdaRound done. Model saved as {save_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ESPCN with quantization for super-resolution.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    set_random_seeds(config["experiment"].get("seed", 42))

    print(config["experiment"])
    device = torch.device(config["experiment"].get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(config, seed=config["experiment"].get("seed", 42))

    # Model
    model = build_model(config).to(device)

    # Determine quantization strategy
    quant_cfg = config.get("quantization", {})
    strategy_name = quant_cfg.get("method", "fp32").lower()

    # Handle different FP32 method names
    if strategy_name in ("none", "fp32"):
        strategy_name = "fp32"

    checkpoint_dir = Path(config.get("paths", {}).get("checkpoints_dir", "./checkpoints"))
    results_dir = Path(config.get("paths", {}).get("results_dir", "./results"))
    run_name = config["experiment"].get("run_name", "espcn_run")

    # Initialize ClearML logging (disabled for simplicity)
    # logging_cfg = config.get("logging", {})
    # clearml_task = init_clearml_task(logging_cfg, config)

    # Setup optimizer (Moved inside training functions)
    # optimizer = Adam(...)
    criterion = nn.L1Loss()

    checkpoint_subdir = checkpoint_dir / run_name
    checkpoint_subdir.mkdir(parents=True, exist_ok=True)

    # --- FP32 Training ---
    if strategy_name == "fp32":
        train_fp32(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            criterion=criterion,
            device=device,
            epochs=config["training"]["epochs"],
            checkpoint_dir=str(checkpoint_subdir),
            save_name="espcn_fp32.pth"
        )

    # --- QAT (LSQ, APoT, QDrop) ---
    elif strategy_name in ("lsq", "apot", "qdrop"):
        train_qat(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            criterion=criterion,
            device=device,
            strategy_name=strategy_name,
            quant_config=quant_cfg,
            epochs=config["training"]["epochs"],
            checkpoint_dir=str(checkpoint_subdir),
            save_name=f"espcn_{strategy_name}.pth"
        )

    # --- AdaRound (PTQ) ---
    elif strategy_name == "adaround":
        # Step 1: Train or load FP32 model
        fp32_epochs = config["training"].get("fp32_epochs", 0)
        fp32_ckpt = quant_cfg.get("base_checkpoint", None)

        if fp32_epochs > 0:
            # Train FP32 from scratch
            train_fp32(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                criterion=criterion,
                device=device,
                epochs=fp32_epochs,
                checkpoint_dir=str(checkpoint_subdir),
                save_name="espcn_fp32_for_adaround.pth"
            )
            fp32_ckpt_name = "espcn_fp32_for_adaround.pth"
        elif fp32_ckpt:
            # Use provided FP32 checkpoint
            fp32_ckpt_path = Path(fp32_ckpt)
            if not fp32_ckpt_path.exists():
                raise FileNotFoundError(f"Base checkpoint not found: {fp32_ckpt}")
            fp32_ckpt_name = fp32_ckpt_path.name
        else:
            raise ValueError("AdaRound requires either fp32_epochs > 0 or base_checkpoint.")

        # Step 2: Apply AdaRound
        apply_adaround(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            fp32_checkpoint=fp32_ckpt_name,
            adaround_config=quant_cfg,
            checkpoint_dir=str(checkpoint_subdir),
            save_name="espcn_adaround.pth"
        )

    else:
        raise ValueError(f"Unknown quantization method: {strategy_name}")

    # --- Final evaluation on test set ---
    test_psnr, test_ssim = validate(model, test_loader, device)
    test_metrics = {"psnr": test_psnr, "ssim": test_ssim}
    print("✅ Final test metrics:", json.dumps(test_metrics, indent=2))

    # Save final results
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{run_name}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "strategy": strategy_name,
            "test_metrics": test_metrics,
            "config": config,
        }, f, indent=2)

    print(f"💾 Results saved to: {results_path}")


if __name__ == "__main__":
    main()
