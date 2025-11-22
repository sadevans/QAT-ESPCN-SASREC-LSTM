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
    model_cfg = config["model"].copy()
    return QuantESPCN(**model_cfg)


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8)).item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
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


def _normalize_image_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu()
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = torch.zeros_like(x)
    return x


def log_images(logger, step, prefix, lr, sr, hr):
    if logger is None:
        return

    lr_img = _normalize_image_tensor(lr[0]).permute(1, 2, 0).numpy()
    sr_img = _normalize_image_tensor(sr[0]).permute(1, 2, 0).numpy()
    hr_img = _normalize_image_tensor(hr[0]).permute(1, 2, 0).numpy()

    logger.report_image(f"{prefix} Images", "LR", iteration=step, image=lr_img)
    logger.report_image(f"{prefix} Images", "Predicted (SR)", iteration=step, image=sr_img)
    logger.report_image(f"{prefix} Images", "Ground Truth (HR)", iteration=step, image=hr_img)


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    logger=None,
    epoch=0
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

        psnr = calculate_psnr(sr, hr)

        total_loss += loss.item()
        total_psnr += psnr
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches

    if logger:
        logger.report_scalar("Loss", "Train", value=avg_loss, iteration=epoch)
        logger.report_scalar("PSNR", "Train", value=avg_psnr, iteration=epoch)

    return avg_loss, avg_psnr


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    criterion: nn.Module = None,
    logger=None,
    epoch=0
) -> Tuple[float, float, float]:
    """Validate model and return average PSNR and SSIM."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    log_image_batch = None

    for batch in val_loader:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        sr = model(lr)
        
        if log_image_batch is None:
            log_image_batch = (lr, sr, hr)

        if criterion:
            loss = criterion(sr, hr)
            total_loss += loss.item()

        psnr = calculate_psnr(sr, hr)
        total_psnr += psnr

        ssim_val = calculate_ssim(sr, hr)
        total_ssim += ssim_val

        num_batches += 1

    avg_loss = total_loss / num_batches if criterion else 0.0
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    if logger:
        if criterion:
            logger.report_scalar("Loss", "Val", value=avg_loss, iteration=epoch)
        logger.report_scalar("PSNR", "Val", value=avg_psnr, iteration=epoch)
        logger.report_scalar("SSIM", "Val", value=avg_ssim, iteration=epoch)
        
        if log_image_batch:
            log_images(logger, epoch, "Val", *log_image_batch)

    return avg_loss, avg_psnr, avg_ssim


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

    ckpt = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
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
    save_name: str,
    logger=None
):
    """Train FP32 model."""
    print("Starting FP32 training...")
    
    optimizer = Adam(
        model.parameters(),
        lr=config["optimization"]["lr"],
        weight_decay=config["optimization"].get("weight_decay", 0.0),
        betas=tuple(config["optimization"].get("betas", (0.9, 0.999))),
    )

    best_psnr = 0.0

    for epoch in range(epochs):
        train_loss, train_psnr = train_epoch(
            model, train_loader, optimizer, criterion, device, logger=logger, epoch=epoch+1
        )

        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, device, criterion=criterion, logger=logger, epoch=epoch+1
        )

        print(f"[FP32] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train PSNR: {train_psnr:.2f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, save_name)
            print(f"Saved best FP32 checkpoint (PSNR: {val_psnr:.2f} dB)")

    print(f"FP32 training done. Best PSNR: {best_psnr:.2f} dB")


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
    save_name: str,
    logger=None
):
    """Train model with quantization-aware training (QAT)."""
    print(f"Starting QAT with {strategy_name.upper()}...")

    model.prepare_quant(strategy_name, quant_config)
    
    model.train()

    with torch.no_grad():
        for batch in train_loader:
            lr = batch["lr"].to(device)
            _ = model(lr)
            break

    optimizer = Adam(
        model.parameters(),
        lr=config["optimization"]["lr"],
        weight_decay=config["optimization"].get("weight_decay", 0.0),
        betas=tuple(config["optimization"].get("betas", (0.9, 0.999))),
    )

    best_psnr = 0.0

    for epoch in range(epochs):
        train_loss, train_psnr = train_epoch(
            model, train_loader, optimizer, criterion, device, logger=logger, epoch=epoch+1
        )

        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, device, criterion=criterion, logger=logger, epoch=epoch+1
        )

        print(f"[QAT {strategy_name}] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train PSNR: {train_psnr:.2f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, save_name)
            print(f"Saved best QAT checkpoint (PSNR: {val_psnr:.2f} dB)")

    print(f"QAT ({strategy_name}) done. Best PSNR: {best_psnr:.2f} dB")


def apply_adaround(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    fp32_checkpoint: str,
    adaround_config: Dict[str, Any],
    checkpoint_dir: str,
    save_name: str,
    logger=None
):
    """Apply AdaRound post-training quantization."""
    print("Starting AdaRound PTQ...")

    fp32_path = os.path.join(checkpoint_dir, fp32_checkpoint)
    if not os.path.exists(fp32_path):
        raise FileNotFoundError(f"FP32 checkpoint not found: {fp32_path}")

    print(f"Loading FP32 checkpoint: {fp32_checkpoint}")
    fp32_state = torch.load(fp32_path, map_location=device, weights_only=False)["model_state_dict"]
    model.load_state_dict(fp32_state, strict=True)

    model.prepare_quant("adaround", adaround_config)

    if logger is not None and hasattr(model, "quant_strategy") and hasattr(model.quant_strategy, "set_logger"):
        model.quant_strategy.set_logger(logger)

    print("Running AdaRound calibration (rounding optimization)...")
    model.calibrate(train_loader)

    print("Validating AdaRound model...")
    val_loss, psnr, ssim = validate(model, val_loader, device, criterion=nn.L1Loss(), logger=logger, epoch=1)
    print(f"[AdaRound] Final PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "strategy": "adaround",
        "config": adaround_config,
    }, os.path.join(checkpoint_dir, save_name))

    print(f"AdaRound done. Model saved as {save_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ESPCN with quantization for super-resolution.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    configure_logging()
    config_path = Path(args.config)
    config_stem = config_path.stem
    config = load_config(args.config)
    set_random_seeds(config["experiment"].get("seed", 42))

    device = torch.device(config["experiment"].get("device", "cuda") if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = create_dataloaders(config, seed=config["experiment"].get("seed", 42))

    model = build_model(config).to(device)

    quant_cfg = config.get("quantization", {})
    strategy_name = quant_cfg.get("method", "fp32").lower()

    if strategy_name in ("none", "fp32"):
        strategy_name = "fp32"

    checkpoint_dir = Path("/netapp/a.gorokhova/itmo/QAT-ESPCN-SASREC/checkpoints/espcn_run")
    run_name = config["experiment"].get("run_name", "espcn_run")
    config["experiment"]["run_name"] = run_name
    config.setdefault("paths", {})["checkpoints_dir"] = str(checkpoint_dir)

    logging_cfg = config.get("logging", {})
    clearml_task = init_clearml_task(logging_cfg, config)
    logger = clearml_task.get_logger() if clearml_task else None

    criterion = nn.L1Loss()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if strategy_name == "fp32":
        train_fp32(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            criterion=criterion,
            device=device,
            epochs=config["training"]["epochs"],
            checkpoint_dir=str(checkpoint_dir),
            save_name="espcn_fp32.pth",
            logger=logger
        )

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
            checkpoint_dir=str(checkpoint_dir),
            save_name=f"espcn_{strategy_name}.pth",
            logger=logger
        )

    elif strategy_name == "adaround":
        fp32_epochs = config["training"].get("fp32_epochs", 0)
        fp32_ckpt = quant_cfg.get("base_checkpoint", None)

        if fp32_epochs > 0:
            train_fp32(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                criterion=criterion,
                device=device,
                epochs=fp32_epochs,
                checkpoint_dir=str(checkpoint_dir),
                save_name="espcn_fp32_for_adaround.pth",
                logger=logger
            )
            fp32_ckpt_name = "espcn_fp32_for_adaround.pth"
        elif fp32_ckpt:
            fp32_ckpt_path = Path(fp32_ckpt)
            if not fp32_ckpt_path.exists():
                raise FileNotFoundError(f"Base checkpoint not found: {fp32_ckpt}")
            fp32_ckpt_name = fp32_ckpt_path.name
        else:
            raise ValueError("AdaRound requires either fp32_epochs > 0 or base_checkpoint.")

        apply_adaround(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            fp32_checkpoint=fp32_ckpt_name,
            adaround_config=quant_cfg,
            checkpoint_dir=str(checkpoint_dir),
            save_name="espcn_adaround.pth",
            logger=logger
        )

    else:
        raise ValueError(f"Unknown quantization method: {strategy_name}")

    val_loss, test_psnr, test_ssim = validate(
        model, test_loader, device, criterion=criterion, logger=None, epoch=999
    )
    if isinstance(test_psnr, torch.Tensor):
        test_psnr = test_psnr.item()
    if isinstance(test_ssim, torch.Tensor):
        test_ssim = test_ssim.item()
    
    test_metrics = {"psnr": test_psnr, "ssim": test_ssim}
    print("Final test metrics:", json.dumps(test_metrics, indent=2))

    config_path = config.get("_metadata", {}).get("loaded_from")
    if config_path:
        config_stem = Path(config_path).stem
    else:
        config_stem = run_name

    results_root = Path("results/espcn_results")
    results_root.mkdir(parents=True, exist_ok=True)
    results_path = results_root / f"{config_stem}_results.json"

    with open(results_path, "w") as f:
        json.dump(
            {
                "run_name": run_name,
                "strategy": strategy_name,
                "test_metrics": test_metrics,
                "config": config,
            },
            f,
            indent=2,
        )

    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
