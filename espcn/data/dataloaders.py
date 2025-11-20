import torch
from torch.utils.data import DataLoader
from .datasets import DIV2KTrainDataset, SRBenchmarkDataset
import random
from pathlib import Path


def get_train_loader(config):
    ds = DIV2KTrainDataset(
        hr_dir=config['data']['train_dir'],
        patch_size=config['data']['patch_size'],
        upscale_factor=config['model']['upscale_factor'],
        rgb_range=config['data']['rgb_range']
    )
    return DataLoader(
        ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        drop_last=True,
    )


def get_val_loaders(config):
    """Returns list of validation DataLoaders (one per val dir)."""
    datasets = [
        SRBenchmarkDataset(
            hr_dir=Path(dir_path),
            upscale_factor=config['model']['upscale_factor'],
            rgb_range=config['data']['rgb_range']
        )
        for dir_path in config['data']['val_dirs']
    ]
    return [
        DataLoader(
            ds,
            batch_size=1,  # full image
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory'],
        )
        for ds in datasets
    ]


def create_dataloaders(config, seed=None):
    """Compatibility wrapper."""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    train_loader = get_train_loader(config)
    val_loaders = get_val_loaders(config)
    
    val_loader = val_loaders[0] if val_loaders else None
    test_loader = val_loaders[1] if len(val_loaders) > 1 else val_loader

    return train_loader, val_loader, test_loader