"""DataLoader factory functions for ESPCN training and evaluation."""

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .datasets import DIV2KTrainDataset, SRBenchmarkDataset


def get_train_loader(config):
    """
    Create DataLoader for training dataset.
    
    Args:
        config: Configuration dictionary with keys:
            - 'data': Contains 'train_dir', 'patch_size', 'rgb_range'.
            - 'model': Contains 'upscale_factor'.
            - 'training': Contains 'batch_size', 'num_workers', 'pin_memory'.
            
    Returns:
        DataLoader for training dataset with shuffled batches.
    """
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
    """
    Create list of validation DataLoaders (one per validation directory).
    
    Args:
        config: Configuration dictionary with keys:
            - 'data': Contains 'val_dirs' (list of paths) and 'rgb_range'.
            - 'model': Contains 'upscale_factor'.
            - 'training': Contains 'num_workers', 'pin_memory'.
            
    Returns:
        List of DataLoaders, one for each validation directory.
    """
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
    """
    Create training, validation, and test DataLoaders.
    
    Args:
        config: Configuration dictionary with data and training settings.
        seed: Optional random seed for reproducibility.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader):
            - train_loader: DataLoader for training.
            - val_loader: DataLoader for validation (first validation directory).
            - test_loader: DataLoader for testing (second validation directory if exists,
                         otherwise same as val_loader).
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    train_loader = get_train_loader(config)
    val_loaders = get_val_loaders(config)
    
    val_loader = val_loaders[0] if val_loaders else None
    test_loader = val_loaders[1] if len(val_loaders) > 1 else val_loader

    return train_loader, val_loader, test_loader