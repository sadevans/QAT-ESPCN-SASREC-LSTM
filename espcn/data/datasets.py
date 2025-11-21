import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class DIV2KTrainDataset(Dataset):
    """Training dataset with random cropping and online LR generation."""

    def __init__(self, hr_dir, patch_size=96, upscale_factor=4, rgb_range=1.0):
        hr_dir = Path(hr_dir)
        if not hr_dir.exists():
            raise FileNotFoundError(f"HR directory does not exist: {hr_dir}")

        self.hr_files = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg")
            for p in hr_dir.rglob(ext)
        )
        if not self.hr_files:
            raise FileNotFoundError(f"No HR images found in {hr_dir}")

        if patch_size % upscale_factor != 0:
            raise ValueError(f"patch_size ({patch_size}) must be divisible by scale ({upscale_factor})")

        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.rgb_range = rgb_range
        self.lr_size = patch_size // upscale_factor
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        w, h = hr.size

        # Upsample small images
        if w < self.patch_size or h < self.patch_size:
            scale = max(self.patch_size / w, self.patch_size / h) * 1.1
            new_w, new_h = int(w * scale), int(h * scale)
            hr = hr.resize((new_w, new_h), Image.BICUBIC)
            w, h = new_w, new_h

        # Random crop
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        hr = hr.crop((x, y, x + self.patch_size, y + self.patch_size))

        hr_tensor = self.to_tensor(hr) * self.rgb_range
        hr_tensor = hr_tensor.clamp(0.0, self.rgb_range)

        # Generate LR online via bicubic downsampling
        lr_tensor = torch.nn.functional.interpolate(
            hr_tensor.unsqueeze(0),
            size=(self.lr_size, self.lr_size),
            mode='bicubic',
            align_corners=False,
            antialias=True  # <-- рекомендуется в современных PyTorch
        ).squeeze(0).clamp(0.0, self.rgb_range)

        return {"lr": lr_tensor, "hr": hr_tensor}


class SRBenchmarkDataset(Dataset):
    """Validation/Test dataset: full-image inference, LR generated online from HR."""

    def __init__(self, hr_dir, upscale_factor=4, rgb_range=1.0):
        hr_dir = Path(hr_dir)
        if not hr_dir.exists():
            raise FileNotFoundError(f"HR directory does not exist: {hr_dir}")

        # Поддерживаем любые изображения как HR (имена не важны)
        self.hr_files = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg")
            for p in hr_dir.rglob(ext)
        )
        if not self.hr_files:
            raise FileNotFoundError(f"No HR images found in {hr_dir}")

        self.upscale_factor = upscale_factor
        self.rgb_range = rgb_range
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        hr = Image.open(hr_path).convert("RGB")
        name = hr_path.stem

        hr_tensor = self.to_tensor(hr) * self.rgb_range
        _, h_hr, w_hr = hr_tensor.shape

        h_lr = h_hr // self.upscale_factor
        w_lr = w_hr // self.upscale_factor

        hr_tensor = hr_tensor[:, :h_lr * self.upscale_factor, :w_lr * self.upscale_factor]

        lr_tensor = torch.nn.functional.interpolate(
            hr_tensor.unsqueeze(0),
            size=(h_lr, w_lr),
            mode='bicubic',
            align_corners=False,
            antialias=True
        ).squeeze(0).clamp(0.0, self.rgb_range)

        return {"lr": lr_tensor, "hr": hr_tensor, "name": name}