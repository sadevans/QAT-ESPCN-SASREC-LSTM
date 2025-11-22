"""Base ESPCN (Efficient Sub-Pixel Convolutional Neural Network) model for super-resolution."""

import math
import torch
from torch import nn, Tensor


class BaseESPCN(nn.Module):
    """
    Base ESPCN model for image super-resolution.
    
    ESPCN uses sub-pixel convolution to upscale images efficiently by learning
    an array of upscaling filters and applying them in the low-resolution space.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB).
        out_channels: Number of output channels (default: 3 for RGB).
        channels: Number of feature channels in hidden layers (default: 64).
        upscale_factor: Super-resolution scale factor (default: 3).
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 64,
        upscale_factor: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.upscale_factor = upscale_factor

        hidden_channels = channels // 2
        output_channels = out_channels * (upscale_factor ** 2)

        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self.sub_pixel = nn.Sequential(
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize model weights using He initialization.
        
        For layers with 32 input channels, uses small normal initialization.
        For other layers, uses He initialization (normal with std = sqrt(2/fan_in)).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:  # как в оригинале
                    nn.init.normal_(m.weight, mean=0.0, std=0.001)
                else:
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    std = math.sqrt(2.0 / fan_in)
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the ESPCN model.
        
        Args:
            x: Input low-resolution image tensor of shape (B, C, H, W).
            
        Returns:
            High-resolution image tensor of shape (B, C, H*scale, W*scale),
            clamped to [0, 1] range.
        """
        x = self.feature_maps(x)
        x = self.sub_pixel(x)
        return torch.clamp(x, 0.0, 1.0)