import math
import torch
from torch import nn, Tensor

class BaseESPCN(nn.Module):
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

    def _init_weights(self):
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
        x = self.feature_maps(x)
        x = self.sub_pixel(x)
        return torch.clamp(x, 0.0, 1.0)