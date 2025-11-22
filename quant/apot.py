from __future__ import annotations

import math
from typing import Any
import torch
from torch import Tensor, nn

from .base import FakeQuantizer, QATQuantStrategy, UniformAffineQuantizer


class APoTQuantizer(FakeQuantizer):
    """Additive Power-of-Two (APoT) Quantizer as described in ICLR 2020.

    Parameters
    ----------
    bits : int
        Total number of bits for quantization.
    m : int
        Number of base groups (must satisfy m * k == bits).
    k : int
        Number of bits per group (must satisfy m * k == bits).
    symmetric : bool, default True
        Whether to use symmetric quantization (required for weights).
    """

    def __init__(
        self,
        bits: int,
        m: int,
        k: int,
        symmetric: bool = True,
    ) -> None:
        if m * k != bits:
            raise ValueError(f"m * k must equal bits. Got m={m}, k={k}, bits={bits}")
        if not symmetric:
            raise ValueError("APoT is designed for symmetric quantization only.")
        super().__init__(bits=bits, symmetric=symmetric, per_channel=False, channel_axis=0)

        self.m = m
        self.k = k
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
        self.levels = self._build_levels()

    def _build_levels(self) -> Tensor:
        """Build quantization levels using APoT additive decomposition."""
        levels_set = {0.0}
        total_bits = self.bits
        # Precompute all possible combinations via bitmask
        for code in range(1, 2**total_bits):
            value = 0.0
            for i in range(total_bits):
                if code & (1 << i):
                    # Group index and position within group
                    group = i // self.k
                    pos_in_group = i % self.k
                    exponent = -(group * self.k + pos_in_group + 1)
                    value += 2.0 ** exponent
            levels_set.add(value)
        levels = sorted(levels_set)
        return torch.tensor(levels, dtype=torch.float32)  # shape: [L]

    def initialize_from_tensor(self, tensor: Tensor) -> None:
        if bool(self.initialized):
            return
        max_val = tensor.detach().abs().max()
        if max_val <= 0:
            max_val = torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype)
        max_level = self.levels.max()
        scale = max_val / max_level.clamp(min=1e-8)
        self.scale.fill_(float(scale))
        self.initialized.fill_(True)

    def _forward_impl(self, tensor: Tensor) -> Tensor:
        device = tensor.device
        levels = self.levels.to(device, dtype=tensor.dtype)
        scale = self.scale.to(device, dtype=tensor.dtype).clamp(min=1e-8)

        # Normalize by scale
        normalized = tensor / scale

        # Quantize by finding closest level
        abs_norm = normalized.abs()
        # Shape: [*tensor.shape, L]
        diff = (abs_norm.unsqueeze(-1) - levels).abs()
        closest_idx = diff.argmin(dim=-1)
        quantized_abs = levels[closest_idx]
        quantized = quantized_abs * normalized.sign()

        # Apply STE
        quantized_ste = normalized + (quantized - normalized).detach()
        dequant = quantized_ste * scale
        return dequant


class APoTQuantStrategy(QATQuantStrategy):
    """Strategy using APoT quantizers for weights."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.apot_m = config.get("apot_m", 2)
        self.apot_k = config.get("apot_k", 2)
        if self.apot_m * self.apot_k != self.bits:
            raise ValueError(
                f"APoT requires m * k == bits. Got m={self.apot_m}, k={self.apot_k}, bits={self.bits}"
            )

    def create_weight_quantizer(self, module_name: str, module: nn.Module) -> FakeQuantizer:
        return APoTQuantizer(
            bits=self.bits,
            m=self.apot_m,
            k=self.apot_k,
            symmetric=True,  # weights are symmetric
        )

    def create_activation_quantizer(self, module_name: str, module: nn.Module) -> UniformAffineQuantizer:
        return UniformAffineQuantizer(
            bits=self.activation_bits,
            symmetric=False,
            per_channel=False,
            channel_axis=-1,
        )
