from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utils import gather_quantizable_layers, move_batch_to_device, replace_module
from .base import QuantStrategy


class AdaRoundConv2d(nn.Conv2d):
    """Conv2d layer wrapper implementing AdaRound soft rounding using PTQ-style scale."""

    def __init__(self, original: nn.Conv2d, bits: int = 8, per_channel: bool = True) -> None:
        super().__init__(
            in_channels=original.in_channels,
            out_channels=original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            dilation=original.dilation,
            groups=original.groups,
            bias=original.bias is not None,
            padding_mode=original.padding_mode,
        )
        self.bits = bits
        self.Q = 2 ** (bits - 1) - 1  # signed symmetric quantization
        self.per_channel = per_channel
        self.ch_axis = 0  # output channels: weight shape is [out_ch, in_ch, kH, kW]

        # Store original parameters (frozen)
        self.weight_fp = nn.Parameter(original.weight.detach().clone(), requires_grad=False)
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        # Learnable rounding parameter
        self.alpha: Optional[nn.Parameter] = None

        # Fixed scale (PTQ-style, no grad)
        self.register_buffer("s", torch.zeros(1))  # will be resized
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("alpha_init", torch.tensor(False))

        # Control hard vs soft rounding
        self.hard_round_in_eval = True

        # Conv properties
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups

        # Initialize scale immediately
        self._init_scale(self.weight_fp)

    @torch.no_grad()
    def _init_scale(self, w: Tensor) -> None:
        if self.per_channel:
            w_perm = w.transpose(0, self.ch_axis).contiguous().flatten(1)
            s = (w_perm.abs().max(dim=1).values / max(self.Q, 1)).clamp(min=1e-8)
        else:
            s = (w.abs().max() / max(self.Q, 1)).clamp(min=1e-8)
        self.s = s.detach()
        self.initialized.fill_(True)

    @torch.no_grad()
    def _init_alpha(self, w: Tensor) -> None:
        s_b = self._broadcast(w, self.s)
        y = (w / s_b).detach()
        k = torch.floor(y)
        f = (y - k).clamp(1e-6, 1 - 1e-6)  # avoid log(0) or log(inf)

        alpha = torch.log(f / (1.0 - f))
        self.alpha = nn.Parameter(alpha.to(dtype=w.dtype, device=w.device))
        self.alpha_init.fill_(True)

    def _broadcast(self, w: Tensor, t: Tensor) -> Tensor:
        if not self.per_channel:
            return t
        view = [1] * w.dim()
        view[self.ch_axis] = -1
        return t.view(view)

    def forward(self, input: Tensor) -> Tensor:
        if self.alpha is None or not bool(self.alpha_init):
            self._init_alpha(self.weight_fp)

        w = self.weight_fp
        s_b = self._broadcast(w, self.s.to(w.device, w.dtype))
        y = w / s_b
        k = torch.floor(y)

        r_soft = torch.sigmoid(self.alpha)
        if (not self.training) and self.hard_round_in_eval:
            r = (r_soft >= 0.5).to(w.dtype)
        else:
            r = r_soft

        z = k + r
        z_clamped = z.clamp(-self.Q, self.Q)

        # Straight-through estimator: gradients flow through z, use rounded value for forward
        z_rounded = z_clamped.detach().round()
        w_q = s_b * (z_rounded + (z_clamped - z_clamped.detach()))

        return F.conv2d(
            input,
            w_q,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def regularization(self, lam: float = 1e-4) -> Tensor:
        if self.alpha is None:
            return torch.tensor(0.0, device=self.weight_fp.device, dtype=self.weight_fp.dtype)
        r = torch.sigmoid(self.alpha)
        reg = (1.0 - (2.0 * r - 1.0).abs()).mean()
        # print(lam)
        return float(lam) * reg

    def set_hard_round(self, hard: bool = True) -> None:
        self.hard_round_in_eval = hard


class AdaRoundQuantStrategy(QuantStrategy):
    """Post-training quantization using AdaRound optimisation for Conv2d layers only."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.bits = config.get("bits", 8)
        self.symmetric = config.get("symmetric", True)
        if not self.symmetric:
            raise ValueError("AdaRoundQuantStrategy only supports symmetric quantization.")
        self.per_channel = config.get("per_channel", True)
        self.rounding_iters = config.get("rounding_iters", 1000)
        self.rounding_reg = config.get("rounding_reg", 1e-4)
        self.quantize_embedding = config.get("quantize_embedding", False)

        self.reference_model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None
        self.model = None

    def _wrap_module(self, name: str, module: nn.Module) -> Optional[nn.Module]:
        if isinstance(module, nn.Conv2d):
            wrapped = AdaRoundConv2d(
                original=module,
                bits=self.bits,
                per_channel=self.per_channel,
            )
            self.handles.append((name, wrapped))
            return wrapped
        return None

    def attach(self, model: nn.Module) -> nn.Module:
        self.reference_model = copy.deepcopy(model).eval()
        modules = gather_quantizable_layers(model, quantize_embedding=self.quantize_embedding)
        for name, module in modules:
            wrapped = self._wrap_module(name, module)
            if wrapped is not None:
                replace_module(model, name, wrapped)
        self.model = model
        return model

    def calibrate(self, loader) -> None:
        if self.model is None or self.reference_model is None:
            return
        # if self.reference_model is None:
        #     return
        device = next(self.model.parameters()).device
        self.device = device
        self.reference_model.to(device)
        self.model.to(device)
        self.reference_model.eval()

        adaround_modules = [
            module for _, module in self.handles
            if isinstance(module, AdaRoundConv2d)
        ]
        if not adaround_modules:
            return

        # Initialize alpha parameters for all modules
        # for module in adaround_modules:
        #     module.train()

        for module in adaround_modules:
            if module.alpha is None:
                # Force initialization by calling forward once
                dummy_input = torch.randn(1, module.in_channels, 8, 8, device=device)
                _ = module(dummy_input)
            module.train()

        # Now all modules should have alpha initialized
        optimizer = torch.optim.Adam([module.alpha for module in adaround_modules if module.alpha is not None], lr=1e-2)
        criterion = nn.MSELoss()
        iterator = iter(loader)

        for iteration in range(self.rounding_iters):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            batch = move_batch_to_device(batch, device)
            if "input" in batch:
                inputs = batch["input"]
            elif "lr" in batch:
                inputs = batch["lr"]
            elif isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                raise KeyError("Batch must contain 'input', 'lr', or be a tuple with input at index 0.")

            optimizer.zero_grad()
            with torch.no_grad():
                target = self.reference_model(inputs)
            output = self.model(inputs)
            loss = criterion(output, target)
            reg = sum(module.regularization(lam=self.rounding_reg) for module in adaround_modules)
            loss = loss + reg
            loss.backward()
            optimizer.step()

        for module in adaround_modules:
            module.eval()

    def step(self) -> None:
        # No per-step updates needed after calibration
        pass