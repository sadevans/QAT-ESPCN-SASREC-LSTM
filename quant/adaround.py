from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utils import (gather_quantizable_layers, move_batch_to_device,
                   replace_module)

from .base import QuantStrategy


class AdaRoundModule(nn.Module):
    """Base class for AdaRound modules to share common logic."""
    def __init__(self, bits: int = 8, per_channel: bool = True, ch_axis: int = 0):
        super().__init__()
        self.bits = bits
        self.Q = 2 ** (bits - 1) - 1
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        
        # self.alpha: Optional[nn.Parameter] = None
        self.register_buffer("s", torch.zeros(1))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("alpha_init", torch.tensor(False))
        self.register_parameter("alpha", None)
        self.hard_round_in_eval = True

    @torch.no_grad()
    def _init_scale(self, w: Tensor) -> None:
        if self.per_channel:
            # Flatten all dimensions except channel axis
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
        f = (y - k).clamp(1e-6, 1 - 1e-6)
        alpha = torch.log(f / (1.0 - f))
        self.alpha = nn.Parameter(alpha.to(dtype=w.dtype, device=w.device))
        self.alpha_init.fill_(True)

    def _broadcast(self, w: Tensor, t: Tensor) -> Tensor:
        if not self.per_channel:
            return t
        view = [1] * w.dim()
        view[self.ch_axis] = -1
        return t.view(view)

    def get_quantized_weight(self, w: Tensor) -> Tensor:
        if self.alpha is None or not bool(self.alpha_init):
            self._init_alpha(w)

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
        z_rounded = z_clamped.detach().round()
        w_q = s_b * (z_rounded + (z_clamped - z_clamped.detach()))
        return w_q

    def regularization(self, lam: float = 1e-4) -> Tensor:
        if self.alpha is None:
            return torch.tensor(0.0, device=self.s.device)
        r = torch.sigmoid(self.alpha)
        reg = (1.0 - (2.0 * r - 1.0).abs()).mean()
        return float(lam) * reg

    def set_hard_round(self, hard: bool = True) -> None:
        self.hard_round_in_eval = hard

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Ensure alpha parameter exists before loading checkpoint weights.
        """
        if "alpha" in state_dict and self.alpha is None:
            loaded_alpha = state_dict["alpha"]
            self.alpha = nn.Parameter(loaded_alpha.clone().detach())
            self.register_parameter("alpha", self.alpha)
            self.alpha_init.fill_(True)
        return super().load_state_dict(state_dict, strict)


class AdaRoundConv2d(nn.Conv2d, AdaRoundModule):
    def __init__(self, original: nn.Conv2d, bits: int = 8, per_channel: bool = True) -> None:
        nn.Conv2d.__init__(
            self,
            original.in_channels, original.out_channels, original.kernel_size,
            original.stride, original.padding, original.dilation,
            original.groups, original.bias is not None, original.padding_mode
        )
        AdaRoundModule.__init__(self, bits, per_channel, ch_axis=0)
        
        self.weight = nn.Parameter(original.weight.detach().clone(), requires_grad=False)
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.detach().clone(), requires_grad=False)
        
        self._init_scale(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        w_q = self.get_quantized_weight(self.weight)
        return F.conv2d(input, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)


class AdaRoundConv1d(nn.Conv1d, AdaRoundModule):
    def __init__(self, original: nn.Conv1d, bits: int = 8, per_channel: bool = True) -> None:
        nn.Conv1d.__init__(
            self,
            original.in_channels, original.out_channels, original.kernel_size,
            original.stride, original.padding, original.dilation,
            original.groups, original.bias is not None, original.padding_mode
        )
        AdaRoundModule.__init__(self, bits, per_channel, ch_axis=0)
        
        self.weight = nn.Parameter(original.weight.detach().clone(), requires_grad=False)
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.detach().clone(), requires_grad=False)
        
        self._init_scale(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        w_q = self.get_quantized_weight(self.weight)
        return F.conv1d(input, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)


class AdaRoundLinear(nn.Linear, AdaRoundModule):
    def __init__(self, original: nn.Linear, bits: int = 8, per_channel: bool = True) -> None:
        nn.Linear.__init__(
            self, original.in_features, original.out_features, original.bias is not None
        )
        AdaRoundModule.__init__(self, bits, per_channel, ch_axis=0)
        
        self.weight = nn.Parameter(original.weight.detach().clone(), requires_grad=False)
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.detach().clone(), requires_grad=False)
        
        self._init_scale(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        w_q = self.get_quantized_weight(self.weight)
        return F.linear(input, w_q, self.bias)


class AdaRoundQuantStrategy(QuantStrategy):
    """Post-training quantization using AdaRound optimisation."""

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
            wrapped = AdaRoundConv2d(module, self.bits, self.per_channel)
            self.handles.append((name, wrapped))
            return wrapped
        if isinstance(module, nn.Conv1d):
            wrapped = AdaRoundConv1d(module, self.bits, self.per_channel)
            self.handles.append((name, wrapped))
            return wrapped
        if isinstance(module, nn.Linear):
            wrapped = AdaRoundLinear(module, self.bits, self.per_channel)
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
        
        device = next(self.model.parameters()).device
        self.device = device
        self.reference_model.to(device)
        self.model.to(device)
        self.reference_model.eval()

        adaround_modules = [
            module for _, module in self.handles
            if isinstance(module, AdaRoundModule)
        ]
        if not adaround_modules:
            return

        # Init alpha
        for module in adaround_modules:
            module.train()
            if module.alpha is None:
                 module._init_alpha(module.weight)

        optimizer = torch.optim.Adam([m.alpha for m in adaround_modules], lr=1e-2)
        criterion = nn.MSELoss()
        
        logger = getattr(self, "logger", None)
        log_interval = int(self.config.get("log_interval", 100))

        iterator = iter(loader)
        for iteration in range(self.rounding_iters):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            batch = _move_batch_like(batch, device)
            
            if isinstance(batch, dict):
                if "input" in batch:
                    inputs = batch["input"]
                elif "lr" in batch:
                    inputs = batch["lr"]
                else:


                    raise KeyError("Batch dict must contain 'input' or 'lr'")
            else:
                if isinstance(batch, (list, tuple)):
                    inputs = tuple(batch)
                else:
                    inputs = (batch,)

            optimizer.zero_grad()
            
            with torch.no_grad():
                target = self.reference_model(*inputs)
            output = self.model(*inputs)
            
            if isinstance(target, (tuple, list)):
                mse_loss = 0.0
                for t, o in zip(target, output):
                    mse_loss += criterion(o, t)
            else:
                mse_loss = criterion(output, target)

            reg = sum(module.regularization(lam=self.rounding_reg) for module in adaround_modules)
            loss = mse_loss + reg
            loss.backward()
            optimizer.step()

            # Optional logging of calibration metrics
            if logger is not None and (iteration % log_interval == 0 or iteration == self.rounding_iters - 1):
                try:
                    logger.report_scalar("AdaRound/mse_loss", "calibration", value=float(mse_loss.item()), iteration=iteration)
                    logger.report_scalar("AdaRound/reg", "calibration", value=float(reg.item() if torch.is_tensor(reg) else reg), iteration=iteration)
                    logger.report_scalar("AdaRound/total_loss", "calibration", value=float(loss.item()), iteration=iteration)
                except Exception:
                    # Logging should never break calibration
                    pass

        for module in adaround_modules:
            module.eval()
            module.set_hard_round(True)

    def step(self) -> None:
        pass


def _move_batch_like(batch, device):
    if isinstance(batch, dict):
        return {k: _move_batch_like(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_batch_like(v, device) for v in batch)
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch
