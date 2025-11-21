from __future__ import annotations

import abc
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utils import gather_quantizable_layers, replace_module


class FakeQuantizer(nn.Module, metaclass=abc.ABCMeta):
    """Abstract fake-quantizer to share between strategies."""

    def __init__(
        self,
        bits: int,
        symmetric: bool = True,
        per_channel: bool = False,
        channel_axis: int = 0,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.qmin, self.qmax = self._quant_bounds()
        self.register_buffer("initialized", torch.tensor(False), persistent=False)

    def _quant_bounds(self) -> Tuple[int, int]:
        if self.symmetric:
            qmax = 2 ** (self.bits - 1) - 1
            return -qmax - 1, qmax
        return 0, 2**self.bits - 1

    def initialize_from_tensor(self, tensor: Tensor) -> None:
        """Optional hook for one-off initialization on the first batch."""

    @abc.abstractmethod
    def _forward_impl(self, tensor: Tensor) -> Tensor:
        """Perform fake quantization."""

    def forward(self, tensor: Tensor) -> Tensor:
        if not bool(self.initialized):
            self.initialize_from_tensor(tensor.detach())
            self.initialized.fill_(True)
        return self._forward_impl(tensor)


class UniformAffineQuantizer(FakeQuantizer):
    """Uniform affine fake quantization with running min/max observers."""

    def __init__(
        self,
        bits: int,
        symmetric: bool,
        per_channel: bool,
        channel_axis: int = 0,
        momentum: float = 0.95,
    ) -> None:
        super().__init__(bits, symmetric=symmetric, per_channel=per_channel, channel_axis=channel_axis)
        shape = (1,)
        self.momentum = momentum
        self.register_buffer("running_min", torch.zeros(shape))
        self.register_buffer("running_max", torch.zeros(shape))

    def initialize_from_tensor(self, tensor: Tensor) -> None:
        dims = list(range(tensor.ndim))
        if self.per_channel:
            axis = self.channel_axis if self.channel_axis >= 0 else tensor.ndim + self.channel_axis
            dims.pop(axis)
        reduce_dims = tuple(dims)
        min_val = tensor.amin(dim=reduce_dims, keepdim=self.per_channel) if reduce_dims else tensor
        max_val = tensor.amax(dim=reduce_dims, keepdim=self.per_channel) if reduce_dims else tensor
        self.running_min.resize_as_(min_val).copy_(min_val.detach())
        self.running_max.resize_as_(max_val).copy_(max_val.detach())

    def update_ranges(self, tensor: Tensor) -> None:
        dims = list(range(tensor.ndim))
        if self.per_channel:
            axis = self.channel_axis if self.channel_axis >= 0 else tensor.ndim + self.channel_axis
            dims.pop(axis)
        reduce_dims = tuple(dims)
        current_min = tensor.amin(dim=reduce_dims, keepdim=self.per_channel) if reduce_dims else tensor
        current_max = tensor.amax(dim=reduce_dims, keepdim=self.per_channel) if reduce_dims else tensor
        self.running_min.mul_(self.momentum).add_(current_min * (1 - self.momentum))
        self.running_max.mul_(self.momentum).add_(current_max * (1 - self.momentum))

    def _calc_qparams(self) -> Tuple[Tensor, Tensor]:
        min_val = self.running_min
        max_val = self.running_max
        if self.symmetric:
            max_val = torch.max(max_val.abs(), min_val.abs())
            min_val = -max_val
        scale = (max_val - min_val) / float(self.qmax - self.qmin)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = self.qmin - torch.round(min_val / scale)
        zero_point = torch.clamp(zero_point, self.qmin, self.qmax)
        device = min_val.device
        return scale.to(device), zero_point.to(device)

    def _forward_impl(self, tensor: Tensor) -> Tensor:
        self.update_ranges(tensor.detach())
        scale, zero_point = self._calc_qparams()
        value = tensor / scale + zero_point
        q = torch.clamp(torch.round(value), self.qmin, self.qmax)
        value_q = value + (q - value).detach()
        dequant = (value_q - zero_point) * scale
        return dequant

class LearnableStepSizeQuantizer(FakeQuantizer):
    def __init__(
        self,
        bits: int,
        per_channel: bool = False,
        symmetric: bool = True,
        channel_axis: int = 0,
        alpha_init: float = 6.0,
    ) -> None:
        super().__init__(bits, symmetric=symmetric, per_channel=per_channel, channel_axis=channel_axis)
        self.alpha_init = alpha_init
        self.scale: Optional[nn.Parameter] = None

    def init_scale_param(self, num_channels: int = 1) -> None:
        """Инициализирует параметр scale с правильной формой. Вызывается один раз."""
        if isinstance(self.scale, nn.Parameter):
            return
        if hasattr(self, "scale"):
            try:
                delattr(self, "scale")
            except Exception:
                pass
        shape = (num_channels,) if self.per_channel else (1,)
        p = nn.Parameter(torch.ones(shape, dtype=torch.float32))
        self.register_parameter("scale", p)
        self.scale = p

    def initialize_from_tensor(self, tensor: Tensor) -> None:
        """Выполняет инициализацию значения scale по статистике тензора."""
        if bool(self.initialized):
            return
        if self.scale is None:
            raise RuntimeError("scale not initialized. Call init_scale_param first.")

        axis = self.channel_axis if self.channel_axis >= 0 else tensor.ndim + self.channel_axis
        reduce_dims = tuple(i for i in range(tensor.ndim) if i != axis) if self.per_channel else tuple(range(tensor.ndim))

        mean_abs = tensor.detach().abs()
        if reduce_dims:
            mean_abs = mean_abs.mean(dim=reduce_dims, keepdim=False)
        else:
            mean_abs = mean_abs.mean()

        # s = α * mean(|x|) / sqrt(Q_max)
        scale_val = self.alpha_init * mean_abs / (self.qmax ** 0.5 + 1e-6)
        scale_val = torch.clamp(scale_val, min=1e-6)

        if self.per_channel:
            # scale_val shape должен совпадать с self.scale
            self.scale.data.copy_(scale_val.detach().to(self.scale.dtype))
        else:
            self.scale.data.fill_(float(scale_val.item()))

        self.initialized.fill_(True)

        if not hasattr(self.scale, "_lsq_hook"):
            def grad_scale_hook(grad):
                # используем float/scalar вычисления, возвращаем grad * множитель
                if self.per_channel:
                    # n — количество элементов на канал
                    n = tensor.numel() // tensor.shape[axis]
                else:
                    n = tensor.numel()
                g = 1.0 / math.sqrt(max(n * float(self.qmax), 1.0))
                # возвращаем grad * g (broadcast по каналу/скаляру)
                return grad * g
            self.scale.register_hook(grad_scale_hook)
            self.scale._lsq_hook = True

    def _forward_impl(self, tensor: Tensor) -> Tensor:
        if self.scale is None:
            raise RuntimeError("scale not initialized. Call init_scale_param and initialize_from_tensor first.")

        scale = self.scale.abs().clamp(min=1e-6)

        if self.per_channel:
            view_shape = [1] * tensor.ndim
            axis = self.channel_axis if self.channel_axis >= 0 else tensor.ndim + self.channel_axis
            view_shape[axis] = -1
            scale = scale.view(*view_shape)

        x_scaled = tensor / scale
        x_q = torch.clamp(torch.round(x_scaled), self.qmin, self.qmax)
        x_q_ste = x_scaled + (x_q - x_scaled).detach()
        return x_q_ste * scale


class QuantLinear(nn.Linear):
    """Linear layer wrapper that applies fake-quantizers to weights and activations."""

    def __init__(
        self,
        original: nn.Linear,
        weight_quantizer: FakeQuantizer,
        activation_quantizer: Optional[FakeQuantizer] = None,
    ) -> None:
        super().__init__(original.in_features, original.out_features, bias=original.bias is not None)
        device = original.weight.device
        self.to(device)
        self.weight.data.copy_(original.weight.data)
        if original.bias is not None and self.bias is not None:
            self.bias.data.copy_(original.bias.data)
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

    def forward(self, input: Tensor) -> Tensor:
        weight_q = self.weight_quantizer(self.weight)
        output = F.linear(input, weight_q, self.bias)
        if self.activation_quantizer is not None:
            output = self.activation_quantizer(output)
        return output


class QuantConv2d(nn.Conv2d):
    """Conv2d layer wrapper that applies fake-quantizers to weights and activations."""

    def __init__(
        self,
        original: nn.Conv2d,
        weight_quantizer: FakeQuantizer,
        activation_quantizer: Optional[FakeQuantizer] = None,
    ) -> None:
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
        device = original.weight.device
        self.to(device)
        self.weight.data.copy_(original.weight.data)
        if original.bias is not None and self.bias is not None:
            self.bias.data.copy_(original.bias.data)
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

    def forward(self, input: Tensor) -> Tensor:
        weight_q = self.weight_quantizer(self.weight)
        output = F.conv2d(
            input,
            weight_q,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.activation_quantizer is not None:
            output = self.activation_quantizer(output)
        return output


class QuantConv1d(nn.Conv1d):
    """Conv1d layer wrapper that applies fake-quantizers to weights and activations."""

    def __init__(
        self,
        original: nn.Conv1d,
        weight_quantizer: FakeQuantizer,
        activation_quantizer: Optional[FakeQuantizer] = None,
    ) -> None:
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
        device = original.weight.device
        self.to(device)
        self.weight.data.copy_(original.weight.data)
        if original.bias is not None and self.bias is not None:
            self.bias.data.copy_(original.bias.data)
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

    def forward(self, input: Tensor) -> Tensor:
        weight_q = self.weight_quantizer(self.weight)
        output = F.conv1d(
            input,
            weight_q,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.activation_quantizer is not None:
            output = self.activation_quantizer(output)
        return output


class QuantEmbedding(nn.Embedding):
    """Embedding wrapper that optionally quantizes embedding weights."""

    def __init__(self, original: nn.Embedding, weight_quantizer: FakeQuantizer) -> None:
        super().__init__(
            num_embeddings=original.num_embeddings,
            embedding_dim=original.embedding_dim,
            padding_idx=original.padding_idx,
        )
        self.to(original.weight.device)
        self.weight.data.copy_(original.weight.data)
        self.weight_quantizer = weight_quantizer

    def forward(self, input: Tensor) -> Tensor:
        weight_q = self.weight_quantizer(self.weight)
        return F.embedding(
            input,
            weight_q,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


class QuantStrategy(metaclass=abc.ABCMeta):
    """Base class shared by all quantization strategies."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.quantize_embedding = config.get("quantize_embedding", False)
        self.handles: List[Tuple[str, nn.Module]] = []
        self.model: Optional[nn.Module] = None
        self.logger: Any = None

    def attach(self, model: nn.Module) -> nn.Module:
        self.model = model
        modules = gather_quantizable_layers(model, quantize_embedding=self.quantize_embedding)
        for name, module in modules:
            wrapped = self._wrap_module(name, module)
            if wrapped is None:
                continue
            replace_module(model, name, wrapped)
            self.handles.append((name, wrapped))
        return model

    @abc.abstractmethod
    def _wrap_module(self, name: str, module: nn.Module) -> Optional[nn.Module]:
        """Wrap a module with fake-quant operations."""

    def set_logger(self, logger: Any) -> None:
        """Attach an experiment logger (ClearML, WandB, etc.)."""
        self.logger = logger

    def calibrate(self, loader) -> None:
        """Optional calibration step for certain strategies."""

    def step(self) -> None:
        """Optional hook to update internal state each training step."""

    def extra_state_dict(self) -> Dict[str, Any]:
        return {"config": self.config}

    def load_extra_state(self, state: Dict[str, Any]) -> None:
        self.config.update(state.get("config", {}))


class QATQuantStrategy(QuantStrategy):
    """Shared logic for QAT-style strategies."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.bits = config.get("bits", 8)
        self.activation_bits = config.get("activation_bits", self.bits)
        self.per_channel = config.get("per_channel", False)
        self.symmetric = config.get("symmetric", True)

    def create_weight_quantizer(self, module_name: str, module: nn.Module) -> FakeQuantizer:
        return UniformAffineQuantizer(
            bits=self.bits,
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            channel_axis=0,
        )

    def create_activation_quantizer(self, module_name: str, module: nn.Module) -> FakeQuantizer:
        return UniformAffineQuantizer(
            bits=self.activation_bits,
            symmetric=False,
            per_channel=False,
            channel_axis=-1,
        )

    def _wrap_module(self, name: str, module: nn.Module) -> Optional[nn.Module]:
        if isinstance(module, nn.Conv2d):
            wq = self.create_weight_quantizer(name, module)
            aq = self.create_activation_quantizer(name, module)
            with torch.no_grad():
                if hasattr(wq, "init_scale_param"):
                    num_channels = module.weight.shape[0] if getattr(wq, "per_channel", False) else 1
                    wq.init_scale_param(num_channels)
                wq.initialize_from_tensor(module.weight.data)
                wq.initialized.fill_(True)
                if aq is not None and hasattr(aq, "init_scale_param"):
                    act_channels = module.out_channels if getattr(aq, "per_channel", False) else 1
                    aq.init_scale_param(act_channels)
            device = module.weight.device
            wq.to(device)
            if aq is not None:
                aq.to(device)
            return QuantConv2d(module, wq, aq)
        if isinstance(module, nn.Conv1d):
            wq = self.create_weight_quantizer(name, module)
            aq = self.create_activation_quantizer(name, module)
            with torch.no_grad():
                if hasattr(wq, "init_scale_param"):
                    num_channels = module.weight.shape[0] if getattr(wq, "per_channel", False) else 1
                    wq.init_scale_param(num_channels)
                wq.initialize_from_tensor(module.weight.data)
                wq.initialized.fill_(True)
                if aq is not None and hasattr(aq, "init_scale_param"):
                    act_channels = module.out_channels if getattr(aq, "per_channel", False) else 1
                    aq.init_scale_param(act_channels)
            device = module.weight.device
            wq.to(device)
            if aq is not None:
                aq.to(device)
            return QuantConv1d(module, wq, aq)
        if isinstance(module, nn.Linear):
            wq = self.create_weight_quantizer(name, module)
            aq = self.create_activation_quantizer(name, module)
            with torch.no_grad():
                if hasattr(wq, "init_scale_param"):
                    num_channels = module.weight.shape[0] if getattr(wq, "per_channel", False) else 1
                    wq.init_scale_param(num_channels)
                wq.initialize_from_tensor(module.weight.data)
                wq.initialized.fill_(True)
                if aq is not None and hasattr(aq, "init_scale_param"):
                    act_channels = module.out_features if getattr(aq, "per_channel", False) else 1
                    aq.init_scale_param(act_channels)
            device = module.weight.device
            wq.to(device)
            if aq is not None:
                aq.to(device)
            return QuantLinear(module, wq, aq)
        if isinstance(module, nn.Embedding) and self.quantize_embedding:
            wq = self.create_weight_quantizer(name, module)
            with torch.no_grad():
                if hasattr(wq, "init_scale_param"):
                    num_channels = module.weight.shape[0] if getattr(wq, "per_channel", False) else 1
                    wq.init_scale_param(num_channels)
                wq.initialize_from_tensor(module.weight.data)
                wq.initialized.fill_(True)
            device = module.weight.device
            wq.to(device)
            return QuantEmbedding(module, wq)
        return None
