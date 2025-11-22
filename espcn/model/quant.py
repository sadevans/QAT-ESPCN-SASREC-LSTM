# espcn/model/quant.py
import torch
from torch import nn, Tensor
from .base import BaseESPCN

__all__ = ["QuantESPCN"]


class QuantESPCN(BaseESPCN):
    """
    ESPCN with QAT/PTQ support via quantization strategies.
    Supports LSQ, APoT, QDrop (QAT) and AdaRound (PTQ).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 64,
        upscale_factor: int = 3,
        quant_strategy=None,  # instance of QuantStrategy
    ):
        super().__init__(in_channels, out_channels, channels, upscale_factor)
        self.quant_strategy = quant_strategy
        self.quant_enabled = True

    def prepare_quant(self, strategy_name: str, config: dict):
        """Attach quantization strategy (for QAT or PTQ)."""
        if strategy_name == "lsq":
            from quant.lsq import LSQQuantStrategy
            self.quant_strategy = LSQQuantStrategy(config)
        elif strategy_name == "apot":
            from quant.apot import APoTQuantStrategy
            self.quant_strategy = APoTQuantStrategy(config)
        elif strategy_name == "qdrop":
            from quant.qdrop import QDropQuantStrategy
            self.quant_strategy = QDropQuantStrategy(config)
        elif strategy_name == "adaround":
            from quant.adaround import AdaRoundQuantStrategy
            self.quant_strategy = AdaRoundQuantStrategy(config)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # attach strategy directly to self (modifies the model in-place)
        self.quant_strategy.attach(self)
        print(f"Quantization strategy attached:\n{self.quant_strategy}")

    def forward(self, x: Tensor) -> Tensor:
        # always use base forward - quantization is applied in-place via strategy
        return super().forward(x)

    def calibrate(self, dataloader):
        """For PTQ strategies like AdaRound."""
        if hasattr(self.quant_strategy, "calibrate"):
            self.quant_strategy.calibrate(dataloader)

    def disable_quant(self):
        self.quant_enabled = False

    def enable_quant(self):
        self.quant_enabled = True

    def state_dict(self, *args, **kwargs):
        # return base model state - quantization parameters are included in the model
        return super().state_dict(*args, **kwargs)
    
    def convert_quantized_weights_to_static(self) -> None:
        """
        Replace all quantized convolution layers with standard nn.Conv2d
        containing the final quantized weights.
        This removes FakeQuantizer logic and makes the model exportable.
        """
        from quant.base import QuantConv2d

        def replace_fn(name: str, module: nn.Module) -> Optional[nn.Module]:
            if isinstance(module, QuantConv2d):
                with torch.no_grad():
                    weight_q = module.weight_quantizer(module.weight)
                    new_conv = nn.Conv2d(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        groups=module.groups,
                        bias=module.bias is not None,
                        padding_mode=module.padding_mode,
                    )
                    new_conv.weight.data.copy_(weight_q)
                    if module.bias is not None:
                        new_conv.bias.data.copy_(module.bias.data)
                    return new_conv
            return None

        from utils import replace_module
        for name, module in list(self.named_modules()):
            new_module = replace_fn(name, module)
            if new_module is not None:
                replace_module(self, name, new_module)