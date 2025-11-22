"""Export ESPCN checkpoints to ONNX format and convert to INT8 using ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from espcn.model.quant import QuantESPCN
from quant.base import QuantConv2d
from quant.adaround import AdaRoundModule
from utils import (
    get_espcn_method_configs,
    load_config,
    replace_module,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for ONNX export script.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Export ESPCN checkpoints referenced in benchmark JSON files to ONNX."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["results/espcn_quant_benchmark.json"],
        help="One or more JSON files containing records with 'checkpoint_path'.",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/espcn/base.yaml",
        help="Base ESPCN config to instantiate model architecture.",
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default="onnx_exports",
        help="Directory for ONNX outputs.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models whose ONNX file already exists.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Whitelist of methods to export (e.g. fp32 lsq adaround).",
    )
    return parser.parse_args()


def require_onnx() -> None:
    """
    Check if ONNX is installed, exit if not.
    
    Raises:
        SystemExit: If ONNX is not installed.
    """
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit(
            "onnx is not installed. Run:\n  pip install onnx onnxruntime"
        ) from exc


def resolve_files(patterns: Sequence[str]) -> List[Path]:
    """
    Resolve file patterns to list of Path objects.
    
    Args:
        patterns: Sequence of file path patterns (supports glob).
        
    Returns:
        Sorted list of unique Path objects for existing files.
        
    Raises:
        SystemExit: If no files match the patterns.
    """
    files: List[Path] = []
    for pattern in patterns:
        matches = list(Path().glob(pattern))
        files.extend(m for m in matches if m.is_file())
    if not files:
        raise SystemExit(f"No input JSON files found: {patterns}")
    return sorted(set(files))


def load_records(path: Path) -> List[Dict[str, Any]]:
    """
    Load benchmark records from JSON file.
    
    Args:
        path: Path to JSON file containing records.
        
    Returns:
        List of benchmark record dictionaries.
        
    Raises:
        ValueError: If JSON format is not supported (must be dict or list).
    """
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported JSON in {path}")


def build_model(base_config: Dict[str, Any]) -> QuantESPCN:
    """
    Build QuantESPCN model from base configuration.
    
    Args:
        base_config: Configuration dictionary with 'model' key.
        
    Returns:
        QuantESPCN model instance.
    """
    return QuantESPCN(**base_config["model"])


def _maybe_prepare_quant(
    model: QuantESPCN,
    method: str,
    quant_cfg: Dict[str, Any],
) -> None:
    """
    Prepare quantization strategy for model if method is not FP32.
    
    Initializes quantizer parameters (for QAT methods) or alpha parameters
    (for AdaRound) before loading state dict.
    
    Args:
        model: ESPCN model to prepare.
        method: Quantization method name ('fp32', 'lsq', 'apot', 'qdrop', 'adaround').
        quant_cfg: Quantization configuration dictionary.
    """
    method = method.lower()
    if method in {"fp32", "none"}:
        return

    model.prepare_quant(method, quant_cfg or {})

    if method in {"lsq", "apot", "qdrop"}:
        model.train()
        dummy = torch.randn(1, 1, 96, 96)
        with torch.no_grad():
            _ = model(dummy)
        model.eval()
    elif method == "adaround":
        adaround_modules = [
            m for _, m in model.named_modules() if isinstance(m, AdaRoundModule)
        ]
        for module in adaround_modules:
            if getattr(module, "alpha", None) is None or not getattr(module, "alpha_init", False):
                module._init_alpha(module.weight)
            if isinstance(module.alpha_init, torch.Tensor):
                with torch.no_grad():
                    module.alpha_init.fill_(True)
            if isinstance(module.alpha_init, torch.Tensor):
                with torch.no_grad():
                    module.alpha_init.fill_(True)


def load_state_dict(checkpoint: Path) -> Dict[str, Any]:
    """
    Load model state dictionary from checkpoint file.
    
    Args:
        checkpoint: Path to checkpoint file.
        
    Returns:
        Model state dictionary (extracted from checkpoint if nested).
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            return state["model_state_dict"]
        if "model_state" in state:
            return state["model_state"]
    return state


def convert_quantized_weights_to_static(model: QuantESPCN) -> None:
    """
    Replace QuantConv2d layers with standard nn.Conv2d containing final quantized weights.
    Removes all FakeQuantizer logic to make model exportable.
    """
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
        if isinstance(module, AdaRoundModule):
            with torch.no_grad():
                weight_q = module.get_quantized_weight(module.weight)
            if isinstance(module, nn.Conv2d):
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
            if isinstance(module, nn.Conv1d):
                new_conv1d = nn.Conv1d(
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
                new_conv1d.weight.data.copy_(weight_q)
                if module.bias is not None:
                    new_conv1d.bias.data.copy_(module.bias.data)
                return new_conv1d
            if isinstance(module, nn.Linear):
                new_linear = nn.Linear(
                    module.in_features, module.out_features, module.bias is not None
                )
                new_linear.weight.data.copy_(weight_q)
                if module.bias is not None:
                    new_linear.bias.data.copy_(module.bias.data)
                return new_linear
        return None

    for name, module in list(model.named_modules()):
        new_module = replace_fn(name, module)
        if new_module is not None:
            replace_module(model, name, new_module)


def export_to_onnx(model: QuantESPCN, sample: torch.Tensor, onnx_path: Path, opset: int) -> None:
    """
    Export model to ONNX format.
    
    Args:
        model: ESPCN model to export (should have quantized weights converted to static).
        sample: Example input tensor for tracing.
        onnx_path: Path to save ONNX model.
        opset: ONNX opset version.
    """
    os.environ.setdefault("TORCH_ONNX_EXPERIMENTAL_EXPORTER", "0")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        str(onnx_path),
        input_names=["lr"],
        output_names=["sr"],
        dynamic_axes={
            "lr": {0: "batch", 2: "height", 3: "width"},
            "sr": {0: "batch", 2: "out_height", 3: "out_width"},
        },
        opset_version=opset,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )


def iter_input_records(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    """
    Iterate over records from multiple JSON files.
    
    Args:
        paths: Iterable of paths to JSON files containing benchmark records.
        
    Yields:
        Benchmark record dictionaries from all input files.
    """
    for path in paths:
        for record in load_records(path):
            record["_source"] = str(path)
            yield record


def main() -> None:
    require_onnx()
    args = parse_args()
    inputs = resolve_files(args.inputs)
    base_config = load_config(args.base_config)
    method_configs = get_espcn_method_configs()
    onnx_dir = Path(args.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    whitelist = {m.lower() for m in args.methods} if args.methods else None

    exported = 0
    for record in iter_input_records(inputs):
        quant_method = str(record.get("quant_method", "fp32")).lower()
        if whitelist and quant_method not in whitelist:
            continue

        checkpoint_key = record.get("checkpoint_path") or record.get("checkpoint")
        if not checkpoint_key:
            raise KeyError(f"'checkpoint_path' missing in {record['_source']}")
        checkpoint = Path(checkpoint_key)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        model_name = record.get("model") or checkpoint.stem
        onnx_path = onnx_dir / f"{model_name}.onnx"
        if args.skip_existing and onnx_path.exists():
            print(f"[skip] {onnx_path} already exists.")
            continue

        config_entry = method_configs.get(quant_method)
        quant_cfg: Dict[str, Any] = {}
        if config_entry:
            cfg_path = Path(config_entry[0])
            if cfg_path.exists():
                quant_cfg = load_config(str(cfg_path)).get("quantization", {})

        print(f"[export] Model={model_name} | method={quant_method} | checkpoint={checkpoint}")
        model = build_model(base_config).to("cpu")
        _maybe_prepare_quant(model, quant_method, quant_cfg)
        state_dict = load_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        if quant_method != "fp32":
            print(f"[convert] Converting to static quantized weights...")
            convert_quantized_weights_to_static(model)

        dummy_sample = torch.randn(1, 1, 96, 96)
        export_to_onnx(model, dummy_sample, onnx_path, opset=args.opset)
        print(f"[done] Saved ONNX model to {onnx_path}")
        exported += 1

    if exported == 0:
        print("No models were exported (check filters or input data).")
    else:
        print(f"Exported {exported} ONNX model(s) to {onnx_dir}")


if __name__ == "__main__":
    main()