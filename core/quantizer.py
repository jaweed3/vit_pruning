"""
Post-Training Quantization (PTQ) for ViT models.
Dynamic quant is CPU-only — handles device move internally.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import torch
import torch.ao.quantization as quant


@dataclass
class PTQConfig:
    quant_type: str = "dynamic"
    dtype: str = "qint8"
    calibration_samples: int = 128
    per_channel: bool = True
    export_onnx: bool = True


def apply_ptq(model: torch.nn.Module,
              config: PTQConfig,
              calibration_loader: Optional[Callable] = None,
              input_shape=(1, 3, 224, 224)) -> Tuple[torch.nn.Module, str]:
    """Apply PTQ — returns (quantized_model, device) where device
    is the actual device the quantized model lives on (CPU for dynamic)."""
    model.eval()

    if config.quant_type == "dynamic":
        return _apply_dynamic_quant(model, config), "cpu"
    elif config.quant_type == "static":
        if calibration_loader is None:
            raise ValueError("calibration_loader required for static quant")
        return _apply_static_quant(model, config, calibration_loader), "cpu"
    else:
        raise ValueError(f"Unknown quant_type: {config.quant_type}")


def _apply_dynamic_quant(model, config):
    """Dynamic quant — CPU only. Moves model to CPU first."""
    model = model.cpu()
    quantized = quant.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=getattr(torch, config.dtype),
    )
    return quantized


def _apply_static_quant(model, config, calib_loader):
    """Static quantization with calibration — CPU only."""
    model = model.cpu()
    model.qconfig = quant.get_default_qconfig('x86')
    quant.prepare(model, inplace=True)
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(calib_loader):
            if i >= config.calibration_samples:
                break
            model(inputs)
    quant.convert(model, inplace=True)
    return model


def export_to_onnx(model: torch.nn.Module,
                   output_path: str,
                   input_shape=(1, 3, 224, 224)):
    model.eval()
    dummy = torch.randn(input_shape).to(next(model.parameters()).device)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"], output_names=["output"],
        opset_version=17, do_constant_folding=True,
    )
    return output_path
