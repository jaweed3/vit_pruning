"""
Post-Training Quantization (PTQ) for ViT models.

Supports:
- Dynamic Quantization (weights only, readily available)
- Static Quantization (weights + activations, with calibration)
- Export to ONNX with INT8 quantization

Note: PyTorch static quantization has limited support for ViT 
architectures (nn.MultiheadAttention fused qkv). Dynamic quantization
is the most practical approach for edge deployment.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import torch
import torch.ao.quantization as quant


@dataclass
class PTQConfig:
    """Post-Training Quantization configuration."""
    quant_type: str = "dynamic"           # "dynamic" or "static"
    dtype: str = "qint8"                  # quantization dtype
    calibration_samples: int = 128        # samples for static calibration
    per_channel: bool = True             # per-channel weight quantization
    export_onnx: bool = True             # export to ONNX after quantization


def apply_ptq(model: torch.nn.Module,
              config: PTQConfig,
              calibration_loader: Optional[Callable] = None,
              input_shape=(1, 3, 224, 224)) -> torch.nn.Module:
    """Apply post-training quantization to a ViT model.

    Args:
        model: The model to quantize (in eval mode).
        config: Quantization configuration.
        calibration_loader: Data iterator for static quantization calibration.
        input_shape: Example input shape for ONNX export.

    Returns:
        Quantized model.
    """
    model.eval()

    if config.quant_type == "dynamic":
        return _apply_dynamic_quant(model, config)
    elif config.quant_type == "static":
        if calibration_loader is None:
            raise ValueError("calibration_loader required for static quantization")
        return _apply_static_quant(model, config, calibration_loader)
    else:
        raise ValueError(f"Unknown quant_type: {config.quant_type}")


def _apply_dynamic_quant(model, config):
    """Apply dynamic quantization (weights only)."""
    # Dynamic quantization works best on Linear layers
    quantized = quant.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=getattr(torch, config.dtype),
    )
    return quantized


def _apply_static_quant(model, config, calib_loader):
    """Apply static quantization with calibration."""
    # Prepare model for fusion and quantization
    model.qconfig = quant.get_default_qconfig('x86')
    quant.prepare(model, inplace=True)

    # Calibrate
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(calib_loader):
            if i >= config.calibration_samples:
                break
            model(inputs)

    # Convert to quantized
    quant.convert(model, inplace=True)
    return model


def export_to_onnx(model: torch.nn.Module,
                   output_path: str,
                   input_shape=(1, 3, 224, 224)):
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(input_shape).to(next(model.parameters()).device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )
    return output_path
