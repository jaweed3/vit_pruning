"""
Evaluation metrics for ViT pruning + quantization benchmarking.

Measures:
- Top-1 / Top-5 accuracy
- Model size (MB, parameter count)
- FLOPs
- Inference latency
- Throughput (images/sec)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
import time


@dataclass
class ModelMetrics:
    """Container for all measured metrics."""
    top1: Optional[float] = None
    top5: Optional[float] = None
    params_m: float = 0.0
    flops_m: float = 0.0
    size_mb: float = 0.0
    latency_ms: float = 0.0
    throughput: float = 0.0

    def __repr__(self):
        top1_str = f"{self.top1:.2f}%" if self.top1 is not None else "N/A"
        top5_str = f"{self.top5:.2f}%" if self.top5 is not None else "N/A"
        lines = [
            "Metrics(",
            f"  Top-1: {top1_str} | Top-5: {top5_str}",
            f"  Params: {self.params_m:.2f}M | FLOPs: {self.flops_m:.2f}M",
            f"  Size: {self.size_mb:.2f}MB | Latency: {self.latency_ms:.2f}ms",
            f"  Throughput: {self.throughput:.0f} img/s",
            ")",
        ]
        return "\n".join(lines)


def evaluate_model(model: nn.Module,
                   val_loader: Optional[torch.utils.data.DataLoader] = None,
                   device: str = "cuda") -> ModelMetrics:
    """Full evaluation: accuracy + compute metrics."""
    metrics = ModelMetrics()
    metrics.params_m = count_parameters(model)
    metrics.size_mb = compute_model_size(model)
    metrics.flops_m = measure_flops(model, device=device)
    metrics.latency_ms, metrics.throughput = measure_latency(model, device=device)

    if val_loader is not None:
        metrics.top1, metrics.top5 = measure_accuracy(model, val_loader, device)

    return metrics


def count_parameters(model: nn.Module) -> float:
    """Count trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def compute_model_size(model: nn.Module) -> float:
    """Estimate model size in MB (assuming float32 storage)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def measure_flops(model: nn.Module,
                  input_shape: Tuple = (1, 3, 224, 224),
                  device: str = "cuda") -> float:
    """Measure FLOPs using thop. Returns millions of FLOPs."""
    try:
        from thop import profile
        model = model.to(device)
        dummy = torch.randn(input_shape).to(device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e6
    except Exception:
        return 0.0


def measure_latency(model: nn.Module,
                    input_shape: Tuple = (1, 3, 224, 224),
                    device: str = "cuda",
                    num_warmup: int = 50,
                    num_iters: int = 200) -> Tuple[float, float]:
    """Measure average inference latency and throughput."""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(input_shape).to(device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)

    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(dummy)
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - start

    avg_latency = (elapsed / num_iters) * 1000
    throughput = num_iters / elapsed
    return avg_latency, throughput


def measure_accuracy(model: nn.Module,
                     val_loader: torch.utils.data.DataLoader,
                     device: str = "cuda",
                     max_batches: Optional[int] = None) -> Tuple[float, float]:
    """Top-1 and Top-5 accuracy."""
    model = model.to(device)
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, pred5 = outputs.topk(5, dim=1)
            top1_correct += (outputs.argmax(dim=1) == targets).sum().item()
            top5_correct += (pred5 == targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.size(0)

    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total
    return top1, top5
