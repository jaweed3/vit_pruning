"""
Utility functions: metrics I/O, parameter counting, profiling helpers.
"""
from typing import Optional, Dict, Tuple
import json
import os
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> float:
    """Millions of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def measure_flops(model: nn.Module,
                  input_shape: Tuple = (1, 3, 224, 224),
                  device: str = "cpu") -> float:
    """Return FLOPs in millions (via thop). Returns 0 on failure."""
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
                    device: str = "cpu",
                    num_warmup: int = 50,
                    num_iters: int = 200) -> Tuple[float, float]:
    """Return (latency_ms, throughput_imgps)."""
    model = model.to(device).eval()
    dummy = torch.randn(input_shape).to(device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)

    torch.cuda.synchronize() if device == "cuda" else None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(dummy)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    avg_latency = elapsed_ms / num_iters
    throughput = (num_iters / elapsed_ms) * 1000
    return avg_latency, throughput


def save_metrics(metrics: dict, output_dir: str = "results/runs",
                 filename: Optional[str] = None) -> str:
    """Save metrics dict to JSON. Returns the saved path."""
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        model = metrics.get("model", "unknown")
        filename = "metrics_{}.json".format(model)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def load_metrics(path: str) -> dict:
    """Load metrics from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)
