#!/usr/bin/env python3
"""
Full benchmark pipeline:
  1. Load a lightweight ViT model
  2. Measure baseline metrics
  3. Apply structured pruning (actually modifies weights)
  4. Apply PTQ (CPU-only, handles device move internally)
  5. Measure & compare
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from core.model_loader import load_model, SUPPORTED_MODELS
from core.pruner import StructuredPruner, PruningConfig
from core.quantizer import PTQConfig, apply_ptq, export_to_onnx
from core.evaluator import evaluate_model, ModelMetrics
from core.utils import save_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="ViT Pruning + PTQ Benchmark")
    parser.add_argument("--model", default="deit_tiny", choices=list(SUPPORTED_MODELS))
    parser.add_argument("--head-ratio", type=float, default=1.0,
                        help="Keep ratio for heads")
    parser.add_argument("--mlp-ratio", type=float, default=1.0,
                        help="Keep ratio for MLP dim")
    parser.add_argument("--skip-blocks", type=int, default=0,
                        help="Remove N last blocks")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply PTQ after pruning (moves to CPU)")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export to ONNX")
    parser.add_argument("--output", default="results/runs",
                        help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print("Config: head_ratio={}, mlp_ratio={}, skip_blocks={}".format(
        args.head_ratio, args.mlp_ratio, args.skip_blocks))

    # 1. Load baseline (on CUDA)
    print("")
    print("=== Loading baseline model ===")
    model, _ = load_model(args.model, pretrained=True)
    baseline = evaluate_model(model, val_loader=None, device=device)
    print("Baseline:", baseline)

    # 2. Prune (stays on CUDA)
    do_prune = (args.head_ratio < 1.0 or args.mlp_ratio < 1.0
                or args.skip_blocks > 0)
    if do_prune:
        print("")
        print("=== Pruning ===")
        config = PruningConfig(
            head_prune_ratio=args.head_ratio,
            mlp_prune_ratio=args.mlp_ratio,
            skip_blocks=args.skip_blocks,
        )
        pruner = StructuredPruner(model, config)
        pruned_model = pruner.prune()
        print("Pruning report:", pruner.summarize())
        pruned_metrics = evaluate_model(pruned_model, val_loader=None,
                                        device=device)
        print("After prune:", pruned_metrics)
    else:
        pruned_model = model
        pruned_metrics = baseline

    # 3. Quantize (moves to CPU internally)
    if args.quantize:
        print("")
        print("=== Quantizing (CPU) ===")
        quant_config = PTQConfig(quant_type="dynamic",
                                 export_onnx=args.export_onnx)
        quant_model, quant_device = apply_ptq(pruned_model, quant_config)

        from core.evaluator import count_parameters, compute_model_size
        quant_metrics = ModelMetrics(
            params_m=count_parameters(quant_model),
            size_mb=compute_model_size(quant_model),
            flops_m=pruned_metrics.flops_m,
        )
        from core.evaluator import measure_latency
        quant_metrics.latency_ms, quant_metrics.throughput = \
            measure_latency(quant_model, device=quant_device)
        print("After PTQ:", quant_metrics)
    else:
        quant_model = pruned_model
        quant_metrics = pruned_metrics

    # 4. Export ONNX
    if args.export_onnx:
        onnx_path = "{}/{}_{:.2f}_ptq.onnx".format(
            args.output, args.model, args.head_ratio)
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        export_to_onnx(quant_model, onnx_path)
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print("")
        print("ONNX exported: {} ({:.2f} MB)".format(onnx_path, onnx_size))

    # 5. Save
    results = {
        "model": args.model,
        "config": {
            "head_ratio": args.head_ratio,
            "mlp_ratio": args.mlp_ratio,
            "skip_blocks": args.skip_blocks,
            "quantize": args.quantize,
        },
        "baseline": {
            "params_m": baseline.params_m,
            "size_mb": baseline.size_mb,
            "flops_m": baseline.flops_m,
            "latency_ms": baseline.latency_ms,
            "throughput": baseline.throughput,
        },
        "pruned": {
            "params_m": pruned_metrics.params_m,
            "size_mb": pruned_metrics.size_mb,
            "flops_m": pruned_metrics.flops_m,
            "latency_ms": pruned_metrics.latency_ms,
            "throughput": pruned_metrics.throughput,
        },
        "quantized": {
            "params_m": quant_metrics.params_m,
            "size_mb": quant_metrics.size_mb,
            "latency_ms": quant_metrics.latency_ms,
            "throughput": quant_metrics.throughput,
        },
    }
    saved = save_metrics(results, args.output)
    print("")
    print("Results saved:", saved)

    # Summary
    print("")
    print("=== SUMMARY ===")
    pc = (1 - quant_metrics.params_m / baseline.params_m) * 100
    sc = (1 - quant_metrics.size_mb / baseline.size_mb) * 100
    lc = (quant_metrics.latency_ms / baseline.latency_ms - 1) * 100
    print("Params:  {:.2f}M -> {:.2f}M ({:.1f}% reduction)".format(
        baseline.params_m, quant_metrics.params_m, pc))
    print("Size:    {:.1f}MB -> {:.1f}MB ({:.1f}% reduction)".format(
        baseline.size_mb, quant_metrics.size_mb, sc))
    print("Latency: {:.1f}ms -> {:.1f}ms ({:.1f}% change)".format(
        baseline.latency_ms, quant_metrics.latency_ms, lc))


if __name__ == "__main__":
    main()
