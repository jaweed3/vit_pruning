#!/usr/bin/env python3
"""
Full benchmark pipeline:
  1. Load a lightweight ViT model
  2. Measure baseline metrics
  3. Apply structured pruning
  4. Apply PTQ
  5. Measure & compare

Usage:
    python scripts/benchmark.py --model deit_tiny --head-ratio 0.75 --mlp-ratio 0.75
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
    parser.add_argument("--head-ratio", type=float, default=1.0, help="Keep ratio for heads")
    parser.add_argument("--mlp-ratio", type=float, default=1.0, help="Keep ratio for MLP dim")
    parser.add_argument("--skip-blocks", type=int, default=0, help="Remove N last blocks")
    parser.add_argument("--quantize", action="store_true", help="Apply PTQ after pruning")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--output", default="results/runs", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Config: head_ratio={args.head_ratio}, mlp_ratio={args.mlp_ratio}, skip_blocks={args.skip_blocks}")

    # 1. Load baseline
    print("\n=== Loading baseline model ===")
    model, _ = load_model(args.model, pretrained=True)
    baseline = evaluate_model(model, val_loader=None, device=device)
    print("Baseline:", baseline)

    # 2. Prune
    if args.head_ratio < 1.0 or args.mlp_ratio < 1.0 or args.skip_blocks > 0:
        print("\n=== Pruning ===")
        config = PruningConfig(
            head_prune_ratio=args.head_ratio,
            mlp_prune_ratio=args.mlp_ratio,
            skip_blocks=args.skip_blocks,
        )
        pruner = StructuredPruner(model, config)
        pruned_model = pruner.prune()
        print("Pruning report:", pruner.summarize())
        pruned_metrics = evaluate_model(pruned_model, val_loader=None, device=device)
        print("After prune:", pruned_metrics)
    else:
        pruned_model = model
        pruned_metrics = baseline

    # 3. Quantize
    if args.quantize:
        print("\n=== Quantizing ===")
        quant_config = PTQConfig(quant_type="dynamic", export_onnx=args.export_onnx)
        quant_model = apply_ptq(pruned_model, quant_config)

        # Recomputed metrics for quantized model
        from core.evaluator import count_parameters, compute_model_size
        quant_metrics = ModelMetrics(
            params_m=count_parameters(quant_model),
            size_mb=compute_model_size(quant_model),
            flops_m=pruned_metrics.flops_m,  # FLOPs same, weights smaller
        )

        # Recompute latency
        from core.evaluator import measure_latency
        quant_metrics.latency_ms, quant_metrics.throughput = measure_latency(quant_model, device=device)
        print("After PTQ:", quant_metrics)
    else:
        quant_model = pruned_model
        quant_metrics = pruned_metrics

    # 4. Export ONNX
    if args.export_onnx:
        onnx_path = f"{args.output}/{args.model}_pruned{args.head_ratio}_ptq.onnx"
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        export_to_onnx(quant_model, onnx_path)
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"\nONNX exported: {onnx_path} ({onnx_size:.2f} MB)")

    # 5. Save results
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
    print(f"\nResults saved: {saved}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"  Params:  {baseline.params_m:.2f}M -> {quant_metrics.params_m:.2f}M "
          f"({(1 - quant_metrics.params_m/baseline.params_m)*100:.1f}% reduction)")
    print(f"  Size:    {baseline.size_mb:.1f}MB -> {quant_metrics.size_mb:.1f}MB "
          f"({(1 - quant_metrics.size_mb/baseline.size_mb)*100:.1f}% reduction)")
    print(f"  Latency: {baseline.latency_ms:.1f}ms -> {quant_metrics.latency_ms:.1f}ms "
          f"({(quant_metrics.latency_ms/baseline.latency_ms-1)*100:.1f}% change)")


if __name__ == "__main__":
    main()
