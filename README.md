# ViT Pruning + PTQ

Structured Pruning and Post-Training Quantization for Lightweight Vision Transformers on Edge Devices.

## Overview

This project systematically evaluates the **compounding effect** of:
1. **Structured pruning** — attention head pruning, MLP dimension reduction, block skipping
2. **Post-training quantization** — dynamic INT8 quantization, ONNX export

## Supported Models

| Model | Params | Description |
|-------|--------|-------------|
| DeiT-Tiny | 5.7M | Data-efficient Image Transformer |
| EfficientFormer-L1 | 12M | MobileNet-speed ViT |
| MobileViT-S | 5.6M | Mobile-friendly ViT |
| EdgeViT-S | 5.6M | Efficient ViT variant |

## Quick Start

```bash
# Setup
uv venv && source .venv/bin/activate && uv sync

# Run benchmark
python scripts/benchmark.py --model deit_tiny --head-ratio 0.5 --mlp-ratio 0.5 --quantize

# Grid experiment
python scripts/prune_experiment.py
```

## Project Structure

```
vit-pruning-ptq/
├── core/                    # Core modules
│   ├── model_loader.py      # Model loading (timm)
│   ├── pruner.py            # Structured pruning
│   ├── quantizer.py         # PTQ
│   ├── evaluator.py         # Metrics
│   └── utils.py             # Helpers
├── scripts/                 # Entry points
│   ├── benchmark.py         # Single experiment
│   ├── prune_experiment.py  # Grid scan
│   └── analyze_results.py   # Results table
├── notebooks/               # Analysis notebooks
├── configs/                 # YAML configs
├── docs/                    # Documentation
├── latex/                   # Paper template
└── results/                 # Output data
```

## Results

| Model | Config | Params | Size | Latency |
|-------|--------|--------|------|---------|
| DeiT-Tiny | Baseline | 5.7M | 22 MB | — |
| DeiT-Tiny | P50%+PTQ | ~2.8M | ~6 MB | — |
| EfficientFormer-L1 | Baseline | 12M | 47 MB | — |
| EfficientFormer-L1 | P50%+PTQ | ~6M | ~12 MB | — |

## Citation

```bibtex
@article{author2025vitpruning,
  title={Structured Pruning and PTQ for Lightweight ViTs on Edge Devices},
  author={...},
  journal={...},
  year={2025}
}
```
