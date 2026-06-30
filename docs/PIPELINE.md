# Pipeline Overview

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│ Load Model  │───▶│  Structured  │───▶│  Post-Train  │───▶│  Benchmark  │
│ (timm)      │    │   Pruning    │    │  Quantization │    │   & Export  │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
```

## 1. Model Loading
- Supports DeiT, EfficientFormer, MobileViT, EdgeViT
- Pretrained weights from ImageNet-1k

## 2. Structured Pruning
- **Head pruning**: Reduce attention heads per block
- **MLP pruning**: Reduce intermediate dimension
- **Block skipping**: Remove entire transformer blocks

## 3. Post-Training Quantization
- Dynamic quantization (weights INT8)
- Static quantization (weights + activations)
- ONNX export

## 4. Benchmarking
Metrics measured:
- Top-1 / Top-5 accuracy
- Parameter count
- FLOPs
- Model size (MB)
- Inference latency (ms)
- Throughput (img/s)

## Running
```bash
# Full pipeline
python scripts/benchmark.py --model deit_tiny --head-ratio 0.5 --mlp-ratio 0.5 --quantize

# Export ONNX
python scripts/benchmark.py --model efficientformer_l1 --head-ratio 0.75 --export-onnx
```
