# Setup Guide

## Requirements
- Ubuntu 22.04+ / WSL2
- Python 3.11+
- CUDA-capable GPU (optional, for faster benchmarking)

## Installation

### Using UV (recommended)
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install
cd vit-pruning-ptq
uv venv
source .venv/bin/activate
uv sync
```

### Using pip
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Verify GPU
```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Quick Test
```bash
python3 scripts/benchmark.py --model deit_tiny --head-ratio 0.75
```
