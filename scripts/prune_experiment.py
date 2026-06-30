#!/usr/bin/env python3
"""
Run a grid of pruning experiments and save results.
Scans different pruning ratios for multiple models.
"""

import sys
import os
import json
import subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODELS = ["deit_tiny", "efficientformer_l1", "mobilevit_s"]
HEAD_RATIOS = [1.0, 0.75, 0.5, 0.25]
MLP_RATIOS = [1.0, 0.75, 0.5]
SKIP_COUNTS = [0, 1, 2]
QUANTIZE_OPTIONS = [False, True]


def run_experiment(model, head_ratio=1.0, mlp_ratio=1.0,
                   skip_blocks=0, quantize=False):
    """Run a single experiment and return the output file path."""
    cmd = [
        sys.executable, "scripts/benchmark.py",
        "--model", model,
        "--head-ratio", str(head_ratio),
        "--mlp-ratio", str(mlp_ratio),
        "--skip-blocks", str(skip_blocks),
        "--output", f"results/runs/{model}",
    ]
    if quantize:
        cmd.append("--quantize")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr[-500:])

    return result.returncode


def main():
    print("=== Grid Experiment Runner ===\n")

    # Quick scan: vary head ratio only
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        # Baseline
        run_experiment(model)
        # Head pruning only
        for hr in HEAD_RATIOS[1:]:  # skip 1.0 (already baseline)
            run_experiment(model, head_ratio=hr)
        # Head + MLP
        for hr in [0.75, 0.5]:
            for mr in [0.75, 0.5]:
                run_experiment(model, head_ratio=hr, mlp_ratio=mr)
        # Best config + quantization
        run_experiment(model, head_ratio=0.75, mlp_ratio=0.75, quantize=True)

    print("\n=== All experiments complete ===")


if __name__ == "__main__":
    main()
