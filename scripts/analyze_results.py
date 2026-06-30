#!/usr/bin/env python3
"""
Load all experiment results and generate comparison tables.
"""

import sys
import os
import json
import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.utils import load_metrics


def collect_results(base_dir="results/runs"):
    """Collect all metric JSONs from experiment runs."""
    results = []
    for path in glob.glob(f"{base_dir}/**/metrics_*.json", recursive=True):
        try:
            data = load_metrics(path)
            results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    return results


def print_table(results):
    """Print a comparison table."""
    print(f"{'Model':<20} {'Config':<25} {'Params(M)':<10} {'Size(MB)':<10} "
          f"{'Lat(ms)':<10} {'Thr(img/s)':<12}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: (x.get("model", ""), x["baseline"]["params_m"])):
        model = r.get("model", "?")
        cfg = f"H{r['config']['head_ratio']:.2f}_M{r['config']['mlp_ratio']:.2f}"
        if r["config"]["skip_blocks"]:
            cfg += f"_skip{r['config']['skip_blocks']}"
        if r["config"]["quantize"]:
            cfg += "_Q"
        b = r["baseline"]
        p = r.get("pruned", b)
        q = r.get("quantized", p)
        print(f"{model:<20} {cfg:<25} {q['params_m']:<10.2f} {q['size_mb']:<10.1f} "
              f"{q['latency_ms']:<10.1f} {q['throughput']:<12.0f}")


def main():
    results = collect_results()
    print(f"Found {len(results)} experiments\n")
    print_table(results)


if __name__ == "__main__":
    main()
