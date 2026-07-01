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
    pattern = os.path.join(base_dir, "**", "metrics_*.json")
    for path in glob.glob(pattern, recursive=True):
        try:
            data = load_metrics(path)
            results.append(data)
        except Exception as e:
            print("Warning: Could not load {}: {}".format(path, e))
    return results


def print_table(results):
    """Print a comparison table."""
    header = "{:<20} {:<25} {:<10} {:<10} {:<10} {:<12}"
    sep = "-" * 90
    row_fmt = "{:<20} {:<25} {:<10.2f} {:<10.1f} {:<10.1f} {:<12.0f}"

    print(header.format("Model", "Config", "Params(M)", "Size(MB)",
                        "Lat(ms)", "Thr(img/s)"))
    print(sep)

    def sort_key(x):
        return (x.get("model", "?"), x["baseline"]["params_m"])

    for r in sorted(results, key=sort_key):
        model = r.get("model", "?")
        cfg = "H{:.2f}_M{:.2f}".format(
            r["config"]["head_ratio"], r["config"]["mlp_ratio"])
        if r["config"]["skip_blocks"]:
            cfg += "_skip{}".format(r["config"]["skip_blocks"])
        if r["config"]["quantize"]:
            cfg += "_Q"

        b = r["baseline"]
        p = r.get("pruned", b)
        q = r.get("quantized", p)
        print(row_fmt.format(model, cfg, q["params_m"], q["size_mb"],
                             q["latency_ms"], q["throughput"]))


def main():
    results = collect_results()
    print("Found {} experiments".format(len(results)))
    print("")
    print_table(results)


if __name__ == "__main__":
    main()
