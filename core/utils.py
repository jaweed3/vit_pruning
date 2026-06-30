"""
Utility functions for experiment management, logging, and helpers.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import torch


def save_metrics(metrics: Dict[str, Any],
                 path: str,
                 filename: Optional[str] = None):
    """Save metrics to a JSON file with timestamp."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    metrics["_timestamp"] = datetime.now().isoformat()
    with open(path / filename, "w") as f:
        json.dump(metrics, f, indent=2)

    return str(path / filename)


def load_metrics(path: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(path) as f:
        return json.load(f)


def save_checkpoint(model: torch.nn.Module,
                    path: str,
                    extra: Dict[str, Any] = None):
    """Save model checkpoint."""
    state = {"model_state_dict": model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(model: torch.nn.Module,
                    path: str,
                    device: str = "cpu"):
    """Load model checkpoint."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    return model


def format_size(n_bytes: float) -> str:
    """Format byte size to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}TB"
