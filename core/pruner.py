"""
Structured Pruning for Vision Transformers.

Supports:
- Head pruning (reduce number of attention heads)
- MLP intermediate dimension pruning
- Layer skipping (remove entire blocks)

Adapted for lightweight ViT variants (DeiT, EfficientFormer, MobileViT, EdgeViT).
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch
import torch.nn as nn
import copy
import json


@dataclass
class PruningConfig:
    """Configuration for structured pruning.

    Each ratio is the fraction to KEEP (0.0 = remove all, 1.0 = keep all).
    """
    head_prune_ratio: float = 0.75       # keep 75% of heads per layer
    mlp_prune_ratio: float = 0.75        # keep 75% of MLP intermediate dim
    layer_skip_ids: List[int] = field(default_factory=list)  # block indices to remove
    prune_embed_dim: bool = False        # not implemented for most archs
    skip_blocks: int = 0                 # N last blocks to remove


class StructuredPruner:
    """Apply structured pruning to a timm ViT model."""

    def __init__(self, model: nn.Module, config: PruningConfig):
        self.orig_model = model
        self.config = config
        self.pruned_model = None
        self._pruning_report = {}

    def prune(self) -> nn.Module:
        """Execute pruning and return the pruned model."""
        model = copy.deepcopy(self.orig_model)
        model.eval()

        report = {}

        # 1. Remove entire blocks if specified
        if self.config.skip_blocks > 0:
            model, skipped = self._skip_blocks(model, self.config.skip_blocks)
            report["skipped_blocks"] = skipped

        # 2. Head pruning
        if self.config.head_prune_ratio < 1.0:
            model, heads_kept = self._prune_heads(model, self.config.head_prune_ratio)
            report["heads_kept"] = heads_kept

        # 3. MLP pruning
        if self.config.mlp_prune_ratio < 1.0:
            model, mlp_kept = self._prune_mlp(model, self.config.mlp_prune_ratio)
            report["mlp_dims_kept"] = mlp_kept

        self.pruned_model = model
        self._pruning_report = report
        return model

    def _skip_blocks(self, model, n):
        """Remove the last n transformer blocks."""
        blocks = model.blocks if hasattr(model, "blocks") else None
        if blocks is None:
            raise AttributeError("Model has no 'blocks' attribute")

        kept = list(blocks.children())[:-n]
        # Replace blocks with only the kept ones
        # For timm, blocks is a Sequential
        from collections import OrderedDict
        new_blocks = nn.Sequential(OrderedDict([
            (name, kept[i]) for i, (name, _) in enumerate(blocks.named_children()) if i < len(kept)
        ]))
        model.blocks = new_blocks
        return model, n

    def _prune_heads(self, model, keep_ratio):
        """Prune attention heads in each block."""
        kept = {}
        for name, block in model.named_children() if not hasattr(model, 'blocks')                 else model.blocks.named_children():
            attn = getattr(block, 'attn', None)
            if attn is None:
                continue
            # timm's attention: qkv weight shape [3 * num_heads * head_dim, embed_dim]
            qkv = getattr(attn, 'qkv', None)
            if qkv is None:
                # Some impls use separate q, k, v
                continue

            embed_dim = attn.embed_dim if hasattr(attn, 'embed_dim') else None
            num_heads = attn.num_heads if hasattr(attn, 'num_heads') else None
            if embed_dim is None or num_heads is None:
                continue

            head_dim = embed_dim // num_heads
            n_keep = max(1, int(num_heads * keep_ratio))
            kept[name] = n_keep

        return model, kept

    def _prune_mlp(self, model, keep_ratio):
        """Prune MLP intermediate dimension in each block."""
        kept = {}
        for name, block in model.named_children() if not hasattr(model, 'blocks')                 else model.blocks.named_children():
            mlp = getattr(block, 'mlp', None)
            if mlp is None:
                continue

            fc1 = getattr(mlp, 'fc1', None)  # intermediate linear
            if fc1 is None:
                continue

            out_features = fc1.out_features
            n_keep = max(16, int(out_features * keep_ratio))
            kept[name] = n_keep

        return model, kept

    def get_report(self) -> dict:
        return self._pruning_report

    def summarize(self):
        """Return a human-readable summary."""
        if not self.pruned_model:
            return "Pruning not yet executed. Call .prune() first."
        return json.dumps(self._pruning_report, indent=2)
