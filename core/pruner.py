"""
Structured Pruning for Vision Transformers.

Supports:
- Head pruning (reduce number of attention heads) — actually modifies qkv weights
- MLP intermediate dimension pruning — actually slices fc1/fc2
- Layer skipping (remove entire blocks)
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch
import torch.nn as nn
import copy
import json


@dataclass
class PruningConfig:
    """Fraction to KEEP (0.0 = remove all, 1.0 = keep all)."""
    head_prune_ratio: float = 1.0
    mlp_prune_ratio: float = 1.0
    layer_skip_ids: List[int] = field(default_factory=list)
    skip_blocks: int = 0


class StructuredPruner:
    """Apply structured pruning to a timm ViT model — actually modifies weights."""

    def __init__(self, model: nn.Module, config: PruningConfig):
        self.orig_model = model
        self.config = config
        self.pruned_model = None
        self._pruning_report = {}

    def prune(self) -> nn.Module:
        model = copy.deepcopy(self.orig_model)
        model.eval()
        report = {}

        if self.config.skip_blocks > 0:
            model, n_skipped = self._skip_blocks(model, self.config.skip_blocks)
            report["blocks_skipped"] = n_skipped

        if self.config.head_prune_ratio < 1.0:
            model, heads = self._prune_heads(model, self.config.head_prune_ratio)
            report["heads_kept"] = heads

        if self.config.mlp_prune_ratio < 1.0:
            model, mlp = self._prune_mlp(model, self.config.mlp_prune_ratio)
            report["mlp_dims_kept"] = mlp

        self.pruned_model = model
        self._pruning_report = report
        return model

    def _skip_blocks(self, model, n):
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise AttributeError("Model has no 'blocks' attr")
        from collections import OrderedDict
        children = list(blocks.named_children())
        new = nn.Sequential(OrderedDict(children[:-n]))
        model.blocks = new
        return model, n

    def _prune_heads(self, model, keep_ratio):
        """Actually prune attention heads — shrink qkv, proj, update num_heads."""
        kept = {}
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            return model, kept

        for name, block in blocks.named_children():
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            qkv = getattr(attn, "qkv", None)
            if qkv is None:
                continue

            # timm Attention uses num_heads + head_dim (not embed_dim)
            num_heads = getattr(attn, "num_heads", None)
            head_dim = getattr(attn, "head_dim", None)
            if num_heads is None or head_dim is None:
                continue

            embed_dim = num_heads * head_dim  # infer from qkv shape
            n_keep = max(1, int(num_heads * keep_ratio))
            kept[name] = n_keep

            # qkv.weight: [3 * num_heads * head_dim, embed_dim]
            W_qkv = qkv.weight.data
            new_qkv = nn.Linear(embed_dim, 3 * n_keep * head_dim, bias=qkv.bias is not None)
            W_reshaped = W_qkv.view(3, num_heads, head_dim, embed_dim)
            W_kept = W_reshaped[:, :n_keep, :, :].contiguous().view(3 * n_keep * head_dim, embed_dim)
            new_qkv.weight.data = W_kept
            if qkv.bias is not None:
                b_reshaped = qkv.bias.data.view(3, num_heads, head_dim)
                b_kept = b_reshaped[:, :n_keep, :].contiguous().view(3 * n_keep * head_dim)
                new_qkv.bias.data = b_kept
            attn.qkv = new_qkv

            # proj: [embed_dim, n_keep * head_dim] — slice columns
            proj = getattr(attn, "proj", None)
            if proj is not None:
                new_proj = nn.Linear(n_keep * head_dim, embed_dim, bias=proj.bias is not None)
                new_proj.weight.data = proj.weight.data[:, :n_keep * head_dim]
                if proj.bias is not None:
                    new_proj.bias.data = proj.bias.data
                attn.proj = new_proj

            attn.num_heads = n_keep
            attn.attn_dim = n_keep * head_dim

        return model, kept

    def _prune_mlp(self, model, keep_ratio):
        """Actually prune MLP intermediate dim — slice fc1/fc2."""
        kept = {}
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            return model, kept

        for name, block in blocks.named_children():
            mlp = getattr(block, "mlp", None)
            if mlp is None:
                continue

            fc1 = getattr(mlp, "fc1", None)
            fc2 = getattr(mlp, "fc2", None)
            if fc1 is None or fc2 is None:
                continue

            embed_dim = fc1.in_features
            inter_dim = fc1.out_features
            n_keep = max(16, int(inter_dim * keep_ratio))
            kept[name] = n_keep

            # fc1: [inter_dim, embed_dim] -> slice first n_keep rows
            new_fc1 = nn.Linear(embed_dim, n_keep, bias=fc1.bias is not None)
            new_fc1.weight.data = fc1.weight.data[:n_keep, :]
            if fc1.bias is not None:
                new_fc1.bias.data = fc1.bias.data[:n_keep]
            mlp.fc1 = new_fc1

            # fc2: [embed_dim, inter_dim] -> slice first n_keep columns
            new_fc2 = nn.Linear(n_keep, embed_dim, bias=fc2.bias is not None)
            new_fc2.weight.data = fc2.weight.data[:, :n_keep]
            if fc2.bias is not None:
                new_fc2.bias.data = fc2.bias.data
            mlp.fc2 = new_fc2

        return model, kept

    def get_report(self):
        return self._pruning_report

    def summarize(self):
        if not self.pruned_model:
            return "Pruning not yet executed. Call .prune() first."
        return json.dumps(self._pruning_report, indent=2)
