"""
Model loader — load lightweight ViT variants via timm.
"""

import torch
import timm

SUPPORTED_MODELS = {
    "deit_tiny":  {"desc": "DeiT-Tiny (5.7M params)", "default": True, "timm_name": "deit_tiny_patch16_224"},
    "deit_small": {"desc": "DeiT-Small (22M params)", "default": False, "timm_name": "deit_small_patch16_224"},
    "efficientformer_l1": {"desc": "EfficientFormer-L1 (12M params)", "default": True, "timm_name": "efficientformer_l1"},
    "mobilevit_s": {"desc": "MobileViT-S (5.6M params)", "default": True, "timm_name": "mobilevit_s"},
    "mobilevit_xs": {"desc": "MobileViT-XS (2.3M params)", "default": False, "timm_name": "mobilevit_xs"},
}


def load_model(name: str, pretrained: bool = True, device: str = "auto"):
    """Load a lightweight ViT model by name.

    Args:
        name: Model name from SUPPORTED_MODELS.
        pretrained: Load ImageNet-1k pretrained weights.
        device: 'auto' -> cuda if available else cpu.

    Returns:
        model (nn.Module), device (str)
    """
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model {name}. Options: {list(SUPPORTED_MODELS)}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    timm_name = SUPPORTED_MODELS[name]["timm_name"]
    model = timm.create_model(timm_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    return model, device


def list_available_models():
    """Return list of supported model names."""
    return list(SUPPORTED_MODELS)
