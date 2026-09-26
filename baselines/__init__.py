"""Baseline models for traditional village remote sensing segmentation."""

from typing import Dict, Any
import torch.nn as nn

from baselines.unet.model import UNet
from baselines.deeplabv3.model import DeepLabV3
from baselines.pspnet.model import PSPNet
from baselines.segformer.model import SegFormer
from baselines.maskformer.model import MaskFormer

BASELINE_MODELS = {
    "unet": UNet,
    "deeplabv3": DeepLabV3,
    "pspnet": PSPNet,
    "segformer": SegFormer,
    "maskformer": MaskFormer,
}

# 掩膜分类范式基线: 训练时输出 (类别 logits, 掩膜 logits) 并使用
# 二分图匹配 (匈牙利算法) 损失, 而非逐像素 CE+Dice (见 baselines/maskformer/criterion.py)
MASK_CLASSIFICATION_BASELINES = {"maskformer"}


def get_baseline_model(
    name: str,
    in_channels: int = 3,
    num_classes: int = 10,
    pretrained: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate a baseline model by name."""
    key = name.lower().strip()
    if key not in BASELINE_MODELS:
        raise ValueError(
            f"Unknown baseline model: {name}. Available models: {list(BASELINE_MODELS.keys())}"
        )
    cls = BASELINE_MODELS[key]
    if key == "unet":
        return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
    return cls(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained, **kwargs)
