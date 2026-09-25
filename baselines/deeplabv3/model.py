"""DeepLabV3 baseline for traditional village remote sensing semantic segmentation.

Reference:
    Chen et al. "Rethinking Atrous Convolution for Semantic Image Segmentation."
    arXiv:1706.05587, 2017.
"""

import torch
import torch.nn as nn
import torchvision.models.segmentation as seg
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabV3(nn.Module):
    """DeepLabV3 with ResNet-50 backbone and ASPP (Atrous Spatial Pyramid Pooling)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        self.num_classes = num_classes

        # We construct ResNet-50 backbone with dilation
        if pretrained:
            self.model = seg.deeplabv3_resnet50(
                weights=seg.DeepLabV3_ResNet50_Weights.DEFAULT,
                num_classes=num_classes
            )
        else:
            # Random initialization, no network fetch
            self.model = seg.deeplabv3_resnet50(
                weights=None,
                weights_backbone=None,
                num_classes=num_classes,
                aux_loss=False
            )

        # In case num_classes differs from default or custom channel count
        if in_channels != 3:
            old_conv = self.model.backbone.conv1
            self.model.backbone.conv1 = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        # torchvision segmentation returns a dict: {'out': Tensor, 'aux': Tensor}
        if isinstance(out, dict):
            return out["out"]
        return out
