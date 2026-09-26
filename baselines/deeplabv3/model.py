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
            # torchvision 硬校验: 传 COCO 权重时 num_classes 必须等于 21 (COCO 类数),
            # 不能直接传本任务类数。标准做法: 先按 21 类构建并完整加载 COCO 预训练
            # (骨干 + ASPP 全部迁移), 再把最后的 1x1 分类头移植为本任务类数。
            # 权重文件 deeplabv3_resnet50_coco-cd0a2569.pth 命中 ~/.cache/torch/hub/checkpoints/ 即离线加载。
            self.model = seg.deeplabv3_resnet50(
                weights=seg.DeepLabV3_ResNet50_Weights.DEFAULT,
                num_classes=21,
            )
            self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
            # torchvision 带 COCO 权重时强制 aux_loss=True (加载需要); 但本训练不用
            # 辅助损失, 移植后整体摘除 aux 头 —— 保证 state_dict 键与评测端
            # pretrained=False (aux_loss=False) 重建完全一致, 评估加载不炸。
            self.model.aux_classifier = None
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
