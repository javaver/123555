"""PSPNet baseline for traditional village remote sensing semantic segmentation.

Reference:
    Zhao et al. "Pyramid Scene Parsing Network." CVPR 2017.

Architecture:
  - Encoder: ResNet-50 (timm, features_only), output_stride=16 —— 与论文一致的
    膨胀 (dilated) 第 4 阶段设置, 也与 DeepLabV3 基线的空洞卷积口径对齐;
  - Pyramid Pooling Module (PPM): 对末级特征并行做 bins={1,2,3,6} 自适应平均
    池化 + 1x1 卷积降维, 上采样后与原特征拼接, 聚合全局/多尺度上下文;
  - Head: 3x3 卷积融合 + Dropout + 1x1 分类器, 双线性上采样回原尺寸。

注: 论文的辅助分类头 (aux loss, 0.4 权重) 未纳入, 保持与其余基线统一的
    (B, K, H, W) 逐像素 logits 训练/评估接口。
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PyramidPoolingModule(nn.Module):
    """PPM: bins={1,2,3,6} 并行多尺度上下文聚合 (论文 Fig.4)。"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int = 512,
        bins: Tuple[int, ...] = (1, 2, 3, 6),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(b),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
                for b in bins
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch + out_ch * len(bins), out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [x]
        for stage in self.stages:
            y = stage(x)
            if y.shape[-2:] != (h, w):
                y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            outs.append(y)
        return self.bottleneck(torch.cat(outs, dim=1))


class PSPNet(nn.Module):
    """PSPNet: Dilated ResNet-50 + Pyramid Pooling Module."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        backbone: str = "resnet50",
        embed_dim: int = 512,
        output_stride: int = 16,
        pretrained: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=in_channels,
            output_stride=output_stride,
        )
        feat_info = self.encoder.feature_info.channels()
        self.ppm = PyramidPoolingModule(in_ch=feat_info[-1], out_ch=embed_dim)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = x.shape[-2:]
        feats = self.encoder(x)  # (1/4, 1/8, 1/16, 1/16*) 末级带空洞卷积
        y = self.ppm(feats[-1])
        logits = self.classifier(y)
        if logits.shape[-2:] != hw:
            logits = F.interpolate(logits, size=hw, mode="bilinear", align_corners=False)
        return logits
