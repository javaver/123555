"""SegFormer baseline for traditional village remote sensing semantic segmentation.

Reference:
    Xie et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers."
    NeurIPS 2021.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SegFormerMLPHead(nn.Module):
    """All-MLP Decoder for SegFormer."""

    def __init__(
        self,
        in_channels: List[int],
        embed_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.linear_c = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for ch in in_channels
        ])
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor], out_hw: Tuple[int, int]) -> torch.Tensor:
        # features: 4 levels (H/4, H/8, H/16, H/32)
        target_hw = features[0].shape[-2:]
        outs = []
        for i, f in enumerate(features):
            proj = self.linear_c[i](f)
            if proj.shape[-2:] != target_hw:
                proj = F.interpolate(proj, size=target_hw, mode="bilinear", align_corners=False)
            outs.append(proj)
        fused = self.linear_fuse(torch.cat(outs, dim=1))
        logits = self.classifier(fused)
        if logits.shape[-2:] != out_hw:
            logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        return logits


class SegFormer(nn.Module):
    """SegFormer: Hierarchical Transformer Encoder + All-MLP Decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        backbone: str = "pvt_v2_b0",
        embed_dim: int = 256,
        pretrained: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=in_channels,
        )
        feat_info = self.encoder.feature_info.channels()
        self.decoder = SegFormerMLPHead(
            in_channels=feat_info,
            embed_dim=embed_dim,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = x.shape[-2:]
        feats = self.encoder(x)
        return self.decoder(feats, hw)
