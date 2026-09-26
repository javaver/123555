"""SegFormer baseline for traditional village remote sensing semantic segmentation.

Reference:
    Xie et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers."
    NeurIPS 2021.

Architecture (faithful to the paper):
  - Encoder: Mix Transformer (MiT-B0 .. MiT-B5), implemented in-repo
    (``baselines/segformer/mit.py``) because current timm releases ship no
    SegFormer models. Weights are key-compatible with the official NVlabs
    ``mit_bX.pth`` ImageNet checkpoints (pass a local path via
    ``pretrained_from``).
  - Decoder: All-MLP decoder (linear fuse of the 4 pyramid stages at 1/4
    resolution, then a 1x1 classifier).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.segformer.mit import MIT_CONFIGS, mit_backbone


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
    """SegFormer: Hierarchical MiT Encoder + All-MLP Decoder.

    Args:
        backbone: ``mit_b0`` .. ``mit_b5`` (in-repo Mix Transformer, default).
            Any other string is passed through to ``timm.create_model(...,
            features_only=True)`` for compatibility with CNN/other backbones
            (e.g. legacy ``pvt_v2_b0`` checkpoints).
        pretrained: load ImageNet weights for timm backbones (MiT 的
            ImageNet 权重请用 ``pretrained_from`` 传入本地官方 mit_bX.pth).
        pretrained_from: local checkpoint path with official NVlabs
            ``mit_bX.pth`` weights to initialise the MiT encoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        backbone: str = "mit_b0",
        embed_dim: int = 256,
        pretrained: bool = False,
        pretrained_from: str | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone

        key = backbone.lower().replace("-", "_").strip()
        if key in MIT_CONFIGS:
            self.encoder = mit_backbone(key, in_chans=in_channels)
            feat_info = self.encoder.channels()
            if pretrained_from is not None:
                state = torch.load(Path(pretrained_from), map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                missing, unexpected = self.encoder.load_state_dict(state, strict=False)
                # 官方 ckpt 只含编码器权重, missing 属预期; 意外键则报错
                unexpected = [k for k in unexpected if not k.startswith(("head", "decode_head", "fc", "classifier"))]
                if unexpected:
                    raise RuntimeError(f"Unexpected keys in MiT checkpoint: {unexpected[:5]}")
        else:
            import timm  # 透传 timm 注册骨干 (兼容 pvt_v2_b0 等旧配置)

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
