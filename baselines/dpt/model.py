"""DPT (Dense Prediction Transformer) baseline for remote sensing semantic segmentation.

Reference:
    Ranftl et al. "Vision Transformers for Dense Prediction."
    ICCV 2021.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ResidualConvUnit(nn.Module):
    """Residual Convolution Unit for DPT fusion."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class FeatureFusionBlock(nn.Module):
    """Feature Fusion Block with residual connections."""

    def __init__(self, channels: int):
        super().__init__()
        self.rcu1 = ResidualConvUnit(channels)
        self.rcu2 = ResidualConvUnit(channels)

    def forward(self, x: torch.Tensor, res: torch.Tensor = None) -> torch.Tensor:
        if res is not None:
            res = self.rcu1(res)
            if x.shape[-2:] != res.shape[-2:]:
                x = F.interpolate(x, size=res.shape[-2:], mode="bilinear", align_corners=False)
            x = x + res
        x = self.rcu2(x)
        return x


class DPTHead(nn.Module):
    """DPT Reassemble & Multi-scale Fusion Head."""

    def __init__(self, in_dim: int = 768, feat_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.in_dim = in_dim
        self.feat_dim = feat_dim

        # 4 reassemble projections (to 1/4, 1/8, 1/16, 1/32)
        self.reassemble0 = nn.Sequential(
            nn.Conv2d(in_dim, feat_dim, kernel_size=1),
            nn.ConvTranspose2d(feat_dim, feat_dim, kernel_size=2, stride=2),
            nn.ConvTranspose2d(feat_dim, feat_dim, kernel_size=2, stride=2),
        )  # -> 1/4
        self.reassemble1 = nn.Sequential(
            nn.Conv2d(in_dim, feat_dim, kernel_size=1),
            nn.ConvTranspose2d(feat_dim, feat_dim, kernel_size=2, stride=2),
        )  # -> 1/8
        self.reassemble2 = nn.Conv2d(in_dim, feat_dim, kernel_size=1)  # -> 1/16
        self.reassemble3 = nn.Sequential(
            nn.Conv2d(in_dim, feat_dim, kernel_size=1),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=2, padding=1),
        )  # -> 1/32

        self.fusion3 = FeatureFusionBlock(feat_dim)
        self.fusion2 = FeatureFusionBlock(feat_dim)
        self.fusion1 = FeatureFusionBlock(feat_dim)
        self.fusion0 = FeatureFusionBlock(feat_dim)

        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, num_classes, kernel_size=1),
        )

    def forward(self, layers: List[torch.Tensor], hw: Tuple[int, int]) -> torch.Tensor:
        b = layers[0].shape[0]
        h_p, w_p = hw[0] // 16, hw[1] // 16

        def to_2d(tokens: torch.Tensor) -> torch.Tensor:
            # tokens: (B, N, D), strip CLS token if present
            if tokens.shape[1] == h_p * w_p + 1:
                t = tokens[:, 1:, :]
            else:
                t = tokens
            return t.transpose(1, 2).reshape(b, self.in_dim, h_p, w_p)

        r0 = self.reassemble0(to_2d(layers[0]))
        r1 = self.reassemble1(to_2d(layers[1]))
        r2 = self.reassemble2(to_2d(layers[2]))
        r3 = self.reassemble3(to_2d(layers[3]))

        f3 = self.fusion3(r3)
        f2 = self.fusion2(f3, r2)
        f1 = self.fusion1(f2, r1)
        f0 = self.fusion0(f1, r0)

        out = self.head(f0)
        if out.shape[-2:] != hw:
            out = F.interpolate(out, size=hw, mode="bilinear", align_corners=False)
        return out


class DPT(nn.Module):
    """Dense Prediction Transformer with ViT backbone."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            dynamic_img_size=True,
        )
        embed_dim = self.encoder.embed_dim
        self.head = DPTHead(in_dim=embed_dim, feat_dim=256, num_classes=num_classes)
        # Extract layers at 4 stages (e.g. layers 2, 5, 8, 11 for 12-layer ViT)
        n_blocks = len(self.encoder.blocks)
        self.selected_indices = {
            n_blocks // 4 - 1,
            2 * (n_blocks // 4) - 1,
            3 * (n_blocks // 4) - 1,
            n_blocks - 1,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = x.shape[-2:]
        # ViT forward extraction
        t = self.encoder.patch_embed(x)
        t = self.encoder._pos_embed(t)
        t = self.encoder.norm_pre(t)

        extracted = []
        for idx, blk in enumerate(self.encoder.blocks):
            t = blk(t)
            if idx in self.selected_indices:
                extracted.append(t)

        return self.head(extracted, hw)
