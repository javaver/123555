"""MaskFormer baseline for traditional village remote sensing semantic segmentation.

Reference:
    Cheng et al. "Per-Pixel Classification is Not All You Need for Semantic Segmentation."
    NeurIPS 2021.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PixelDecoder(nn.Module):
    """Multi-scale feature pyramid pixel decoder."""

    def __init__(self, in_channels: List[int], out_dim: int = 256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # feats: 1/4, 1/8, 1/16, 1/32
        p0 = self.projs[0](feats[0])
        target_hw = p0.shape[-2:]
        acc = p0
        for i in range(1, len(feats)):
            p = self.projs[i](feats[i])
            acc = acc + F.interpolate(p, size=target_hw, mode="bilinear", align_corners=False)
        return self.fuse(acc)


class TransformerMaskDecoder(nn.Module):
    """Transformer decoder with learnable queries predicting mask and class embeddings."""

    def __init__(
        self,
        num_queries: int = 50,
        embed_dim: int = 256,
        num_classes: int = 10,
        num_layers: int = 3,
        nhead: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Class head predicts (num_classes + 1) where +1 is no-object / background
        self.class_head = nn.Linear(embed_dim, num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # memory: (B, C, H, W)
        b, c, h, w = memory.shape
        mem_flat = memory.flatten(2).transpose(1, 2)  # (B, H*W, C)
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # (B, N, C)

        out_q = self.decoder(q, mem_flat)  # (B, N, C)

        pred_classes = self.class_head(out_q)  # (B, N, K+1)
        mask_embeds = self.mask_embed(out_q)  # (B, N, C)

        # Mask dot product: (B, N, C) x (B, C, H*W) -> (B, N, H*W) -> (B, N, H, W)
        pred_masks = torch.bmm(mask_embeds, memory.flatten(2)).view(b, self.num_queries, h, w)
        return pred_classes, pred_masks


class MaskFormer(nn.Module):
    """MaskFormer: Mask-Classification Architecture for Semantic Segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        backbone: str = "resnet50",
        embed_dim: int = 256,
        num_queries: int = 50,
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
        # Use stages 1..4 (1/4, 1/8, 1/16, 1/32)
        all_channels = self.encoder.feature_info.channels()
        used_channels = all_channels[1:] if len(all_channels) > 4 else all_channels

        self.pixel_decoder = PixelDecoder(in_channels=used_channels, out_dim=embed_dim)
        self.mask_decoder = TransformerMaskDecoder(
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_hw = x.shape[-2:]
        feats = self.encoder(x)
        used_feats = feats[1:] if len(feats) > 4 else feats
        pixel_embeds = self.pixel_decoder(used_feats)  # (B, C, H/4, W/4)

        pred_classes, pred_masks = self.mask_decoder(pixel_embeds)
        # pred_classes: (B, N, K+1), pred_masks: (B, N, H/4, W/4)

        # Semantic assembly: P(c, h, w) = sum_q P(c|q) * sigmoid(M_q(h, w))
        class_probs = F.softmax(pred_classes, dim=-1)[..., : self.num_classes]  # (B, N, K)
        mask_probs = torch.sigmoid(pred_masks)  # (B, N, H/4, W/4)

        b, n, h_sub, w_sub = mask_probs.shape
        # (B, K, N) x (B, N, H_sub*W_sub) -> (B, K, H_sub*W_sub)
        sem_probs = torch.bmm(class_probs.transpose(1, 2), mask_probs.flatten(2)).view(
            b, self.num_classes, h_sub, w_sub
        )

        # Convert normalized probabilities to logits: log(P + eps)
        sem_logits = torch.log(sem_probs + 1e-7)

        if sem_logits.shape[-2:] != orig_hw:
            sem_logits = F.interpolate(sem_logits, size=orig_hw, mode="bilinear", align_corners=False)
        return sem_logits
