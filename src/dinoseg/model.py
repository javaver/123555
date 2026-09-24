"""DINO-Seg: 冻结 DINOv3 编码器 + 重采样块 + 融合上采样解码器 + 轻量分割头。

编码器经 timm 加载 (vit_*_patch16_dinov3.{lvd1689m,sat493m}), 完全冻结;
取最后 4 个 block 的 patch tokens (32x32 @512 输入), 丢弃 CLS+register tokens;
重采样块用亚像素卷积把单尺度特征重建为 {1/16,1/8,1/4,1/2} 金字塔;
融合解码器自顶向下逐级 ×2 上采样相加融合; 头上采样回全分辨率出 logits。
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_ENCODER = "vit_base_patch16_dinov3.lvd1689m"


class FrozenDINOv3Encoder(nn.Module):
    """冻结的 DINOv3 ViT, 返回最后 4 个 block 的 2D patch 特征。"""

    def __init__(self, model_id: str = DEFAULT_ENCODER, pretrained: bool = True,
                 checkpoint_path: str | None = None):
        super().__init__()
        try:
            self.enc = timm.create_model(
                model_id, pretrained=pretrained and not checkpoint_path,
                num_classes=0, checkpoint_path=checkpoint_path)
        except Exception as e:
            raise RuntimeError(
                f"加载编码器 {model_id} 失败: {e}\n"
                "若 HuggingFace 下载受阻, 可设 HF_ENDPOINT=https://hf-mirror.com 后重试; "
                "或用 --enc-ckpt 指向本地 model.safetensors; "
                "或先 --pretrained 0 验证管线再开预训练。") from e
        self.enc.eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.feat_dim = self.enc.embed_dim
        self.n_prefix = self.enc.num_prefix_tokens  # CLS + registers
        self.patch = self.enc.patch_embed.patch_size[0]
        self._feats: list[torch.Tensor] = []
        for blk in list(self.enc.blocks)[-4:]:
            blk.register_forward_hook(self._hook)

    def _hook(self, _m, _inp, out):
        self._feats.append(out)

    def train(self, mode: bool = True):  # 编码器永远 eval (冻结 BN/dropout 语义)
        return super().train(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._feats = []
        # 编码器前向用 fp16 加速(无 backward, 不会溢出); 解码器/反向全程 fp32,
        # 规避 fp16 反向经 ~1e3 量级特征时 GradScaler 梯度溢出 -> NaN
        with torch.no_grad(), torch.amp.autocast("cuda"):
            self.enc(x)
        b = x.shape[0]
        h = w = x.shape[-1] // self.patch
        outs = []
        for f in self._feats:
            t = f[:, self.n_prefix:, :]
            outs.append(t.transpose(1, 2).reshape(b, self.feat_dim, h, w))
        return outs


class ResampleBlock(nn.Module):
    """单尺度特征 -> 指定倍率的上采样特征 (亚像素卷积, 学习式上采样)。"""

    def __init__(self, dim: int, ch: int, factor: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, ch, 1)
        ups = []
        for _ in range(int(math.log2(factor))):
            ups += [nn.Conv2d(ch, ch * 4, 3, padding=1), nn.PixelShuffle(2)]
        self.up = nn.Sequential(*ups)
        self.refine = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.up(self.proj(x)))


class FusionUpsampleDecoder(nn.Module):
    """自顶向下: ×2 上采样 + 与更细一级相加 + 3x3 卷积融合。"""

    def __init__(self, chans=(256, 192, 128, 96), mid: int = 256, out: int = 128):
        super().__init__()
        self.projs = nn.ModuleList(nn.Conv2d(c, mid, 1) for c in chans)
        self.fuse = nn.ModuleList(nn.Conv2d(mid, mid, 3, padding=1) for _ in range(len(chans) - 1))
        self.out_conv = nn.Conv2d(mid, out, 3, padding=1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = self.projs[0](feats[0])
        for i in range(1, len(feats)):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.fuse[i - 1](x + self.projs[i](feats[i]))
        return self.out_conv(x)


class DINOvSeg(nn.Module):
    def __init__(self, encoder_id: str = DEFAULT_ENCODER, pretrained: bool = True,
                 num_classes: int = 10, chans=(256, 192, 128, 96),
                 checkpoint_path: str | None = None):
        super().__init__()
        self.encoder = FrozenDINOv3Encoder(encoder_id, pretrained, checkpoint_path)
        dim = self.encoder.feat_dim
        self.resample = nn.ModuleList(
            ResampleBlock(dim, c, f) for c, f in zip(chans, (1, 2, 4, 8)))
        self.decoder = FusionUpsampleDecoder(chans)
        self.head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1))

    def forward_feats(self, feats: list[torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
        pyr = [blk(f) for blk, f in zip(self.resample, feats)]
        x = self.head(self.decoder(pyr))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [f.float() for f in self.encoder(x)]  # fp16 特征转 fp32 再进解码器
        with torch.amp.autocast("cuda", enabled=False):
            return self.forward_feats(feats, x.shape[-2:])

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)
