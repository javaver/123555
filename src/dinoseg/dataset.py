"""瓦片数据集: 读 datasets/images + masks, 训练增强 (翻转/90°旋转/光度抖动)。

只依赖 PIL/numpy/torch, 不依赖 torchvision。
光度统计只在非 ignore 像素上无意义于整图增强, 故用整图标准增强即可
(padding 近黑且被 ignore, 不参与损失, 不影响训练)。
"""
from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_array(img: Image.Image) -> torch.Tensor:
    a = np.asarray(img).astype(np.float32) / 255.0
    a = (a - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(a.transpose(2, 0, 1)).float()


class TileDataset(Dataset):
    def __init__(self, root, rows, aug: bool = False,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD, seed: int = 42):
        self.root = root
        self.rows = rows
        self.aug = aug
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.root / r["image_rel"]).convert("RGB")
        mask = Image.open(self.root / r["mask_rel"])
        a = np.asarray(img).astype(np.float32) / 255.0
        m = np.asarray(mask, dtype=np.int64)

        if self.aug:
            if self.rng.random() < 0.5:
                a, m = a[:, ::-1], m[:, ::-1]
            if self.rng.random() < 0.5:
                a, m = a[::-1], m[::-1]
            k = self.rng.choice([0, 1, 2, 3])
            if k:
                a, m = np.rot90(a, k), np.rot90(m, k)
            img2 = Image.fromarray((a * 255).astype(np.uint8))
            if self.rng.random() < 0.5:
                img2 = ImageEnhance.Brightness(img2).enhance(self.rng.uniform(0.8, 1.2))
            if self.rng.random() < 0.5:
                img2 = ImageEnhance.Contrast(img2).enhance(self.rng.uniform(0.8, 1.2))
            a = np.asarray(img2).astype(np.float32) / 255.0
            a = np.ascontiguousarray(a)
        m = np.ascontiguousarray(m)

        a = (a - self.mean) / self.std
        return (torch.from_numpy(a.transpose(2, 0, 1)).float(),
                torch.from_numpy(m).long())


class FeatureDataset(Dataset):
    """读预计算的编码器特征 (cache_features.py 产出) + 掩膜。

    增强只做几何类 (翻转/90°旋转), 对特征图与掩膜同步施加;
    光度抖动仅在 TileDataset(端到端模式) 可用。
    """

    def __init__(self, root, rows, feats_dir, aug: bool = False, seed: int = 42):
        self.root = root
        self.rows = rows
        self.feats_dir = feats_dir
        self.aug = aug
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        feats = []
        for f in torch.load(self.feats_dir / f"{r['village']}__{r['tile']}.pt",
                            map_location="cpu", weights_only=True):
            if f.dim() == 4:  # 兼容早期带 batch 维的缓存
                f = f[0]
            feats.append(f.float())
        m = np.asarray(Image.open(self.root / r["mask_rel"]), dtype=np.int64)
        if self.aug:
            m = torch.from_numpy(m)
            if self.rng.random() < 0.5:
                feats = [f.flip(-1) for f in feats]
                m = m.flip(1)
            if self.rng.random() < 0.5:
                feats = [f.flip(-2) for f in feats]
                m = m.flip(0)
            k = self.rng.choice([0, 1, 2, 3])
            if k:
                feats = [torch.rot90(f, k, (-2, -1)) for f in feats]
                m = torch.rot90(m, k, (-2, -1))
            m = m.numpy()
        return feats, torch.from_numpy(np.ascontiguousarray(m)).long()
