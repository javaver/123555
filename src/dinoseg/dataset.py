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
