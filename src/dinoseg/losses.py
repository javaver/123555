"""CE(ignore=255, 类权重) + 逐类 soft Dice, 1:1。"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IGNORE = 255


class SegLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        valid = target != IGNORE
        if valid.sum() == 0:
            return ce
        probs = logits.softmax(dim=1)
        dice_acc = []
        num_classes = logits.shape[1]
        for c in range(num_classes):
            p = probs[:, c][valid]
            g = (target == c)[valid].float()
            if g.sum() == 0 and p.sum() == 0:
                continue
            inter = (p * g).sum()
            dice_acc.append((2 * inter + 1) / (p.sum() + g.sum() + 1))
        dice = 1 - torch.stack(dice_acc).mean() if dice_acc else logits.new_zeros(())
        return ce + dice
