"""MaskFormer 训练准则: 二分图匹配 (匈牙利算法) 损失.

Reference:
    Cheng et al. "Per-Pixel Classification is Not All You Need for Semantic
    Segmentation." NeurIPS 2021. (mask_former/modeling/{matcher, criterion}.py)

流程:
  1. GT 逐类构建为 G 个 "段落" (class c, binary mask m_c), ignore=255 剔除;
  2. HungarianMatcher 以
         C = 1.0 * CE_class + 1.0 * BCE_mask + 1.0 * Dice
     为代价, 用 scipy.optimize.linear_sum_assignment 从 N 个 query 中选出
     与 G 个 GT 段落一一对应的最优匹配 (其余 query 匹配到 ∅ 无对象类);
  3. 损失 (与论文/官方实现一致):
         L = 2.0 * CE(类别, ∅ 权重 0.1) + 5.0 * BCE(掩膜) + 5.0 * Dice(掩膜),
     掩膜损失在 1/4 分辨率上按全部匹配对全局平均。
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

IGNORE = 255


def build_targets(
    target: torch.Tensor,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """(B, H, W) GT 标签 -> 每图 (labels (G,), masks (G, H, W)) 段落列表。"""
    targets: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for y in target:
        valid = y != IGNORE
        labels = torch.unique(y[valid])
        masks = (y.unsqueeze(0) == labels.view(-1, 1, 1)).float()  # (G, H, W)
        targets.append((labels, masks))
    return targets


def _pairwise_bce_cost(
    pred: torch.Tensor, gt: torch.Tensor
) -> torch.Tensor:
    """逐对掩膜 BCE 代价. pred: (N, h, w) logits, gt: (G, h, w) -> (N, G)."""
    n, g = pred.shape[0], gt.shape[0]
    p = pred[:, None].expand(n, g, -1, -1)
    t = gt[None].expand(n, g, -1, -1)
    return F.binary_cross_entropy_with_logits(p, t, reduction="none").mean(dim=(-2, -1))


def _pairwise_dice_cost(
    pred: torch.Tensor, gt: torch.Tensor
) -> torch.Tensor:
    """逐对掩膜 Dice 代价. pred: (N, h, w) logits, gt: (G, h, w) -> (N, G)."""
    p = pred.sigmoid().flatten(1)  # (N, hw)
    t = gt.flatten(1)  # (G, hw)
    inter = (p[:, None] * t[None]).sum(dim=-1)  # (N, G)
    union = p.sum(dim=-1)[:, None] + t.sum(dim=-1)[None]
    return 1.0 - (2.0 * inter + 1.0) / (union + 1.0)


class HungarianMatcher(nn.Module):
    """二分图匹配器: 为每个 GT 段落分配唯一的最优 query。"""

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,  # (B, N, K+1)
        pred_masks: torch.Tensor,  # (B, N, h, w)
        targets: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Tuple[List[int], List[int]]]:
        """返回每图 (query 索引列表, GT 段落索引列表), 长度均为 G_i (未匹配 query -> ∅)。"""
        pred_probs = pred_logits.detach().float().softmax(dim=-1)  # (B, N, K+1)
        mask_hw = pred_masks.shape[-2:]

        indices: List[Tuple[List[int], List[int]]] = []
        for b, (labels, gt_masks) in enumerate(targets):
            if labels.numel() == 0:
                indices.append(([], []))
                continue
            # GT 掩膜对齐到预测分辨率 (nearest 保持二值)
            if gt_masks.shape[-2:] != mask_hw:
                gt_masks = F.interpolate(
                    gt_masks.unsqueeze(1), size=mask_hw, mode="nearest"
                ).squeeze(1)
            cost_class = -pred_probs[b][:, labels]  # (N, G)
            cost_mask = _pairwise_bce_cost(pred_masks[b].detach().float(), gt_masks)
            cost_dice = _pairwise_dice_cost(pred_masks[b].detach().float(), gt_masks)
            cost = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            ).cpu().numpy()
            row, col = linear_sum_assignment(cost)  # (G,) query idx, (G,) GT idx
            indices.append((row.tolist(), col.tolist()))
        return indices


class MaskFormerCriterion(nn.Module):
    """MaskFormer 二分图匹配损失 (类别 CE + 掩膜 BCE + Dice)。"""

    def __init__(
        self,
        num_classes: int,
        class_weight: torch.Tensor | None = None,
        ce_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        no_object_weight: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.matcher = HungarianMatcher()

        # 类别 CE 权重: 真类沿用类别均衡权重 (∅ 追加在末位, 权重 0.1)
        if class_weight is None:
            class_weight = torch.ones(num_classes)
        else:
            class_weight = class_weight.detach().float().cpu()
        ce_vec = torch.cat([class_weight, torch.tensor([no_object_weight])])
        self.register_buffer("ce_class_weight", ce_vec)

    def forward(
        self,
        pred_logits: torch.Tensor,  # (B, N, K+1)
        pred_masks: torch.Tensor,  # (B, N, h, w)
        target: torch.Tensor,  # (B, H, W) 值域 0..K-1 / 255
    ) -> Tuple[torch.Tensor, dict]:
        targets = build_targets(target)
        indices = self.matcher(pred_logits, pred_masks, targets)

        # ---- 1) 分类损失: 匹配 query -> GT 类, 未匹配 -> ∅ (index K) ----
        b, n = pred_logits.shape[:2]
        tgt_classes = pred_logits.new_full((b, n), self.num_classes, dtype=torch.long)
        for i, (q_idx, g_idx) in enumerate(indices):
            if len(q_idx):
                tgt_classes[i, torch.as_tensor(q_idx, device=tgt_classes.device)] = targets[i][0][
                    torch.as_tensor(g_idx, device=tgt_classes.device)
                ]
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2), tgt_classes, weight=self.ce_class_weight
        )

        # ---- 2) 掩膜损失: 匹配对上 BCE + Dice (1/4 分辨率, 全局平均) ----
        mask_hw = pred_masks.shape[-2:]
        bce_terms: List[torch.Tensor] = []
        dice_terms: List[torch.Tensor] = []
        for i, (q_idx, g_idx) in enumerate(indices):
            if len(q_idx) == 0:
                continue
            q = torch.as_tensor(q_idx, device=pred_masks.device)
            g = torch.as_tensor(g_idx, device=pred_masks.device)
            pm = pred_masks[i, q]  # (G, h, w)
            gm = targets[i][1][g].to(pred_masks.device)  # (G, H, W)
            if gm.shape[-2:] != mask_hw:
                gm = F.interpolate(gm.unsqueeze(1), size=mask_hw, mode="nearest").squeeze(1)
            bce_terms.append(
                F.binary_cross_entropy_with_logits(pm, gm, reduction="none")
                .mean(dim=(-2, -1))
                .sum()
            )
            p = pm.sigmoid().flatten(1)
            t = gm.flatten(1)
            inter = (p * t).sum(dim=-1)
            union = p.sum(dim=-1) + t.sum(dim=-1)
            dice_terms.append((1.0 - (2.0 * inter + 1.0) / (union + 1.0)).sum())

        num_matched = sum(len(q) for q, _ in indices)
        if num_matched > 0:
            loss_bce = torch.stack(bce_terms).sum() / num_matched
            loss_dice = torch.stack(dice_terms).sum() / num_matched
        else:
            loss_bce = pred_logits.new_zeros(())
            loss_dice = pred_logits.new_zeros(())

        loss = (
            self.ce_weight * loss_ce
            + self.mask_weight * loss_bce
            + self.dice_weight * loss_dice
        )
        return loss, {
            "loss_ce": float(loss_ce.detach()),
            "loss_mask_bce": float(loss_bce.detach()),
            "loss_mask_dice": float(loss_dice.detach()),
            "num_matched": num_matched,
        }
