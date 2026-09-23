"""混淆矩阵口径的 P/R/F1/IoU (ignore=255 全程剔除), 与论文口径对齐。"""
from __future__ import annotations

import numpy as np

IGNORE = 255


class ConfusionMeter:
    def __init__(self, num_classes: int):
        self.k = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)  # [gt, pred]

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        valid = gt != IGNORE
        p, g = pred[valid], gt[valid]
        np.add.at(self.mat, (g, p), 1)

    def result(self) -> dict:
        mat = self.mat
        tp = np.diag(mat)
        gt_n = mat.sum(axis=1) + 1e-9
        pred_n = mat.sum(axis=0) + 1e-9
        rec = tp / gt_n
        prec = tp / pred_n
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        iou = tp / np.clip(mat.sum(axis=1) + mat.sum(axis=0) - tp, 1e-9, None)
        # 宏平均: 只统计真实出现过的类 (gt>0), 与论文常见口径一致
        present = mat.sum(axis=1) > 0
        macro = lambda a: float(a[present].mean()) if present.any() else 0.0
        tot = mat.sum()
        return {
            "precision": prec, "recall": rec, "f1": f1, "iou": iou,
            "mPrecision": macro(prec), "mRecall": macro(rec),
            "mF1": macro(f1), "mIoU": macro(iou),
            "pixel_acc": float(tp.sum() / tot) if tot else 0.0,
            "micro_precision": float(tp.sum() / pred_n.sum()) if tot else 0.0,
            "micro_recall": float(tp.sum() / gt_n.sum()) if tot else 0.0,
        }
