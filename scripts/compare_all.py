#!/usr/bin/env python3
"""生成论文 Table 1 格式的 4 大村落风貌核心要素对比表。

涵盖:
  - Traditional Buildings (老建筑)
  - New Buildings (新建建筑)
  - Greenery (生态绿化: 山林 + 树木合并)
  - Water Bodies (水系水体)
  - Avg (四大要素平均)

指标: Precision, Recall, F1-score, IoU

用法:
    python scripts/compare_all.py --datasets datasets/
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from ds_common import CLASS_NAMES, read_manifest
from dinoseg import ConfusionMeter, DINOvSeg
from dinoseg.dataset import TileDataset
from baselines import get_baseline_model

# 论文发表的原表基准数值（作为对照参考）
PAPER_REFERENCE = {
    "UNet": {
        "Precision": [0.7051, 0.7821, 0.8331, 0.3734, 0.6734],
        "Recall": [0.7586, 0.7724, 0.8621, 0.9367, 0.8325],
        "F1-score": [0.7309, 0.7772, 0.8473, 0.5339, 0.7223],
        "IoU": [0.5759, 0.6356, 0.7351, 0.3642, 0.5777],
    },
    "DeepLabV3": {
        "Precision": [0.5942, 0.7254, 0.7876, 0.4466, 0.6384],
        "Recall": [0.7508, 0.6907, 0.7823, 0.8444, 0.7671],
        "F1-score": [0.6634, 0.7076, 0.7850, 0.5832, 0.6850],
        "IoU": [0.4963, 0.5475, 0.6460, 0.4126, 0.5266],
    },
    "DPT": {
        "Precision": [0.7522, 0.7626, 0.8439, 0.5414, 0.7250],
        "Recall": [0.7419, 0.8078, 0.8398, 0.8109, 0.8001],
        "F1-score": [0.7470, 0.7846, 0.8419, 0.6493, 0.7557],
        "IoU": [0.5962, 0.6455, 0.7269, 0.4807, 0.6123],
    },
    "SegFormer": {
        "Precision": [0.7486, 0.7917, 0.8580, 0.5943, 0.7481],
        "Recall": [0.7367, 0.7840, 0.8432, 0.7780, 0.7855],
        "F1-score": [0.7426, 0.7878, 0.8505, 0.6739, 0.7637],
        "IoU": [0.5906, 0.6499, 0.7400, 0.5081, 0.6222],
    },
    "MaskFormer": {
        "Precision": [0.7541, 0.8503, 0.8948, 0.7797, 0.8197],
        "Recall": [0.7994, 0.8207, 0.8570, 0.9309, 0.8520],
        "F1-score": [0.7761, 0.8353, 0.8755, 0.8487, 0.8339],
        "IoU": [0.6376, 0.7220, 0.7845, 0.7423, 0.7216],
    },
    "DINO-Seg": {
        "Precision": [0.7776, 0.8474, 0.8808, 0.8393, 0.8363],
        "Recall": [0.8030, 0.8478, 0.8724, 0.9578, 0.8702],
        "F1-score": [0.7901, 0.8476, 0.8766, 0.8946, 0.8522],
        "IoU": [0.6530, 0.7355, 0.7802, 0.8093, 0.7445],
    },
}


def load_test_rows(datasets: Path, key: str = "split_a"):
    splits = {}
    with open(datasets / "splits.csv", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            splits[(row["village"], row["tile"])] = row
    return [
        r
        for r in read_manifest(datasets / "manifest.csv")
        if splits[(r["village"], r["tile"])][key] == "test"
    ]


def compute_4elements_metrics(cm: np.ndarray) -> Dict[str, List[float]]:
    """从 10 类混淆矩阵计算 4 大核心风貌要素的 P, R, F1, IoU 及 Avg。

    类索引映射:
      - 传统建筑 (Traditional Buildings): 6 (old_building)
      - 新建建筑 (New Buildings): 5 (new_building)
      - 生态绿化 (Greenery): 3 (mountain_forest) + 8 (tree)
      - 水体水系 (Water Bodies): 9 (water)
    """

    def prf_iou(tp, fp, fn):
        p = tp / max(tp + fp, 1e-9)
        r = tp / max(tp + fn, 1e-9)
        f1 = 2 * p * r / max(p + r, 1e-9)
        iou = tp / max(tp + fp + fn, 1e-9)
        return p, r, f1, iou

    # 1. Traditional Buildings (class 6)
    tp_trad = cm[6, 6]
    fp_trad = cm[:, 6].sum() - tp_trad
    fn_trad = cm[6, :].sum() - tp_trad
    p_trad, r_trad, f1_trad, iou_trad = prf_iou(tp_trad, fp_trad, fn_trad)

    # 2. New Buildings (class 5)
    tp_new = cm[5, 5]
    fp_new = cm[:, 5].sum() - tp_new
    fn_new = cm[5, :].sum() - tp_new
    p_new, r_new, f1_new, iou_new = prf_iou(tp_new, fp_new, fn_new)

    # 3. Greenery (classes 3 & 8 merged)
    tp_grn = cm[3, 3] + cm[3, 8] + cm[8, 3] + cm[8, 8]
    pred_grn = cm[:, 3].sum() + cm[:, 8].sum()
    gt_grn = cm[3, :].sum() + cm[8, :].sum()
    fp_grn = pred_grn - tp_grn
    fn_grn = gt_grn - tp_grn
    p_grn, r_grn, f1_grn, iou_grn = prf_iou(tp_grn, fp_grn, fn_grn)

    # 4. Water Bodies (class 9)
    tp_wat = cm[9, 9]
    fp_wat = cm[:, 9].sum() - tp_wat
    fn_wat = cm[9, :].sum() - tp_wat
    p_wat, r_wat, f1_wat, iou_wat = prf_iou(tp_wat, fp_wat, fn_wat)

    # 组装 4 要素及其 Avg (均值)
    results = {}
    for name, vals in [
        ("Precision", [p_trad, p_new, p_grn, p_wat]),
        ("Recall", [r_trad, r_new, r_grn, r_wat]),
        ("F1-score", [f1_trad, f1_new, f1_grn, f1_wat]),
        ("IoU", [iou_trad, iou_new, iou_grn, iou_wat]),
    ]:
        results[name] = vals + [float(np.mean(vals))]
    return results


@torch.no_grad()
def eval_checkpoint(ckpt_path: Path, loader: DataLoader, device: torch.device) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = len(CLASS_NAMES)
    meter = ConfusionMeter(num_classes)

    # 判断模型类型
    if "cfg" in ckpt and "encoder" in ckpt["cfg"]:
        # DINO-Seg
        model = DINOvSeg(
            encoder_name=ckpt["cfg"]["encoder"],
            pretrained=False,
            num_classes=num_classes,
        )
        model.load_state_dict(ckpt["model"])
    elif "arch" in ckpt:
        # Baseline model
        model = get_baseline_model(ckpt["arch"], num_classes=num_classes, pretrained=False)
        model.load_state_dict(ckpt["model"])
    else:
        raise ValueError(f"无法识别检查点结构: {ckpt_path}")

    model.to(device)
    model.eval()

    for x, y in loader:
        logits = model(x.to(device))
        meter.update(logits.argmax(1).cpu().numpy(), y.numpy())
    return meter.cm


def main() -> int:
    ap = argparse.ArgumentParser(description="对比评估 Table 1")
    ap.add_argument("--datasets", type=Path, default=Path("datasets"))
    ap.add_argument("--split-key", default="split_a")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("runs/comparison_table1.csv"))
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_rows = load_test_rows(a.datasets, a.split_key)
    test_ds = TileDataset(a.datasets, test_rows)
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    # 待对比的算法清单
    models_to_compare = ["UNet", "DeepLabV3", "DPT", "SegFormer", "MaskFormer", "DINO-Seg"]
    ckpt_map = {
        "UNet": a.runs / "unet" / "best.pt",
        "DeepLabV3": a.runs / "deeplabv3" / "best.pt",
        "DPT": a.runs / "dpt" / "best.pt",
        "SegFormer": a.runs / "segformer" / "best.pt",
        "MaskFormer": a.runs / "maskformer" / "best.pt",
        "DINO-Seg": a.runs / "dinoseg" / "best.pt",
    }

    final_table = []
    print("\n" + "=" * 80)
    print("Table 1. Quantitative comparison of different methods on traditional village dataset")
    print("=" * 80)

    header = ["Method", "Metric", "Traditional Buildings", "New Buildings", "Greenery", "Water Bodies", "Avg"]
    print(f"{header[0]:14s} | {header[1]:10s} | {header[2]:21s} | {header[3]:13s} | {header[4]:10s} | {header[5]:12s} | {header[6]:8s}")
    print("-" * 105)

    for m_name in models_to_compare:
        ckpt_path = ckpt_map[m_name]
        is_local_evaluated = False

        if ckpt_path.exists():
            print(f"-> 发现本地检查点: {ckpt_path}, 正在测试集评测...")
            cm = eval_checkpoint(ckpt_path, test_dl, device)
            metrics = compute_4elements_metrics(cm)
            is_local_evaluated = True
        else:
            # 使用原论文基准对照
            metrics = PAPER_REFERENCE[m_name]

        src_tag = "(eval)" if is_local_evaluated else "(paper)"

        for idx, metric_name in enumerate(["Precision", "Recall", "F1-score", "IoU"]):
            vals = metrics[metric_name]
            method_col = f"{m_name} {src_tag}" if idx == 0 else ""
            row_str = (
                f"{method_col:14s} | {metric_name:10s} | "
                f"{vals[0]:21.4f} | {vals[1]:13.4f} | "
                f"{vals[2]:10.4f} | {vals[3]:12.4f} | "
                f"{vals[4]:8.4f}"
            )
            print(row_str)
            final_table.append({
                "Method": f"{m_name} {src_tag}",
                "Metric": metric_name,
                "Traditional Buildings": f"{vals[0]:.4f}",
                "New Buildings": f"{vals[1]:.4f}",
                "Greenery": f"{vals[2]:.4f}",
                "Water Bodies": f"{vals[3]:.4f}",
                "Avg": f"{vals[4]:.4f}",
            })
        print("-" * 105)

    # 导出 CSV
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(final_table)
    print(f"\n[OK] 对比总表已导出 -> {a.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
