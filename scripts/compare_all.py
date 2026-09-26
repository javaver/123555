#!/usr/bin/env python3
"""生成论文双表对比: Table 1 四大村落风貌核心要素表 + Table 2 十类逐类 IoU 表。

Table 1 涵盖 4 大核心要素 (指标: Precision / Recall / F1-score / IoU):
  - Traditional Buildings (老建筑)
  - New Buildings (新建建筑)
  - Greenery (生态绿化: 山林 + 树木合并)
  - Water Bodies (水系水体)
  - Avg (四大要素平均)

Table 2 涵盖全部 10 类地物的逐类 IoU + 宏平均 mIoU (仅本地已训练模型可算,
无检查点的模型以 "—" 占位——论文基准只提供 4 要素口径数值)。

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
    # TODO(论文数值): PSPNet 为本轮新增基线, 论文定稿后在此填入 4 要素基准值,
    # 未填入前 compare_all 对无本地权重的 PSPNet 以 "-" 占位 (不虚构对照数字)。
    # "PSPNet": {
    #     "Precision": [...], "Recall": [...], "F1-score": [...], "IoU": [...],
    # },
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

# Table 2 终端打印用的短列名 (与 CLASS_NAMES 一一对应)
CLASS_SHORT = [
    "BareSoil", "CultLand", "Rail/Hwy", "MtnForest", "NakedMtn",
    "NewBldg", "OldBldg", "Road", "Tree", "Water",
]


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


def compute_perclass_iou(cm: np.ndarray) -> Tuple[List[float], float]:
    """从混淆矩阵计算 10 类逐类 IoU 与宏平均 mIoU。

    宏平均只统计真实出现过的类 (gt 行和 > 0), 与 src/dinoseg/metrics.py 口径一致。
    """
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=1) + cm.sum(axis=0) - tp
    iou = np.where(union > 0, tp / np.clip(union, 1e-9, None), 0.0)
    present = cm.sum(axis=1) > 0
    miou = float(iou[present].mean()) if present.any() else 0.0
    return [float(v) for v in iou], miou


@torch.no_grad()
def eval_checkpoint(ckpt_path: Path, loader: DataLoader, device: torch.device) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_classes = len(CLASS_NAMES)
    meter = ConfusionMeter(num_classes)

    # 判断模型类型
    if "cfg" in ckpt and "encoder" in ckpt.get("cfg", {}):
        # DINO-Seg (参数名是 encoder_id, 不是 encoder_name)
        model = DINOvSeg(
            encoder_id=ckpt["cfg"]["encoder"],
            pretrained=False,
            num_classes=num_classes,
        )
        model.load_state_dict(ckpt["model"])
    elif "arch" in ckpt:
        # Baseline: arch_kwargs 含骨干等 (如 segformer 的 mit_b0)
        arch_kwargs = dict(ckpt.get("arch_kwargs") or {})
        arch_kwargs.pop("pretrained_from", None)  # 评估不重新加载预训练文件
        model = get_baseline_model(
            ckpt["arch"],
            num_classes=num_classes,
            pretrained=False,
            **arch_kwargs,
        )
        model.load_state_dict(ckpt["model"])
    else:
        raise ValueError(f"无法识别检查点结构: {ckpt_path}")

    model.to(device)
    model.eval()

    for x, y in loader:
        logits = model(x.to(device))
        meter.update(logits.argmax(1).cpu().numpy(), y.numpy())
    return meter.cm


def print_table1(rows: List[dict]) -> None:
    header = ["Method", "Metric", "Traditional Buildings", "New Buildings", "Greenery", "Water Bodies", "Avg"]
    print("\n" + "=" * 105)
    print("Table 1. Quantitative comparison of 4 core village elements (P / R / F1 / IoU)")
    print("=" * 105)
    print(f"{header[0]:14s} | {header[1]:10s} | {header[2]:21s} | {header[3]:13s} | {header[4]:10s} | {header[5]:12s} | {header[6]:8s}")
    print("-" * 105)
    for r in rows:
        print(
            f"{r['Method']:14s} | {r['Metric']:10s} | {r['Traditional Buildings']:>21s} | "
            f"{r['New Buildings']:>13s} | {r['Greenery']:>10s} | {r['Water Bodies']:>12s} | {r['Avg']:>8s}"
        )
    print("-" * 105)


def print_table2(rows: List[dict]) -> None:
    print("\n" + "=" * 120)
    print("Table 2. Per-class IoU comparison on the traditional village dataset (10 classes + mIoU)")
    print("=" * 120)
    head = f"{'Method':14s} | " + " | ".join(f"{c:>9s}" for c in CLASS_SHORT) + f" | {'mIoU':>8s}"
    print(head)
    print("-" * 120)
    for r in rows:
        # 数据键是 CLASS_NAMES; CLASS_SHORT 仅用于表头显示
        vals = " | ".join(f"{r[c]:>9s}" for c in CLASS_NAMES)
        print(f"{r['Method']:14s} | {vals} | {r['mIoU']:>8s}")
    print("-" * 120)


def main() -> int:
    ap = argparse.ArgumentParser(description="对比评估: 论文 Table 1 + Table 2 双表")
    ap.add_argument("--datasets", type=Path, default=Path("datasets"))
    ap.add_argument("--split-key", default="split_a")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("runs/comparison_table1.csv"))
    ap.add_argument("--out2", type=Path, default=Path("runs/comparison_table2.csv"))
    ap.add_argument(
        "--allow-paper-fallback",
        action="store_true",
        help="缺本地 best.pt 时填入论文对照值并标 (paper); 默认缺权重直接报错, 避免误当本地结果",
    )
    a = ap.parse_args()

    if not (a.datasets / "manifest.csv").exists():
        raise SystemExit(f"缺少数据集: {a.datasets}/manifest.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_rows = load_test_rows(a.datasets, a.split_key)
    if not test_rows:
        raise SystemExit(f"测试集为空 (split_key={a.split_key})")
    test_ds = TileDataset(a.datasets, test_rows)
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    # 待对比的算法清单 (最终基线阵容: UNet / PSPNet / DeepLabV3 / SegFormer / MaskFormer + 主模型 DINO-Seg)
    models_to_compare = ["UNet", "PSPNet", "DeepLabV3", "SegFormer", "MaskFormer", "DINO-Seg"]
    ckpt_map = {
        "UNet": a.runs / "unet" / "best.pt",
        "PSPNet": a.runs / "pspnet" / "best.pt",
        "DeepLabV3": a.runs / "deeplabv3" / "best.pt",
        "SegFormer": a.runs / "segformer" / "best.pt",
        "MaskFormer": a.runs / "maskformer" / "best.pt",
        "DINO-Seg": a.runs / "dinoseg" / "best.pt",
    }

    table1_rows: List[dict] = []
    table2_rows: List[dict] = []
    missing = [m for m, p in ckpt_map.items() if not p.exists()]
    if missing and not a.allow_paper_fallback:
        raise SystemExit(
            "以下模型缺少本地 best.pt, 拒绝静默填入论文数字:\n  - "
            + "\n  - ".join(f"{m}: {ckpt_map[m]}" for m in missing)
            + "\n若只要对照表预览, 请显式加 --allow-paper-fallback"
        )

    for m_name in models_to_compare:
        ckpt_path = ckpt_map[m_name]
        is_local_evaluated = False
        perclass: Tuple[List[float], float] | None = None

        if ckpt_path.exists():
            print(f"-> 发现本地检查点: {ckpt_path}, 正在测试集评测...")
            cm = eval_checkpoint(ckpt_path, test_dl, device)
            metrics = compute_4elements_metrics(cm)
            perclass = compute_perclass_iou(cm)
            is_local_evaluated = True
        else:
            if m_name in PAPER_REFERENCE:
                print(f"-> 缺少 {ckpt_path}, 使用论文对照值 (paper)")
                metrics = PAPER_REFERENCE[m_name]
            else:
                print(f"-> 缺少 {ckpt_path}, 且论文对照值尚未提供 ({m_name}), 以 '-' 占位")
                metrics = None

        src_tag = "(eval)" if is_local_evaluated else "(paper)"

        # ---- Table 1: 4 大核心要素 ----
        for metric_name in ["Precision", "Recall", "F1-score", "IoU"]:
            vals = metrics[metric_name] if metrics is not None else None
            table1_rows.append({
                "Method": f"{m_name} {src_tag}",
                "Metric": metric_name,
                "Traditional Buildings": f"{vals[0]:.4f}" if vals else "-",
                "New Buildings": f"{vals[1]:.4f}" if vals else "-",
                "Greenery": f"{vals[2]:.4f}" if vals else "-",
                "Water Bodies": f"{vals[3]:.4f}" if vals else "-",
                "Avg": f"{vals[4]:.4f}" if vals else "-",
            })

        # ---- Table 2: 10 类逐类 IoU + mIoU ----
        row2: Dict[str, str] = {"Method": f"{m_name} {src_tag}"}
        if perclass is not None:
            ious, miou = perclass
            for cname, v in zip(CLASS_NAMES, ious):
                row2[cname] = f"{v:.4f}"
            row2["mIoU"] = f"{miou:.4f}"
        else:
            for cname in CLASS_NAMES:
                row2[cname] = "-"
            row2["mIoU"] = "-"
        table2_rows.append(row2)

    # ---- 终端打印双表 ----
    print_table1(table1_rows)
    print_table2(table2_rows)

    # ---- 导出 CSV: Table 1 ----
    header1 = ["Method", "Metric", "Traditional Buildings", "New Buildings", "Greenery", "Water Bodies", "Avg"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=header1)
        writer.writeheader()
        writer.writerows(table1_rows)
    print(f"\n[OK] Table 1 已导出 -> {a.out}")

    # ---- 导出 CSV: Table 2 ----
    header2 = ["Method"] + list(CLASS_NAMES) + ["mIoU"]
    a.out2.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out2, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=header2)
        writer.writeheader()
        writer.writerows(table2_rows)
    print(f"[OK] Table 2 已导出 -> {a.out2}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
