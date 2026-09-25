#!/usr/bin/env python3
"""逐图 10 类要素占比统计(核心交付)。

用法:
    python tools/analyze.py datasets/ --mode gt              # 用标注掩膜做数据集画像
    python tools/analyze.py datasets/ --mode pred --pred-dir runs/pred_masks/

输出:
    datasets/analysis/proportions_{mode}.csv   一张图一行:
        village, tile, valid_pct, p_<10类>(有效区内和为1), g_<组合占比>
    datasets/analysis/village_summary_{mode}.csv  按村以 valid 像素加权平均

分母 = 有效像素:
    gt 模式   : 掩膜 != 255 (多边形并集);
    pred 模式 : 图像 max(R,G,B) > 5 (实测与多边形并集补集一致性 >= 99.4%, PLAN §1.6)。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ds_common import CLASS_NAMES, CLASS_TO_ID, IGNORE, read_manifest, slug

GROUPS = {  # 组合占比映射, 可按需增删
    "g_building": ["new building", "old building"],
    "g_vegetation": ["tree", "mountain forest"],
    "g_water": ["water"],
    "g_transport": ["road", "high speed rail and highway"],
    "g_bare": ["bare soil", "naked mountain"],
    "g_cultivated": ["cultivated land"],
}

FIELDS = ["village", "tile", "source", "valid_pct"] \
    + [f"p_{slug(n)}" for n in CLASS_NAMES] + list(GROUPS)


def proportions(mask: np.ndarray, valid: np.ndarray) -> dict:
    v = int(valid.sum())
    px = {n: int(((mask == cid) & valid).sum()) for n, cid in CLASS_TO_ID.items()}
    row = {f"p_{slug(n)}": (px[n] / v if v else 0.0) for n in CLASS_NAMES}
    for g, members in GROUPS.items():
        row[g] = sum(row[f"p_{slug(n)}"] for n in members)
    row["valid_pct"] = 100.0 * v / valid.size
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", type=Path)
    ap.add_argument("--mode", choices=["gt", "pred"], default="gt")
    ap.add_argument("--pred-dir", type=Path, default=None)
    a = ap.parse_args()

    rows = read_manifest(a.datasets / "manifest.csv")
    out_dir = a.datasets / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"proportions_{a.mode}.csv"

    results = []
    for r in rows:
        if a.mode == "gt":
            if not r["mask_rel"]:
                print(f"  [warn] {r['tile']} 无掩膜, 跳过 (先跑 build_masks.py)")
                continue
            mask = np.asarray(Image.open(a.datasets / r["mask_rel"]))
            valid = mask != IGNORE
        else:
            pred_dir = a.pred_dir or (a.datasets / "pred_masks")
            pm = pred_dir / f"{r['village']}__{r['tile']}.png"
            if not pm.exists():
                print(f"  [warn] 预测缺失: {pm.name}, 跳过")
                continue
            mask = np.asarray(Image.open(pm))
            img = np.asarray(Image.open(a.datasets / r["image_rel"]).convert("RGB"))
            valid = img.max(axis=2) > 5  # 近黑 padding 规则

        row = proportions(mask, valid)
        row.update(village=r["village"], tile=r["tile"], source=a.mode)
        results.append(row)

    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in results:
            w.writerow(row)

    # 按村加权汇总
    summ = out_dir / f"village_summary_{a.mode}.csv"
    agg: dict[str, list] = {}
    for row in results:
        a_ = agg.setdefault(row["village"], [])
        a_.append(row)
    with open(summ, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["village", "n_tiles", "mean_valid_pct"]
                           + [f"p_{slug(n)}" for n in CLASS_NAMES] + list(GROUPS))
        w.writeheader()
        for v, rs in sorted(agg.items()):
            wts = np.array([r["valid_pct"] for r in rs])
            line = {"village": v, "n_tiles": len(rs),
                    "mean_valid_pct": float(wts.mean())}
            for f in [f"p_{slug(n)}" for n in CLASS_NAMES] + list(GROUPS):
                line[f] = float(np.average([r[f] for r in rs], weights=wts))
            w.writerow(line)

    print(f"{a.mode} 模式统计 {len(results)} 张 -> {out}")
    print(f"村级汇总 -> {summ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
