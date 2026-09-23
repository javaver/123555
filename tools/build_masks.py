#!/usr/bin/env python3
"""由 LabelMe 多边形生成逐像素类别掩膜 (0..9, ignore=255)。

用法:
    python tools/build_masks.py datasets/

规则(见 docs/PLAN.md §1):
  * 按多边形面积降序绘制 => 小目标后画, 不被大目标吞边;
  * 未被任何多边形覆盖的像素 = ignore(255)(村界外近黑 padding 也落在并集补集里);
  * 类别外标签告警并跳过; 越界坐标 clip 到 [0, W/H);
  * 统计 valid_px / pad_px / 每类像素数, 回写 manifest.csv。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ds_common import (CLASS_NAMES, CLASS_TO_ID, IGNORE, polygon_area,
                       read_labelme, read_manifest, slug, write_manifest)


def rasterize(doc: dict, w: int, h: int) -> tuple[np.ndarray, list[str]]:
    warns: list[str] = []
    shapes = []
    for s in doc.get("shapes", []):
        lab = s.get("label")
        if lab not in CLASS_TO_ID:
            warns.append(f"类别外标签跳过: {lab!r}")
            continue
        if s.get("shape_type") not in (None, "polygon"):
            warns.append(f"非 polygon 形状跳过: {s.get('shape_type')}")
            continue
        pts = [(min(max(x, 0), w - 1), min(max(y, 0), h - 1)) for x, y in s["points"]]
        shapes.append((polygon_area(pts), CLASS_TO_ID[lab], pts))

    mask = Image.new("L", (w, h), IGNORE)
    dr = ImageDraw.Draw(mask)
    for _area, cid, pts in sorted(shapes, key=lambda t: -t[0]):  # 大->小, 小目标后画
        dr.polygon(pts, fill=cid)
    return np.asarray(mask), warns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", type=Path)
    a = ap.parse_args()

    mpath = a.datasets / "manifest.csv"
    if not mpath.exists():
        print(f"先运行 ingest.py ({mpath} 不存在)", file=sys.stderr)
        return 2

    rows = read_manifest(mpath)
    mask_root = a.datasets / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    n_warn = 0

    for r in rows:
        doc = read_labelme(Path(r["json_src"]))
        w, h = int(r["width"]), int(r["height"])
        mask, warns = rasterize(doc, w, h)
        for msg in warns:
            print(f"  [warn] {r['village']}/{r['tile']}: {msg}")
        n_warn += len(warns)

        # 尺寸一致性体检
        jp_w, jp_h = doc.get("imageWidth"), doc.get("imageHeight")
        if (jp_w, jp_h) != (w, h):
            print(f"  [warn] {r['village']}/{r['tile']}: JSON 头尺寸 {(jp_w, jp_h)} != 图像 {(w, h)}")
            n_warn += 1

        mp = mask_root / f"{r['village']}__{r['tile']}.png"
        Image.fromarray(mask).save(mp)
        r["mask_rel"] = mp.relative_to(a.datasets).as_posix()

        valid = mask != IGNORE
        r["valid_px"] = int(valid.sum())
        r["pad_px"] = int((~valid).sum())
        for name in CLASS_NAMES:
            r[f"px_{slug(name)}"] = int((mask == CLASS_TO_ID[name]).sum())

    write_manifest(mpath, rows)
    tot_valid = sum(int(r["valid_px"]) for r in rows)
    print(f"掩膜 {len(rows)} 张 -> {mask_root}/  (告警 {n_warn} 条)")
    print("像素级类别分布 (valid 像素):")
    for name in CLASS_NAMES:
        tot = sum(int(r[f"px_{slug(name)}"]) for r in rows)
        print(f"    {name:32s} {tot:10d}  {100 * tot / max(tot_valid, 1):5.2f}%")
    missing = [n for n in CLASS_NAMES
               if sum(int(r[f"px_{slug(n)}"]) for r in rows) == 0]
    if missing:
        print(f"  [warn] 当前数据未出现的类别: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
