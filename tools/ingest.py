#!/usr/bin/env python3
"""数据入库: 扫描 LabelMe 标注目录, 修复容器格式, 统一写出真 PNG + manifest.csv。

用法:
    python tools/ingest.py train/ --out datasets/

行为:
  * 主键 (village, tile) 取自 JSON imagePath (Windows 反斜杠路径已归一);
  * 磁盘图按内容识别(TIFF/PNG/JPEG 均可)重存为真 PNG;
  * 磁盘图缺失时回退 JSON 内嵌 imageData (source=embedded);
  * 两者都在且像素不一致时告警 (source=disk, embed_match=0)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ds_common import (IMG_EXTS, image_from_embedded, load_image, parse_key,
                       read_labelme, write_manifest)


def find_sibling_image(json_path: Path) -> Path | None:
    for ext in IMG_EXTS:
        cand = json_path.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="标注目录 (含 *.json 与图像)")
    ap.add_argument("--out", type=Path, default=Path("datasets"))
    a = ap.parse_args()

    if not a.root.is_dir():
        print(f"目录不存在: {a.root}", file=sys.stderr)
        return 2

    jsons = sorted(a.root.rglob("*.json"))
    if not jsons:
        print(f"{a.root} 下没有 *.json", file=sys.stderr)
        return 2

    img_root = a.out / "images"
    rows, seen, n_warn = [], set(), 0

    for jp in jsons:
        doc = read_labelme(jp)
        village, tile = parse_key(doc, jp.stem)
        key = (village, tile)
        if key in seen:  # 同村同名: 数据本身的问题, 加后缀保底并告警
            k = 1
            while (village, f"{tile}_dup{k}") in seen:
                k += 1
            print(f"  [warn] 主键冲突 {village}/{tile} -> {tile}_dup{k} ({jp.name})")
            tile, n_warn = f"{tile}_dup{k}", n_warn + 1
            key = (village, tile)
        seen.add(key)

        src = "disk"
        img_path = find_sibling_image(jp)
        img = None
        if img_path is not None:
            try:
                img = load_image(img_path)
            except Exception as e:  # 容器损坏等
                print(f"  [warn] 磁盘图读取失败 {img_path.name}: {e}; 尝试内嵌图")
        if img is None:
            img = image_from_embedded(doc)
            src = "embedded"
            if img is None:
                print(f"  [error] {jp.name}: 无可用图像, 跳过")
                n_warn += 1
                continue
        elif doc.get("imageData"):
            emb = image_from_embedded(doc)
            if emb is not None and not np.array_equal(np.asarray(img), np.asarray(emb)):
                print(f"  [warn] {village}/{tile}: 磁盘图与内嵌图像素不一致, 以磁盘图为准")
                n_warn += 1

        dst = img_root / village / f"{tile}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "PNG")

        rows.append({
            "village": village, "tile": tile,
            "image_rel": dst.relative_to(a.out).as_posix(),
            "mask_rel": "", "json_src": str(jp),
            "source": src, "width": img.width, "height": img.height,
            "valid_px": "", "pad_px": "",
        })

    write_manifest(a.out / "manifest.csv", rows)
    villages = sorted({r["village"] for r in rows})
    print(f"入库 {len(rows)} 组样本, {len(villages)} 个村: {', '.join(villages)}")
    print(f"manifest -> {a.out / 'manifest.csv'}  (告警 {n_warn} 条)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
