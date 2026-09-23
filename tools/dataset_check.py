#!/usr/bin/env python3
"""DINO-Seg 数据集体检 + 归一化工具。

用法:
    python tools/dataset_check.py train/                    # 只体检, 不改文件
    python tools/dataset_check.py train/ --fix              # 把"假png"(实为TIFF)重写为真PNG
    python tools/dataset_check.py train/ --report out.csv   # 导出逐图报告

不依赖 PIL / cv2 / numpy, 直接读容器头, 所以能识别"扩展名与实际格式不符"的文件。
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import struct
import sys
from collections import Counter
from pathlib import Path

CLASS_NAMES = [  # 论文里的 10 类, 顺序即 mask 像素值 0..9
    "bare soil", "cultivated land", "high speed rail and highway",
    "mountain forest", "naked mountain", "new building",
    "old building", "road", "tree", "water",
]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


# ---------- 容器识别(不看扩展名, 只看魔数) ----------

def sniff(data: bytes) -> tuple[str, int, int]:
    """返回 (真实格式, 宽, 高)。无法识别时返回 ('unknown', 0, 0)。"""
    if data.startswith(PNG_MAGIC):
        w, h = struct.unpack(">II", data[16:24])
        return "png", w, h
    if data.startswith(b"II*\x00") or data.startswith(b"MM\x00*"):
        return ("tiff",) + _tiff_wh(data)
    if data.startswith(JPEG_MAGIC):
        return ("jpeg",) + _jpeg_wh(data)
    return "unknown", 0, 0


def _tiff_wh(d: bytes) -> tuple[int, int]:
    bo = "<" if d[:2] == b"II" else ">"
    if len(d) < 8:
        return 0, 0
    off = struct.unpack(bo + "I", d[4:8])[0]
    if len(d) < off + 2:
        return 0, 0
    n = struct.unpack(bo + "H", d[off:off + 2])[0]
    w = h = 0
    for i in range(n):
        chunk = d[off + 2 + i * 12: off + 2 + i * 12 + 8]
        if len(chunk) < 8:
            break
        tag, typ, _cnt = struct.unpack(bo + "HHI", chunk)
        raw = d[off + 2 + i * 12 + 8: off + 2 + i * 12 + 12]
        if len(raw) < 4:
            break
        val = struct.unpack(bo + "I", raw)[0]
        if tag == 256:
            w = val
        elif tag == 257:
            h = val
    return w, h


def _jpeg_wh(d: bytes) -> tuple[int, int]:
    i, n = 2, len(d)
    while i + 9 < n:
        if d[i] != 0xFF:
            i += 1
            continue
        marker, size = d[i + 1], struct.unpack(">H", d[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB):
            h, w = struct.unpack(">HH", d[i + 5:i + 9])
            return w, h
        i += 2 + max(size, 2)
    return 0, 0


# ---------- LabelMe ----------

def read_labelme(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def embedded_size(doc: dict) -> tuple[str, int, int]:
    blob = doc.get("imageData")
    if not blob:
        return "none", 0, 0
    try:
        return sniff(base64.b64decode(blob))
    except Exception:
        return "corrupt", 0, 0


# ---------- 体检 ----------

def check(root: Path) -> list[dict]:
    rows = []
    images = {p.stem: p for p in sorted(root.rglob("*"))
              if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}}
    metas = {p.stem: p for p in sorted(root.rglob("*.json"))}

    for stem in sorted(set(images) | set(metas)):
        img, meta = images.get(stem), metas.get(stem)
        row = {"stem": stem, "image": img.name if img else "", "json": meta.name if meta else ""}
        if img:
            head = img.read_bytes()[:8192]
            fmt, w, h = sniff(head)
            row.update(real_format=fmt, ext=img.suffix.lower().lstrip("."),
                       file_w=w, file_h=h)
            row["ext_matches"] = (fmt == "jpeg" and row["ext"] in {"jpg", "jpeg"}) or fmt == row["ext"]
        if meta:
            doc = read_labelme(meta)
            efmt, ew, eh = embedded_size(doc)
            labs = [s["label"] for s in doc.get("shapes", [])]
            coords = [c for s in doc.get("shapes", []) for p in s["points"] for c in p]
            row.update(json_w=doc.get("imageWidth"), json_h=doc.get("imageHeight"),
                       embedded_format=efmt, embedded_w=ew, embedded_h=eh,
                       n_shapes=len(labs), labels="|".join(sorted(set(labs))),
                       coord_max=round(max(coords), 2) if coords else None,
                       coord_min=round(min(coords), 2) if coords else None,
                       unknown_labels="|".join(sorted(set(labs) - set(CLASS_TO_ID))))
            # 三方尺寸是否一致
            dims = {v for v in [(row.get("file_w"), row.get("file_h")),
                                (row.get("json_w"), row.get("json_h")),
                                (ew, eh)] if v != (0, 0) and None not in v}
            row["dims_agree"] = len(dims) == 1
        rows.append(row)
    return rows


def report(rows: list[dict]) -> None:
    sizes = Counter((r.get("file_w"), r.get("file_h")) for r in rows if r.get("file_w"))
    fmts = Counter(r.get("real_format") for r in rows if r.get("real_format"))
    mismatched = [r for r in rows if r.get("ext_matches") is False]
    disagree = [r for r in rows if r.get("dims_agree") is False]
    orphan_img = [r for r in rows if r["image"] and not r["json"]]
    orphan_json = [r for r in rows if r["json"] and not r["image"]]
    unknown = {r["stem"]: r["unknown_labels"] for r in rows if r.get("unknown_labels")}
    all_labels = Counter()
    for r in rows:
        for lab in (r.get("labels") or "").split("|"):
            if lab:
                all_labels[lab] += 1

    print(f"扫描到 {len(rows)} 组样本\n")
    print("像素尺寸分布 (file_w x file_h):")
    for (w, h), c in sizes.most_common():
        print(f"    {w} x {h:<6} {c} 张")
    print("\n真实容器格式:", dict(fmts))
    print("扩展名与实际格式不符:", len(mismatched), "个", [r["stem"] for r in mismatched][:8])
    print("图/json/内嵌图 三方尺寸不一致:", len(disagree), "个", [r["stem"] for r in disagree][:8])
    print("有图无标注:", len(orphan_img), "| 有标注无图:", len(orphan_json))
    print("类别外标签:", unknown or "无")
    print("\n类别出现次数(按图计):")
    for k, v in all_labels.most_common():
        print(f"    {k:32s} {v}")
    print("\n未出现的类别:", [c for c in CLASS_NAMES if c not in all_labels] or "无")


# ---------- 修复 ----------

def fix(root: Path) -> None:
    """把扩展名是 .png 但内容不是 PNG 的文件重写成真 PNG(不装 PIL 时只能改名, 这里给出提示)。"""
    changed = 0
    for p in sorted(root.rglob("*.png")):
        fmt, _w, _h = sniff(p.read_bytes()[:8192])
        if fmt == "png":
            continue
        if fmt in {"tiff", "jpeg"}:
            tgt = p.with_suffix("." + ("tif" if fmt == "tiff" else "jpg"))
            p.rename(tgt)
            print(f"  {p.name} 实际是 {fmt} -> 重命名为 {tgt.name}")
            changed += 1
    print("注: 只做改名, 不重编码。若下游用 PIL/cv2 读, 内容能被正确解码;")
    print("    若必须得到真 PNG, 用: python -m PIL ... 或 cv2.imread + cv2.imwrite。")
    print(f"共处理 {changed} 个文件。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()
    if not a.root.is_dir():
        print(f"目录不存在: {a.root}", file=sys.stderr)
        return 2
    rows = check(a.root)
    report(rows)
    if a.report:
        keys = sorted({k for r in rows for k in r})
        with open(a.report, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\n逐图报告已写入 {a.report}")
    if a.fix:
        print("\n--- 修复 ---")
        fix(a.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
