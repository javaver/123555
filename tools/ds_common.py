#!/usr/bin/env python3
"""DINO-Seg 数据管线公共工具。只依赖 Pillow / numpy / 标准库。

约定:
  类别 id 0..9 与 tools/dataset_check.py 一致, ignore = 255。
  manifest.csv 一行一个样本, 主键 (village, tile)。
"""
from __future__ import annotations

import base64
import csv
import io
import json
import struct
from pathlib import Path

from PIL import Image

CLASS_NAMES = [  # 顺序即 mask 像素值 0..9
    "bare soil", "cultivated land", "high speed rail and highway",
    "mountain forest", "naked mountain", "new building",
    "old building", "road", "tree", "water",
]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
IGNORE = 255


def slug(name: str) -> str:
    return name.replace(" ", "_")


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
IMG_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")

MANIFEST_FIELDS = [
    "village", "tile", "image_rel", "mask_rel", "json_src", "source",
    "width", "height", "valid_px", "pad_px",
] + [f"px_{slug(n)}" for n in CLASS_NAMES]


# ---------- 容器识别(不看扩展名, 只看魔数) ----------

def sniff(data: bytes) -> str:
    if data.startswith(PNG_MAGIC):
        return "png"
    if data.startswith(b"II*\x00") or data.startswith(b"MM\x00*"):
        return "tiff"
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    return "unknown"


def load_image(path: Path) -> Image.Image:
    """PIL 按内容识别格式, 扩展名是 .png 但实为 TIFF 也能读。"""
    with open(path, "rb") as fh:
        head = fh.read(8)
    fmt = sniff(head)
    if fmt == "unknown":
        raise ValueError(f"{path}: 无法识别的图像容器 (魔数 {head[:4].hex()})")
    return Image.open(path).convert("RGB")


def image_from_embedded(doc: dict) -> Image.Image | None:
    blob = doc.get("imageData")
    if not blob:
        return None
    return Image.open(io.BytesIO(base64.b64decode(blob))).convert("RGB")


# ---------- LabelMe 解析 ----------

def parse_key(doc: dict, fallback_stem: str) -> tuple[str, str]:
    """从 imagePath 提取 (village, tile); 形如 ..\\sanglincun14\\tile_0_1024.png。"""
    p = (doc.get("imagePath") or "").replace("\\", "/").strip()
    if p:
        parts = [x for x in p.split("/") if x]
        if len(parts) >= 2:
            return parts[-2], Path(parts[-1]).stem
        if len(parts) == 1:
            return "unknown", Path(parts[0]).stem
    return "unknown", fallback_stem


def polygon_area(pts: list) -> float:
    """鞋带公式, 用于'大目标先画、小目标后画'的栅格化顺序。"""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def read_labelme(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------- manifest ----------

def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
