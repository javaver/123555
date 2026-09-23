#!/usr/bin/env python3
"""全量推理写掩膜, 供 tools/analyze.py --mode pred 出逐图占比。

用法:
    python scripts/infer.py --datasets datasets/ --ckpt runs/dinoseg/best.pt \
        --out runs/pred_masks/ [--split all]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch
from PIL import Image

from ds_common import read_manifest
from dinoseg import DINOvSeg
from dinoseg.dataset import TileDataset
from train import load_rows


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, default=Path("datasets"))
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/pred_masks"))
    ap.add_argument("--split", default="all")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = DINOvSeg(cfg["encoder"], pretrained=False, num_classes=cfg["num_classes"])
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    if a.split == "all":
        rows = read_manifest(a.datasets / "manifest.csv")
    else:
        rows = load_rows(a.datasets, a.split)
    ds = TileDataset(a.datasets, rows)
    a.out.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(rows):
        x, _ = ds[i]
        logits = model(x[None].to(device))
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        img = np.asarray(Image.open(a.datasets / r["image_rel"]).convert("RGB"))
        pred[img.max(axis=2) <= 5] = 255  # padding 恢复为 ignore
        Image.fromarray(pred).save(a.out / f"{r['village']}__{r['tile']}.png")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}")
    print(f"预测掩膜 {len(rows)} 张 -> {a.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
