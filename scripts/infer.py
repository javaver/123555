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
import torch.nn.functional as F
from PIL import Image

from ds_common import CLASS_NAMES, CLASS_TO_ID, IMG_EXTS, read_manifest, slug
from dinoseg import DINOvSeg
from dinoseg.dataset import TileDataset, normalize_array
from train import load_rows


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, default=Path("datasets"))
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/pred_masks"))
    ap.add_argument("--split", default="all")
    ap.add_argument("--split-key", default="split_a", choices=["split_a", "split_c"])
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="新图目录(无标注): 逐图出掩膜+10类占比CSV")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = DINOvSeg(cfg["encoder"], pretrained=False, num_classes=cfg["num_classes"])
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    if a.image_dir:  # 无标注新图: 逐图出掩膜 + 10 类占比 CSV
        import csv as _csv
        from analyze import GROUPS
        files = sorted(p for p in a.image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not files:
            print(f"{a.image_dir} 下没有图像文件", file=sys.stderr)
            return 2
        a.out.mkdir(parents=True, exist_ok=True)
        rows_out = []
        for i, p in enumerate(files):
            img = Image.open(p).convert("RGB")
            x = normalize_array(img)[None].to(device)
            h0, w0 = x.shape[-2:]
            if h0 * w0 > 2048 * 2048:
                print(f"  [warn] {p.name} 尺寸较大({h0}x{w0}), 若显存不足请切块后推理")
            ph, pw = (16 - h0 % 16) % 16, (16 - w0 % 16) % 16
            if ph or pw:  # 任意尺寸新图: reflect 填充到 16 的倍数, 推完裁回
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
            pred = pred[:h0, :w0]
            arr = np.asarray(img)
            valid = arr.max(axis=2) > 5
            pred[~valid] = 255
            Image.fromarray(pred).save(a.out / f"{p.stem}.png")
            v = int(valid.sum())
            row = {"image": p.name, "valid_pct": round(100 * v / valid.size, 2)}
            for n, cid in CLASS_TO_ID.items():
                row[f"p_{slug(n)}"] = round(int(((pred == cid) & valid).sum()) / v, 6) if v else 0.0
            for g, members in GROUPS.items():
                row[g] = round(sum(row[f"p_{slug(n)}"] for n in members), 6)
            rows_out.append(row)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(files)}")
        with open(a.out / "proportions.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"新图推理 {len(rows_out)} 张 -> {a.out}/ (掩膜 + proportions.csv)")
        return 0

    if a.split == "all":
        rows = read_manifest(a.datasets / "manifest.csv")
    else:
        rows = load_rows(a.datasets, a.split, a.split_key)
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
