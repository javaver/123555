#!/usr/bin/env python3
"""评估: 每类 P/R/F1/IoU + 宏平均, 对标论文 0.8363/0.8702/0.8522/0.7445。

用法:
    python scripts/eval.py --datasets datasets/ --ckpt runs/dinoseg/best.pt --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import torch
from torch.utils.data import DataLoader

from ds_common import CLASS_NAMES
from dinoseg import ConfusionMeter, DINOvSeg
from dinoseg.dataset import TileDataset
from train import load_rows


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, default=Path("datasets"))
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = DINOvSeg(cfg["encoder"], pretrained=False, num_classes=cfg["num_classes"])
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    ds = TileDataset(a.datasets, load_rows(a.datasets, a.split))
    dl = DataLoader(ds, batch_size=2, num_workers=0)
    meter = ConfusionMeter(cfg["num_classes"])
    for x, y in dl:
        logits = model(x.to(device))
        meter.update(logits.argmax(1).cpu().numpy(), y.numpy())
    r = meter.result()

    print(f"split={a.split}  论文对标: P .8363 / R .8702 / F1 .8522 / IoU .7445")
    print(f"{'class':32s} {'P':>6s} {'R':>6s} {'F1':>6s} {'IoU':>6s}")
    for i, n in enumerate(CLASS_NAMES):
        print(f"{n:32s} {r['precision'][i]:6.4f} {r['recall'][i]:6.4f} "
              f"{r['f1'][i]:6.4f} {r['iou'][i]:6.4f}")
    print(f"{'MACRO':32s} {r['mPrecision']:6.4f} {r['mRecall']:6.4f} "
          f"{r['mF1']:6.4f} {r['mIoU']:6.4f}   pixel_acc={r['pixel_acc']:.4f}")
    out = a.out or (a.ckpt.parent / f"metrics_{a.split}.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("class,precision,recall,f1,iou\n")
        for i, n in enumerate(CLASS_NAMES):
            fh.write(f"{n},{r['precision'][i]:.5f},{r['recall'][i]:.5f},"
                     f"{r['f1'][i]:.5f},{r['iou'][i]:.5f}\n")
        fh.write(f"MACRO,{r['mPrecision']:.5f},{r['mRecall']:.5f},{r['mF1']:.5f},{r['mIoU']:.5f}\n")
    print(f"指标表 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
