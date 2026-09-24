#!/usr/bin/env python3
"""定位 loss=nan: 逐段检查 fp16 AMP 是否产生 NaN (编码器/解码器/损失)。

用法:
    python tools/debug_nan.py --datasets datasets/ --enc-ckpt /root/weights/model.safetensors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import torch

from dinoseg import DINOvSeg, SegLoss
from dinoseg.dataset import TileDataset
from train import load_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, default=Path("datasets"))
    ap.add_argument("--enc-ckpt", default=None)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOvSeg(pretrained=False, checkpoint_path=a.enc_ckpt).to(device).eval()
    ds = TileDataset(a.datasets, load_rows(a.datasets, "train"))
    crit = SegLoss(weight=torch.ones(10, device=device))

    for mode in ("fp16-amp", "fp32"):
        print(f"--- {mode} ---")
        amp = (mode == "fp16-amp") and device.type == "cuda"
        for i in range(a.n):
            x, y = ds[i]
            x, y = x[None].to(device), y[None].to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
                feats = model.encoder(x)
                en = [bool(torch.isnan(f).any()) for f in feats]
                em = [float(f.abs().max()) for f in feats]
                logits = model.forward_feats(feats, (512, 512))
            with torch.amp.autocast("cuda", enabled=amp):
                loss = crit(logits, y)
            print(f"  s{i}: enc_nan={en} enc_max={[round(v, 1) for v in em]} "
                  f"logits_nan={bool(torch.isnan(logits).any())} "
                  f"logits_max={float(logits.abs().max()):.2f} loss={loss.item():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
