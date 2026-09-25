#!/usr/bin/env python3
"""一次性预计算冻结 DINOv3 编码器的 4 级特征并存盘 (fp16, 约 6.3MB/张)。

用法:
    python scripts/cache_features.py --datasets datasets/ --out runs/feats/

之后训练用 --cache-dir runs/feats/, 每 epoch 不再跑编码器, CPU 也可训。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import torch
from PIL import Image

from dinoseg import DINOvSeg
from dinoseg.dataset import normalize_array


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, default=Path("datasets"))
    ap.add_argument("--out", type=Path, default=Path("runs/feats"))
    ap.add_argument("--encoder", default="vit_base_patch16_dinov3.lvd1689m")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--enc-ckpt", default=None)
    a = ap.parse_args()

    from ds_common import read_manifest
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} (缓存只跑一次, CPU 亦可)")
    model = DINOvSeg(a.encoder, pretrained=bool(a.pretrained),
                     checkpoint_path=a.enc_ckpt).to(device).eval()

    rows = read_manifest(a.datasets / "manifest.csv")
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, r in enumerate(rows):
        name = f"{r['village']}__{r['tile']}.pt"
        dst = a.out / name
        if dst.exists():
            continue
        img = Image.open(a.datasets / r["image_rel"]).convert("RGB")
        x = normalize_array(img)[None].to(device)
        feats = model.encoder(x)
        torch.save([f[0].half().cpu() for f in feats], dst)  # 去 batch 维
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(rows)}  ({el / 60:.1f} min, "
                  f"预计剩余 {(len(rows) - i - 1) * el / (i + 1) / 60:.1f} min)")
    print(f"特征缓存完成 -> {a.out}/  共 {len(rows)} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
