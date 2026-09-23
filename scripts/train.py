#!/usr/bin/env python3
"""训练 DINO-Seg。

用法:
    python scripts/train.py --datasets datasets/ [--config configs/dinoseg_vitb16_512.yaml]
    CPU 冒烟: python scripts/train.py --datasets datasets/ --pretrained 0 --epochs 1 --batch 2
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ds_common import CLASS_NAMES, read_manifest, slug
from dinoseg import ConfusionMeter, DINOvSeg, SegLoss
from dinoseg.dataset import TileDataset


def load_rows(datasets: Path, want: str, key: str = "split_a"):
    splits = {}
    with open(datasets / "splits.csv", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            splits[(row["village"], row["tile"])] = row
    return [r for r in read_manifest(datasets / "manifest.csv")
            if splits[(r["village"], r["tile"])][key] == want]


def class_weights(rows) -> torch.Tensor:
    tot = np.array([sum(int(r[f"px_{slug(n)}"]) for r in rows) for n in CLASS_NAMES], float)
    tot = np.clip(tot, 0, None)
    pos = tot[tot > 0]
    med = np.median(pos) if pos.size else 1.0
    w = np.ones_like(tot)
    nz = tot > 0
    w[nz] = np.clip(med / tot[nz], 1.0, 20.0)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, num_classes) -> dict:
    model.eval()
    meter = ConfusionMeter(num_classes)
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        meter.update(logits.argmax(1).cpu().numpy(), y.numpy())
    return meter.result()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, default=Path("datasets"))
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run", default="dinoseg")
    ap.add_argument("--split-key", default="split_a", choices=["split_a", "split_c"])
    a = ap.parse_args()

    cfg = {"encoder": "vit_base_patch16_dinov3.lvd1689m", "epochs": 120,
           "batch": 8, "lr": 8e-4, "num_classes": 10}
    if a.config:
        cfg.update(yaml.safe_load(open(a.config, encoding="utf-8")))
    for k in ("encoder", "epochs", "batch", "lr"):
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_rows = load_rows(a.datasets, "train", a.split_key)
    va_rows = load_rows(a.datasets, "val", a.split_key)
    tr_ds = TileDataset(a.datasets, tr_rows, aug=True, seed=a.seed)
    va_ds = TileDataset(a.datasets, va_rows)
    tr_dl = DataLoader(tr_ds, batch_size=cfg["batch"], shuffle=True,
                       num_workers=0, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=2, num_workers=0)

    model = DINOvSeg(cfg["encoder"], pretrained=bool(a.pretrained),
                     num_classes=cfg["num_classes"]).to(device)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"参数: 可训练 {n_train / 1e6:.2f}M / 总 {n_all / 1e6:.2f}M (编码器冻结)")

    w = class_weights(tr_rows).to(device)
    print("类权重:", {n: round(float(x), 2) for n, x in zip(CLASS_NAMES, w)})
    crit = SegLoss(weight=w)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = ROOT / "runs" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        fh.write("epoch,loss,mIoU,mF1\n")

    best = -1.0
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        t0 = time.time()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
        sched.step()
        res = evaluate(model, va_dl, device, cfg["num_classes"])
        line = f"ep{ep:03d} loss={np.mean(losses):.4f} mIoU={res['mIoU']:.4f} " \
               f"mF1={res['mF1']:.4f} ({time.time() - t0:.0f}s)"
        print(line)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{ep},{np.mean(losses):.5f},{res['mIoU']:.5f},{res['mF1']:.5f}\n")
        if res["mIoU"] > best:
            best = res["mIoU"]
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "metrics": {k: float(v) for k, v in res.items() if np.isscalar(v)}},
                       run_dir / "best.pt")
            print(f"  ^ 保存最佳 mIoU={best:.4f}")
    print(f"训练完成, 最佳 val mIoU={best:.4f}, ckpt -> {run_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
