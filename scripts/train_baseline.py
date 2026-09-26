#!/usr/bin/env python3
"""训练语义分割基线模型 (UNet, DeepLabV3, DPT, SegFormer, MaskFormer)。

- SegFormer 默认使用仓库内实现的 MiT (Mix Transformer) 编码器 (mit_b0..mit_b5),
  可用 --backbone 覆盖 (兼容 timm 骨干, 如旧版 pvt_v2_b0);
- MaskFormer 训练采用二分图匹配 (匈牙利算法) 损失 (baselines/maskformer/criterion.py),
  其余基线为逐像素 CE + Dice。

用法:
    python scripts/train_baseline.py --model unet --datasets datasets/ --batch 16 --epochs 80
    python scripts/train_baseline.py --model deeplabv3 --datasets datasets/ --batch 16 --epochs 80
    python scripts/train_baseline.py --model dpt --datasets datasets/ --batch 16 --epochs 80
    python scripts/train_baseline.py --model segformer --datasets datasets/ --batch 16 --epochs 80
    python scripts/train_baseline.py --model segformer --backbone mit_b2 --datasets datasets/
    python scripts/train_baseline.py --model maskformer --datasets datasets/ --batch 16 --epochs 80
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
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from ds_common import CLASS_NAMES, read_manifest, slug
from dinoseg import ConfusionMeter, SegLoss
from dinoseg.dataset import TileDataset
from baselines import get_baseline_model, BASELINE_MODELS, MASK_CLASSIFICATION_BASELINES
from baselines.maskformer.criterion import MaskFormerCriterion


def load_rows(datasets: Path, want: str, key: str = "split_a"):
    splits = {}
    with open(datasets / "splits.csv", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            splits[(row["village"], row["tile"])] = row
    return [
        r
        for r in read_manifest(datasets / "manifest.csv")
        if splits[(r["village"], r["tile"])][key] == want
    ]


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
        logits = model(x.to(device))
        meter.update(logits.argmax(1).cpu().numpy(), y.numpy())
    return meter.result()


def main() -> int:
    ap = argparse.ArgumentParser(description="训练语义分割基线模型")
    ap.add_argument(
        "--model",
        required=True,
        choices=list(BASELINE_MODELS.keys()),
        help=f"模型选择: {list(BASELINE_MODELS.keys())}",
    )
    ap.add_argument("--datasets", type=Path, default=Path("datasets"))
    ap.add_argument("--pretrained", type=int, default=0, help="是否使用 ImageNet 预训练权重 (0/1)")
    ap.add_argument(
        "--backbone",
        default=None,
        help="骨干网络覆盖 (如 segformer: mit_b0..mit_b5 / pvt_v2_b0; maskformer: resnet50 等 timm 骨干)",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-key", default="split_a", choices=["split_a", "split_c"])
    ap.add_argument("--out", type=Path, default=None, help="输出目录 (默认 runs/<model>/)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_rows = load_rows(a.datasets, "train", a.split_key)
    va_rows = load_rows(a.datasets, "val", a.split_key)

    tr_ds = TileDataset(a.datasets, tr_rows, aug=True, seed=a.seed)
    va_ds = TileDataset(a.datasets, va_rows)
    tr_dl = DataLoader(tr_ds, batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=min(4, a.batch), num_workers=0)

    num_classes = len(CLASS_NAMES)
    model_kwargs = {}
    if a.backbone:
        model_kwargs["backbone"] = a.backbone
    model = get_baseline_model(
        a.model, in_channels=3, num_classes=num_classes, pretrained=bool(a.pretrained), **model_kwargs
    ).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{a.model.upper()}] 参数量: {n_train / 1e6:.2f}M | 训练集: {len(tr_rows)} | 验证集: {len(va_rows)}")

    # 掩膜分类范式 (MaskFormer): 二分图匹配 (匈牙利) 损失; 其余基线: 逐像素 CE+Dice
    is_mask_cls = a.model in MASK_CLASSIFICATION_BASELINES
    w = class_weights(tr_rows).to(device)
    crit = (
        MaskFormerCriterion(num_classes=num_classes, class_weight=w)
        if is_mask_cls
        else SegLoss(weight=w)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = a.out or (ROOT / "runs" / a.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        fh.write("epoch,loss,mIoU,mF1\n")

    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                if is_mask_cls:
                    # 匈牙利匹配训练: (类别 logits (B,N,K+1), 掩膜 logits (B,N,h,w))
                    pred_logits, pred_masks = model.forward_mask_classification(x)
                    loss, _ = crit(pred_logits, pred_masks, y)
                else:
                    out = model(x)
                    loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
        sched.step()

        res = evaluate(model, va_dl, device, num_classes)
        elapsed = time.time() - t0
        line = (
            f"[{a.model}] ep{ep:03d}/{a.epochs:03d} "
            f"loss={np.mean(losses):.4f} val_mIoU={res['mIoU']:.4f} val_mF1={res['mF1']:.4f} ({elapsed:.0f}s)"
        )
        print(line)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{ep},{np.mean(losses):.5f},{res['mIoU']:.5f},{res['mF1']:.5f}\n")

        if res["mIoU"] > best:
            best = res["mIoU"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": a.model,
                    "arch_kwargs": model_kwargs,  # 骨干等架构参数 (如 mit_b0), 供评估端复原
                    "metrics": {k: float(v) for k, v in res.items() if np.isscalar(v)},
                },
                run_dir / "best.pt",
            )
            print(f"  ^ [{a.model}] 保存最佳权重 mIoU={best:.4f} -> {run_dir / 'best.pt'}")

    print(f"[{a.model}] 训练完成! 最佳 val mIoU={best:.4f}, 检查点 -> {run_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
