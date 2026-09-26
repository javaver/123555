#!/usr/bin/env python3
"""训练语义分割基线模型 (UNet, PSPNet, DeepLabV3, SegFormer, MaskFormer)。

- SegFormer 默认 MiT (mit_b0..mit_b5); 官方 ImageNet 权重用 --pretrained-from mit_b0.pth
- MaskFormer 用匈牙利匹配损失; 其余基线为逐像素 CE + Dice
- 默认开启 ImageNet 预训练 (--pretrained 1), 便于与论文基线公平对比

用法 (GPU 服务器示例):
    python scripts/train_baseline.py --model deeplabv3 --datasets datasets/ --pretrained 1 --batch 8 --workers 4
    python scripts/train_baseline.py --model pspnet --datasets datasets/ --pretrained 1 --batch 8 --workers 4
    python scripts/train_baseline.py --model segformer --datasets datasets/ \\
        --pretrained-from /root/weights/mit_b0.pth --batch 8 --workers 4
    python scripts/train_baseline.py --model maskformer --datasets datasets/ --pretrained 1 --batch 4 --workers 4
"""
from __future__ import annotations

import argparse
import json
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

# 各模型默认骨干 (写入 ckpt.arch_kwargs, 评估端可复原)
DEFAULT_BACKBONES = {
    "segformer": "mit_b0",
    "pspnet": "resnet50",
    "maskformer": "resnet50",
}


def load_rows(datasets: Path, want: str, key: str = "split_a"):
    splits = {}
    with open(datasets / "splits.csv", encoding="utf-8-sig") as fh:
        import csv
        for row in csv.DictReader(fh):
            splits[(row["village"], row["tile"])] = row
    rows = [
        r
        for r in read_manifest(datasets / "manifest.csv")
        if splits[(r["village"], r["tile"])][key] == want
    ]
    if not rows:
        raise SystemExit(
            f"划分 {key}={want} 为空。请确认 datasets/manifest.csv 与 splits.csv 已在服务器上准备好。"
        )
    return rows


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


def _reseed_aug_rng(worker_id: int) -> None:
    """num_workers>0 时, 各 worker 继承同一份 TileDataset.rng 状态,
    会导致跨 worker 的翻转/旋转增强序列完全重复。
    用 PyTorch 为每个 worker 分配的唯一 seed (逐 epoch 变化) 重播种。"""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng.seed(info.seed)


def save_ckpt(path: Path, model, arch: str, arch_kwargs: dict, metrics: dict, meta: dict):
    torch.save(
        {
            "model": model.state_dict(),
            "arch": arch,
            "arch_kwargs": arch_kwargs,
            "metrics": {k: float(v) for k, v in metrics.items() if np.isscalar(v)},
            "meta": meta,
        },
        path,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="训练语义分割基线模型")
    ap.add_argument("--model", required=True, choices=list(BASELINE_MODELS.keys()))
    ap.add_argument("--datasets", type=Path, default=Path("datasets"))
    ap.add_argument(
        "--pretrained",
        type=int,
        default=1,
        help="timm/torchvision ImageNet 预训练 (0/1)。SegFormer-MiT 请改用 --pretrained-from",
    )
    ap.add_argument(
        "--pretrained-from",
        type=Path,
        default=None,
        help="本地骨干权重路径 (SegFormer 官方 mit_bX.pth; 其它模型一般不需要)",
    )
    ap.add_argument(
        "--backbone",
        default=None,
        help="骨干覆盖 (segformer: mit_b0..b5; pspnet/maskformer: resnet50 等 timm 骨干)",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--crop", type=int, default=None,
        help="训练时随机方形裁剪边长 (如 512)。1024 原图全分辨率显存不够时的标准做法, "
             "验证/测试仍全图; 不设即全图训练",
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=20, help="val mIoU 无提升则早停; 0=关闭")
    ap.add_argument("--split-key", default="split_a", choices=["split_a", "split_b", "split_c"])
    ap.add_argument("--require-cuda", action="store_true", help="无 GPU 则退出 (服务器推荐)")
    ap.add_argument("--out", type=Path, default=None, help="输出目录 (默认 runs/<model>/)")
    a = ap.parse_args()

    if not (a.datasets / "manifest.csv").exists() or not (a.datasets / "splits.csv").exists():
        raise SystemExit(f"缺少 {a.datasets}/manifest.csv 或 splits.csv, 请先在服务器上跑完数据管线。")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.require_cuda and not torch.cuda.is_available():
        raise SystemExit("已指定 --require-cuda, 但当前无可用 GPU。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    tr_rows = load_rows(a.datasets, "train", a.split_key)
    va_rows = load_rows(a.datasets, "val", a.split_key)

    tr_ds = TileDataset(a.datasets, tr_rows, aug=True, seed=a.seed, crop=a.crop)
    va_ds = TileDataset(a.datasets, va_rows)
    tr_dl = DataLoader(
        tr_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
        drop_last=True, pin_memory=device.type == "cuda",
        worker_init_fn=_reseed_aug_rng,
    )
    va_dl = DataLoader(
        va_ds, batch_size=min(4, a.batch), num_workers=a.workers,
        pin_memory=device.type == "cuda",
    )

    num_classes = len(CLASS_NAMES)
    model_kwargs: dict = {}
    if a.backbone:
        model_kwargs["backbone"] = a.backbone
    elif a.model in DEFAULT_BACKBONES:
        model_kwargs["backbone"] = DEFAULT_BACKBONES[a.model]
    if a.pretrained_from is not None:
        if not a.pretrained_from.exists():
            raise SystemExit(f"--pretrained-from 不存在: {a.pretrained_from}")
        model_kwargs["pretrained_from"] = str(a.pretrained_from)

    if a.model == "segformer" and a.pretrained_from is None:
        print(
            "[warn] SegFormer-MiT: --pretrained 对仓库内 MiT 无效; "
            "请下载官方 mit_b0.pth 并用 --pretrained-from 加载, 否则为随机初始化。",
            flush=True,
        )

    build_kwargs = {k: v for k, v in model_kwargs.items() if k != "pretrained_from"}
    # 仅 SegFormer-MiT 支持本地官方权重; 其它模型忽略 --pretrained-from
    if a.model == "segformer" and "pretrained_from" in model_kwargs:
        build_kwargs["pretrained_from"] = model_kwargs["pretrained_from"]
    elif a.pretrained_from is not None and a.model != "segformer":
        print(f"[warn] --pretrained-from 仅用于 SegFormer-MiT, 已忽略 (model={a.model})", flush=True)

    model = get_baseline_model(
        a.model,
        in_channels=3,
        num_classes=num_classes,
        pretrained=bool(a.pretrained),
        **build_kwargs,
    ).to(device)

    # ckpt 只保留可复原架构的 kwargs (路径记在 meta)
    arch_kwargs = {k: v for k, v in model_kwargs.items() if k != "pretrained_from"}

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[{a.model.upper()}] params={n_train / 1e6:.2f}M | train={len(tr_rows)} | val={len(va_rows)} "
        f"| arch_kwargs={arch_kwargs} | pretrained={a.pretrained}",
        flush=True,
    )

    is_mask_cls = a.model in MASK_CLASSIFICATION_BASELINES
    w = class_weights(tr_rows).to(device)
    if is_mask_cls:
        crit = MaskFormerCriterion(num_classes=num_classes, class_weight=w).to(device)
    else:
        crit = SegLoss(weight=w)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = a.out or (ROOT / "runs" / a.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": a.model,
        "split_key": a.split_key,
        "epochs": a.epochs,
        "batch": a.batch,
        "lr": a.lr,
        "seed": a.seed,
        "pretrained": bool(a.pretrained),
        "crop": a.crop,
        "pretrained_from": str(a.pretrained_from) if a.pretrained_from else None,
        "arch_kwargs": arch_kwargs,
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    log_path = run_dir / "log.csv"
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("epoch,loss,mIoU,mF1\n")

    best = -1.0
    stale = 0
    for ep in range(1, a.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if is_mask_cls:
                    pred_logits, pred_masks = model.forward_mask_classification(x)
                    loss, _ = crit(pred_logits, pred_masks, y)
                else:
                    loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
        sched.step()

        res = evaluate(model, va_dl, device, num_classes)
        elapsed = time.time() - t0
        print(
            f"[{a.model}] ep{ep:03d}/{a.epochs:03d} "
            f"loss={np.mean(losses):.4f} val_mIoU={res['mIoU']:.4f} val_mF1={res['mF1']:.4f} ({elapsed:.0f}s)",
            flush=True,
        )
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{ep},{np.mean(losses):.5f},{res['mIoU']:.5f},{res['mF1']:.5f}\n")

        save_ckpt(run_dir / "last.pt", model, a.model, arch_kwargs, res, {**meta, "epoch": ep})
        if res["mIoU"] > best:
            best = res["mIoU"]
            stale = 0
            save_ckpt(run_dir / "best.pt", model, a.model, arch_kwargs, res, {**meta, "epoch": ep, "best": best})
            print(f"  ^ [{a.model}] best mIoU={best:.4f} -> {run_dir / 'best.pt'}", flush=True)
        else:
            stale += 1
            if a.patience > 0 and stale >= a.patience:
                print(f"[{a.model}] early stop @ ep{ep} (patience={a.patience}), best={best:.4f}", flush=True)
                break

    print(f"[{a.model}] done. best val mIoU={best:.4f} -> {run_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
