#!/usr/bin/env python3
"""训练 DINO-Seg。

用法:
    python scripts/train.py --datasets datasets/ [--config configs/dinoseg_vitb16_512.yaml]
    CPU 冒烟: python scripts/train.py --datasets datasets/ --pretrained 0 --epochs 1 --batch 2

与基线对齐的协议 (推荐, 见 scripts/train_all_baselines.sh 的 WITH_DINOSEG=1):
    python scripts/train.py --datasets datasets/ --crop 512 --epochs 80 --patience 20 \
        --batch 8 --workers 4 --out runs/dinoseg_aligned
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
from dinoseg.dataset import FeatureDataset, TileDataset


def _reseed_aug_rng(worker_id: int) -> None:
    """num_workers>0 时各 worker 继承同一份 Dataset.rng 状态, 会导致跨 worker
    增强 (及随机裁剪) 序列完全重复; 用 PyTorch 分配的唯一 seed 重播种。"""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng.seed(info.seed)


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
        if isinstance(x, list):
            logits = model.forward_feats([t.to(device) for t in x], y.shape[-2:])
        else:
            logits = model(x.to(device))
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
    ap.add_argument("--out", type=Path, default=None,
                    help="输出目录 (默认 runs/<run>/; Split B/C 用 runs_b/dinoseg 等隔离)")
    ap.add_argument("--split-key", default="split_a", choices=["split_a", "split_b", "split_c"])
    ap.add_argument("--crop", type=int, default=None,
                    help="训练期随机方形裁剪边长 (如 512, 与基线协议对齐); 不设即全图。"
                         "仅 TileDataset 端到端模式有效, --cache-dir 下忽略")
    ap.add_argument("--patience", type=int, default=0,
                    help="val mIoU 无提升则早停; 0=关闭 (与基线对齐用 20)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--enc-ckpt", default=None, help="本地编码器权重 model.safetensors (HF 不通时用)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="预计算特征目录 (cache_features.py 产出); 设置后训练不跑编码器")
    a = ap.parse_args()

    if a.cache_dir is not None and a.crop:
        print("[warn] --cache-dir 模式的特征为全图预计算, --crop 被忽略。", flush=True)

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
    if a.cache_dir:
        tr_ds = FeatureDataset(a.datasets, tr_rows, a.cache_dir, aug=True, seed=a.seed)
        va_ds = FeatureDataset(a.datasets, va_rows, a.cache_dir)
    else:
        tr_ds = TileDataset(a.datasets, tr_rows, aug=True, seed=a.seed, crop=a.crop)
        va_ds = TileDataset(a.datasets, va_rows)
    tr_dl = DataLoader(tr_ds, batch_size=cfg["batch"], shuffle=True,
                       num_workers=a.workers, drop_last=False,
                       worker_init_fn=_reseed_aug_rng)
    va_dl = DataLoader(va_ds, batch_size=2, num_workers=a.workers)

    model = DINOvSeg(cfg["encoder"], pretrained=bool(a.pretrained),
                     num_classes=cfg["num_classes"],
                     checkpoint_path=a.enc_ckpt).to(device)
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

    run_dir = a.out or (ROOT / "runs" / a.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.update({"crop": a.crop, "patience": a.patience, "split_key": a.split_key, "seed": a.seed})
    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        fh.write("epoch,loss,mIoU,mF1\n")

    best = -1.0
    stale = 0
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        t0 = time.time()
        for x, y in tr_dl:
            y = y.to(device)
            if isinstance(x, list):
                x = [t.to(device) for t in x]
                fwd = lambda: model.forward_feats(x, y.shape[-2:])
            else:
                x = x.to(device)
                fwd = lambda: model(x)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(fwd(), y)
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
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{ep},{np.mean(losses):.5f},{res['mIoU']:.5f},{res['mF1']:.5f}\n")

        torch.save({"model": model.state_dict(), "cfg": cfg,
                    "metrics": {k: float(v) for k, v in res.items() if np.isscalar(v)}},
                   run_dir / "last.pt")
        if res["mIoU"] > best:
            best = res["mIoU"]
            stale = 0
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "metrics": {k: float(v) for k, v in res.items() if np.isscalar(v)}},
                       run_dir / "best.pt")
            print(f"  ^ 保存最佳 mIoU={best:.4f}", flush=True)
        else:
            stale += 1
            if a.patience > 0 and stale >= a.patience:
                print(f"early stop @ ep{ep} (patience={a.patience}), best={best:.4f}", flush=True)
                break
    print(f"训练完成, 最佳 val mIoU={best:.4f}, ckpt -> {run_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
