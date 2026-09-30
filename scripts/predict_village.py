#!/usr/bin/env python3
"""整幅大图村级识别一条龙: 切块 → DINO-Seg 推理 → 拼回整幅 → 上色 → 面积统计。

替代原先的五段式流程 (split_tiff → infer → colorize → reassemble → area):
模型加载、切块、无缝拼回、编号映射、上色、面积统计一次完成。

用法:
    # 单张大图 (jpg / png / tif)
    python scripts/predict_village.py \\
        --image F:/segment/data_process/yingshancun.jpg \\
        --ckpt  runs/dinoseg/best.pt \\
        --out   F:/segment/out --scale 120

    # --image 传目录: 对目录下每张图各出一套结果, 并汇总 area_summary.csv

关键参数:
    --tile   1024   切块边长 (模型训练分辨率 1024, 不建议改小; 512 会丢全局上下文)
    --stride 1024   滑动步长; 设小于 tile (如 896) 重叠拼接缓解接缝 (重叠区后写覆盖)
    --scale  120    比例尺: 多少像素 = 10 米, 用于面积(m²)换算; 不填则只出占比
    --device auto   cuda / cpu / auto
    --skip-overlay  超大图内存不足时跳过叠加预览图

输出编号体系 (已自动映射为原 11 类体系, 可直接对接旧下游代码):
    0 = 背景/村界外(黑边), 1..10 = 裸土/耕地/高铁高速/山林/荒山/新建建筑/
                              老旧建筑/一般道路/散生树木/水体
    (模型内部编号为 0..9 + 255 忽略, 本脚本输出时已整体 +1)

输出 (out/<图名>/):
    mask_full.png     整幅单通道掩膜, 与原图逐像素同尺寸
    color_full.png    按标注色标的纯彩色分类图
    overlay_full.png  原图 50% + 彩色 50% 叠加预览 (--skip-overlay 跳过)
    area.csv          逐类像素数 / 占比 / 面积 (UTF-8-BOM, Excel 直接打开)
    多张大图时另汇总 out/area_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # 允许读取超大正射影像

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# (模型ID 0..9, 输出ID 1..10, 英文名, 中文名) —— 输出ID = 模型ID + 1
CLASSES = [
    (0, 1, "bare soil", "裸土"),
    (1, 2, "cultivated land", "耕地"),
    (2, 3, "high speed rail and highway", "高铁与高速公路"),
    (3, 4, "mountain forest", "山林"),
    (4, 5, "naked mountain", "荒山"),
    (5, 6, "new building", "新建建筑"),
    (6, 7, "old building", "老旧建筑"),
    (7, 8, "road", "一般道路"),
    (8, 9, "tree", "散生树木"),
    (9, 10, "water", "水体"),
]
PALETTE = {  # 输出ID -> RGB, 与标注色标一致
    1: (201, 162, 39), 2: (124, 179, 66), 3: (229, 57, 53), 4: (46, 125, 50),
    5: (141, 110, 99), 6: (255, 152, 0), 7: (109, 76, 65), 8: (158, 158, 158),
    9: (27, 94, 32), 10: (30, 136, 229), 0: (0, 0, 0),
}
IMG_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


def load_model(ckpt: Path, device):
    """加载 best.pt (自包含权重, 无需联网)"""
    import torch
    from dinoseg import DINOvSeg

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = DINOvSeg(cfg["encoder"], pretrained=False, num_classes=cfg["num_classes"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def predict_tile(model, pil_img: Image.Image, device) -> np.ndarray:
    """单块推理: 返回模型编号 0..9; 黑边/无效像素 = 255 (与 infer.py 同规则)"""
    import torch
    import torch.nn.functional as F
    from dinoseg.dataset import normalize_array

    x = normalize_array(pil_img)[None].to(device)
    h0, w0 = x.shape[-2:]
    ph, pw = (16 - h0 % 16) % 16, (16 - w0 % 16) % 16
    if ph or pw:  # 任意尺寸: reflect 补齐到 16 的倍数, 推完裁回
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
    pred = pred[:h0, :w0]
    arr = np.asarray(pil_img)
    pred[arr.max(axis=2) <= 5] = 255  # 近黑填充区 -> 忽略
    return pred


def predict_full(model, arr: np.ndarray, device, tile=1024, stride=1024):
    """整幅切块推理后拼回: 返回模型编号掩膜 (0..9, 255=黑边), 尺寸与原图一致。

    边缘块"向内回缩"取满 tile 大小 (不补零, 避免黑边伪影); stride < tile 时
    重叠区由后处理的块覆盖, 可缓解接缝。
    """
    H, W = arr.shape[:2]
    stride = min(stride, tile)
    full = np.full((H, W), 255, np.uint8)
    tops, lefts = list(range(0, H, stride)), list(range(0, W, stride))
    n = len(tops) * len(lefts)
    k = 0
    t0 = time.time()
    for top in tops:
        for left in lefts:
            bottom, right = min(top + tile, H), min(left + tile, W)
            top2, left2 = max(0, bottom - tile), max(0, right - tile)
            patch = Image.fromarray(arr[top2:bottom, left2:right])
            full[top2:bottom, left2:right] = predict_tile(model, patch, device)
            k += 1
            if k % 20 == 0 or k == n:
                print(f"    块 {k}/{n}  ({time.time() - t0:.0f}s)")
    return full


def remap_to_legacy(full: np.ndarray) -> np.ndarray:
    """模型编号 (0..9, 255忽略) -> 原体系 (0=背景/村界外, 1..10=要素)"""
    return np.where(full == 255, 0, full.astype(np.int16) + 1).astype(np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    for k, c in PALETTE.items():
        rgb[mask == k] = c
    return rgb


def area_rows(mask: np.ndarray, scale: float | None) -> list[dict]:
    total = mask.size
    valid = int((mask > 0).sum())
    px_area = (10.0 / scale) ** 2 if scale else None  # scale 像素 = 10 米
    rows = [{
        "class_id": 0, "name_en": "background", "name_cn": "背景/村界外",
        "pixels": int((mask == 0).sum()),
        "pct_of_image": round(100 * (mask == 0).sum() / total, 4),
        "pct_of_valid": 0.0,
        **({"area_m2": round(int((mask == 0).sum()) * px_area, 2)} if px_area else {}),
    }]
    for _mid, oid, en, cn in CLASSES:
        cnt = int((mask == oid).sum())
        rows.append({
            "class_id": oid, "name_en": en, "name_cn": cn, "pixels": cnt,
            "pct_of_image": round(100 * cnt / total, 4),
            "pct_of_valid": round(100 * cnt / valid, 4) if valid else 0.0,
            **({"area_m2": round(cnt * px_area, 2)} if px_area else {}),
        })
    return rows


def write_csv(path: Path, rows: list[dict], extra_first_col: tuple[str, str] | None = None):
    if not rows:
        return
    if extra_first_col:
        k, v = extra_first_col
        rows = [{k: v, **r} for r in rows]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True, help="大图文件或目录")
    ap.add_argument("--ckpt", type=Path, default=Path("runs/dinoseg/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("runs/pred_village"))
    ap.add_argument("--tile", type=int, default=1024, help="切块边长 (默认 1024=训练分辨率)")
    ap.add_argument("--stride", type=int, default=None,
                    help="滑动步长 (默认=tile 即无重叠; 设 896 之类可重叠拼接)")
    ap.add_argument("--scale", type=float, default=None,
                    help="多少像素=10米, 用于面积(m²)换算; 不填只出占比")
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--skip-overlay", action="store_true", help="超大图内存不足时跳过叠加图")
    a = ap.parse_args()

    import torch
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    stride = a.stride or a.tile

    if a.image.is_file():
        paths = [a.image]
    elif a.image.is_dir():
        paths = sorted(p for p in a.image.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not paths:
            print(f"{a.image} 下没有图像文件 ({'/'.join(IMG_EXTS)})", file=sys.stderr)
            return 2
    else:
        print(f"路径不存在: {a.image}", file=sys.stderr)
        return 2

    print(f"加载模型: {a.ckpt} (device={a.device})")
    model = load_model(a.ckpt, a.device)

    all_rows = []
    for p in paths:
        print(f"\n[{p.name}] 读取 ...")
        arr = np.asarray(Image.open(p).convert("RGB"))
        H, W = arr.shape[:2]
        print(f"  {W}x{H}px, tile={a.tile}, stride={stride}"
              + (f", scale={a.scale:g}px/10m" if a.scale else ""))
        full = predict_full(model, arr, a.device, a.tile, stride)
        mask = remap_to_legacy(full)

        od = a.out / p.stem
        od.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(od / "mask_full.png")
        if not a.skip_overlay:
            rgb = colorize(mask)
            Image.fromarray(rgb).save(od / "color_full.png")
            Image.fromarray((0.5 * arr + 0.5 * rgb).astype(np.uint8)).save(
                od / "overlay_full.png")

        rows = area_rows(mask, a.scale)
        write_csv(od / "area.csv", rows)
        all_rows.extend({"image": p.name, **r} for r in rows)

        unit = "m²" if a.scale else "%"
        print(f"  -> {od}/  (mask_full / color_full / overlay_full / area.csv)")
        print(f"  有效区域占比 {100 * (mask > 0).mean():.1f}%, 逐类 "
              + ", ".join(f"{r['name_cn']}={r['area_m2'] if a.scale else r['pct_of_valid']:,.1f}"
                          for r in rows[1:]))

    if len(paths) > 1:
        write_csv(a.out / "area_summary.csv", all_rows)
        print(f"\n[OK] 汇总 -> {a.out / 'area_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
