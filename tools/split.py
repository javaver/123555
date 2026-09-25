#!/usr/bin/env python3
"""训练/验证/测试划分, 写 datasets/splits.csv。

用法:
    python tools/split.py datasets/ [--seed 42] [--ratios 0.7 0.15 0.15]

划分 A (对标论文口径): 按瓦片分层随机——分层键 = 该图的主导类别
  (像素数最多的类, 约 10 个层, 避免"类别集合签名"碎片化),
  层内 shuffle 后按 70/15/15 切; 并打印各折的类别像素占比平衡报告。
划分 B (泛化口径): 村庄级留出——shuffle 村庄, 前 2 个做 test, 第 3 个做 val,
  其余 train; 村庄数不足时降级并告警。
划分 C (空间去偏口径): 1024x1024 空间块(2x2 瓦片)整块进同一折,
  按村以瓦片数贪心配 70/15/15, 消除随机切分下邻片同折泄漏造成的指标虚高。
类别权重等统计只允许用 split_a==train 的样本(下游自行过滤)。
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

from ds_common import CLASS_NAMES, read_manifest, slug


def class_px(r: dict) -> dict:
    return {n: int(r[f"px_{slug(n)}"]) for n in CLASS_NAMES}


def dominant_class(r: dict) -> str:
    px = class_px(r)
    return max(px, key=px.get) if sum(px.values()) else "__empty__"


def split_a(rows: list[dict], ratios: list[float], rng: random.Random) -> dict:
    tr, va, te = ratios
    out = {}
    strata: dict[str, list[dict]] = {}
    for r in rows:
        strata.setdefault(dominant_class(r), []).append(r)

    for dom, members in sorted(strata.items()):
        rng.shuffle(members)
        n = len(members)
        if n >= 3:
            n_te = max(1, round(n * te))
            n_va = max(1, round(n * va))
            n_va = min(n_va, n - n_te - 1)
            test, val = members[:n_te], members[n_te:n_te + n_va]
            train = members[n_te + n_va:]
        elif n == 2:
            test, val, train = members[:1], [], members[1:]
            print(f"  [warn] 层 {dom} 仅 2 样本: 1 train / 1 test, 无 val")
        else:
            test, val, train = [], [], members
            print(f"  [warn] 层 {dom} 仅 1 样本: 全部进 train")
        for r in train:
            out[(r["village"], r["tile"])] = "train"
        for r in val:
            out[(r["village"], r["tile"])] = "val"
        for r in test:
            out[(r["village"], r["tile"])] = "test"
    return out


def split_b(rows: list[dict], rng: random.Random) -> dict:
    villages = sorted({r["village"] for r in rows})
    rng.shuffle(villages)
    if len(villages) >= 4:
        test_v, val_v = set(villages[:2]), {villages[2]}
    elif len(villages) == 3:
        test_v, val_v = {villages[0]}, {villages[1]}
        print("  [warn] 村庄数=3, 划分 B 降级为 1/1/1")
    else:
        print("  [warn] 村庄数<3, 划分 B 全部置 train")
        test_v, val_v = set(), set()
    out = {}
    for r in rows:
        v = r["village"]
        out[(v, r["tile"])] = ("test" if v in test_v else "val" if v in val_v else "train")
    return out


def block_key(r: dict, block: int = 1024):
    m = re.match(r"tile_(\d+)_(\d+)", r["tile"])
    if not m:
        return (r["village"], r["tile"])  # 名字不规则: 自成一块
    x, y = map(int, m.groups())
    return (r["village"], x // block, y // block)


def split_c(rows: list[dict], ratios: list[float], rng: random.Random) -> dict:
    blocks: dict[tuple, list[dict]] = {}
    for r in rows:
        blocks.setdefault(block_key(r), []).append(r)
    per_village: dict[str, list] = {}
    for k, members in blocks.items():
        per_village.setdefault(k[0], []).append(members)

    out = {}
    for v, blks in per_village.items():
        rng.shuffle(blks)
        total = sum(len(m) for m in blks)
        targets = {"train": total * ratios[0], "val": total * ratios[1],
                   "test": total * ratios[2]}
        cnt = {"train": 0, "val": 0, "test": 0}
        for members in blks:  # 整块分给"缺口最大"的折
            s = max(cnt, key=lambda k: targets[k] - cnt[k])
            cnt[s] += len(members)
            for r in members:
                out[(r["village"], r["tile"])] = s
        if cnt["val"] == 0 or cnt["test"] == 0:
            print(f"  [warn] 划分 C 村 {v} 块数过少, 折分布 {cnt}")
    return out


def report_balance(rows: list[dict], assign: dict, name: str) -> None:
    tot = {n: 0 for n in CLASS_NAMES}
    per = {s: {n: 0 for n in CLASS_NAMES} for s in ("train", "val", "test")}
    for r in rows:
        s = assign[(r["village"], r["tile"])]
        for n, v in class_px(r).items():
            tot[n] += v
            per[s][n] += v
    grand = max(sum(tot.values()), 1)
    print(f"划分 {name} 各类像素占比 (%):")
    print(f"    {'class':32s} {'all':>6s} {'train':>6s} {'val':>6s} {'test':>6s}")
    for n in CLASS_NAMES:
        shares = {s: 100 * per[s][n] / max(sum(per[s].values()), 1)
                  for s in ("train", "val", "test")}
        flag = "  <-- 某折缺失" if any(per[s][n] == 0 for s in per) else ""
        print(f"    {n:32s} {100 * tot[n] / grand:6.2f} "
              f"{shares['train']:6.2f} {shares['val']:6.2f} {shares['test']:6.2f}{flag}")
    missing = [n for n in CLASS_NAMES if any(per[s][n] == 0 for s in per)]
    if missing:
        print(f"  [warn] 划分 {name} 有折缺类: {missing}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    a = ap.parse_args()

    mpath = a.datasets / "manifest.csv"
    if not mpath.exists():
        print(f"先运行 ingest.py ({mpath} 不存在)", file=sys.stderr)
        return 2
    rows = read_manifest(mpath)
    if any(r["mask_rel"] == "" for r in rows):
        print("先运行 build_masks.py (manifest 中存在无掩膜样本)", file=sys.stderr)
        return 2

    rngs = {k: random.Random(a.seed + i) for i, k in enumerate(("a", "b", "c"))}
    sa = split_a(rows, a.ratios, rngs["a"])
    sb = split_b(rows, rngs["b"])
    sc = split_c(rows, a.ratios, rngs["c"])

    out = a.datasets / "splits.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write("village,tile,split_a,split_b,split_c\n")
        for r in rows:
            k = (r["village"], r["tile"])
            fh.write(f"{r['village']},{r['tile']},{sa[k]},{sb[k]},{sc[k]}\n")

    for name, m in (("A", sa), ("B", sb), ("C", sc)):
        cnt = {"train": 0, "val": 0, "test": 0}
        for v in m.values():
            cnt[v] += 1
        print(f"划分 {name}: train={cnt['train']} val={cnt['val']} test={cnt['test']}")
    report_balance(rows, sa, "A")
    report_balance(rows, sc, "C")
    print(f"splits -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
