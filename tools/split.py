#!/usr/bin/env python3
"""训练/验证/测试划分, 写 datasets/splits.csv。

用法:
    python tools/split.py datasets/ [--seed 42] [--ratios 0.7 0.15 0.15]

划分 A (对标论文口径): 按瓦片分层随机——分层键 = 该图出现的类别集合,
  层内 shuffle 后按比例切; 小层(<=2)尽量保证 test 有样本, 其余进 train 并告警。
划分 B (泛化口径): 村庄级留出——shuffle 村庄, 前 2 个做 test, 第 3 个做 val,
  其余 train; 村庄数不足时降级并告警。
类别权重等统计只允许用 split_a==train 的样本(下游自行过滤)。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from ds_common import CLASS_NAMES, read_manifest, slug


def split_a(rows: list[dict], ratios: list[float], rng: random.Random) -> dict:
    tr, va, te = ratios
    out = {}
    strata: dict[tuple, list[dict]] = {}
    for r in rows:
        sig = tuple(n for n in CLASS_NAMES if int(r[f"px_{slug(n)}"]) > 0)
        strata.setdefault(sig or ("__empty__",), []).append(r)

    for sig, members in strata.items():
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
            print(f"  [warn] 层 {sig} 仅 2 样本: 1 train / 1 test, 无 val")
        else:
            test, val, train = [], [], members
            print(f"  [warn] 层 {sig} 仅 1 样本: 全部进 train")
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

    rng_a, rng_b = random.Random(a.seed), random.Random(a.seed + 1)
    sa = split_a(rows, a.ratios, rng_a)
    sb = split_b(rows, rng_b)

    out = a.datasets / "splits.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write("village,tile,split_a,split_b\n")
        for r in rows:
            k = (r["village"], r["tile"])
            fh.write(f"{r['village']},{r['tile']},{sa[k]},{sb[k]}\n")

    for name, m in (("A", sa), ("B", sb)):
        cnt = {"train": 0, "val": 0, "test": 0}
        for v in m.values():
            cnt[v] += 1
        print(f"划分 {name}: train={cnt['train']} val={cnt['val']} test={cnt['test']}")
    # 划分 A test 的类别覆盖检查
    test_keys = {k for k, v in sa.items() if v == "test"}
    covered = set()
    for r in rows:
        if (r["village"], r["tile"]) in test_keys:
            covered |= {n for n in CLASS_NAMES if int(r[f"px_{slug(n)}"]) > 0}
    miss = set(CLASS_NAMES) - covered
    if miss:
        print(f"  [warn] 划分 A test 未覆盖类别: {sorted(miss)} (稀有类样本量不足时属预期)")
    print(f"splits -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
