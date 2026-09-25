#!/usr/bin/env python3
"""检查每个村的瓦片网格步长, 判断相邻航片有无像素重叠。

用法:
    python tools/tiling_check.py datasets/

判读:
  * min_dx / min_dy = 512  => 无重叠网格切分(相邻片只共享边界);
  * = 256 等 <512          => 重叠切分, 相邻片共享像素, 随机划分存在真重复泄漏;
  * 偏移非 512 整数倍      => 非规则网格, 需人工确认。
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from ds_common import read_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", type=Path)
    a = ap.parse_args()

    grid: dict[str, set] = defaultdict(set)
    n_bad = 0
    for r in read_manifest(a.datasets / "manifest.csv"):
        m = re.match(r"tile_(\d+)_(\d+)", r["tile"])
        if not m:
            n_bad += 1
            continue
        grid[r["village"]].add(tuple(map(int, m.groups())))

    print(f"{'village':20s} {'n':>5s} {'min_dx':>7s} {'min_dy':>7s}  结论")
    for v, s in sorted(grid.items()):
        xs = sorted({p[0] for p in s})
        ys = sorted({p[1] for p in s})
        dx = min((b - x for x, b in zip(xs, xs[1:])), default=None)
        dy = min((b - y for y, b in zip(ys, ys[1:])), default=None)
        if dx is None or dy is None:
            verdict = "瓦片太少, 无法判断"
        elif dx == 512 and dy == 512:
            verdict = "无重叠网格"
        elif dx < 512 or dy < 512:
            verdict = "!! 重叠切分(相邻片共享像素)"
        else:
            verdict = "步长>512, 有间隔"
        off_ok = all(p[0] % 512 == 0 and p[1] % 512 == 0 for p in s)
        print(f"{v:20s} {len(s):5d} {str(dx):>7s} {str(dy):>7s}  {verdict}"
              + ("" if off_ok else " (偏移非512整数倍!)"))
    if n_bad:
        print(f"  [warn] {n_bad} 张瓦片名不符合 tile_x_y, 未参与统计")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
