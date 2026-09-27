#!/usr/bin/env python3
"""聚合多个数据划分 (Split A/B/C) 的对比表, 输出 mean±std 汇总 (论文最终表)。

用法:
    python scripts/aggregate_splits.py --runs runs --runs runs_b --runs runs_c
    python scripts/aggregate_splits.py --runs runs --runs runs_b --runs runs_c --perclass

读取各 <runs>/comparison_table{1,2}.csv (compare_all.py 产出), 只聚合 (eval) 行;
(paper)/(n/a) 行跳过并警示 (口径不同, 不可与本地评测混算)。
输出: 终端表格 + <首个 runs>/splits_summary.csv; --perclass 另出逐类 IoU mean±std。
标签自动识别: runs_b -> B, runs_c -> C, runs -> 按传入顺序 A/B/C。
std 为样本标准差 (n-1); 某模型缺某个 split 时该格显示 '-' 且不参与该模型统计。
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path


def _strip_tag(method: str) -> str | None:
    """返回去标签后的方法名; 非 (eval) 行返回 None。"""
    for tag in ("(eval)", "(paper)", "(n/a)"):
        if method.endswith(tag):
            return method[: -len(tag)].strip() if tag == "(eval)" else None
    return None


def read_table2(path: Path):
    """-> {method: {类名: IoU, ..., 'mIoU': v}}, [跳过的行]"""
    out, skipped = {}, []
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = _strip_tag(row["Method"])
            if name is None:
                skipped.append(row["Method"])
                continue
            out[name] = {k: float(v) for k, v in row.items() if k != "Method"}
    return out, skipped


def read_table1(path: Path):
    """-> {method: {metric: Avg}}"""
    out = {}
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = _strip_tag(row["Method"])
            if name is None:
                continue
            out.setdefault(name, {})[row["Metric"]] = float(row["Avg"])
    return out


def ms(vals):
    """mean±std 字符串; 少于 2 个有效值时 std 记 '-'。"""
    if not vals:
        return "-", "-"
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) >= 2 else None
    return mean, std


def fmt_ms(mean, std, nd=4):
    s = f"{mean:.{nd}f}" if isinstance(mean, (int, float)) else "-"
    if isinstance(std, (int, float)):
        s += f"±{std:.{nd}f}"
    elif s != "-":
        s += "  (n=1)"
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="聚合 Split A/B/C 对比表 -> mean±std")
    ap.add_argument("--runs", action="append", required=True,
                    help="各 split 的 runs 目录 (可重复, 如 --runs runs --runs runs_b --runs runs_c)")
    ap.add_argument("--perclass", action="store_true", help="另出逐类 IoU mean±std 表")
    a = ap.parse_args()

    labels = []
    for i, r in enumerate(a.runs):
        base = Path(r).name
        if base.startswith("runs_"):
            labels.append(base[5:].upper())
        elif base in ("runs", ""):
            labels.append(chr(ord("A") + i))
        else:
            labels.append(base.upper())

    t2: dict[str, dict[str, dict]] = {}   # method -> label -> {class/mIoU: val}
    t1: dict[str, dict[str, dict]] = {}   # method -> label -> {metric: avg}
    skipped_all: set[str] = set()
    for r, lab in zip(a.runs, labels):
        p2 = Path(r) / "comparison_table2.csv"
        if not p2.exists():
            raise SystemExit(f"缺少 {p2} —— 请先在该 split 上运行 compare_all.py")
        d, sk = read_table2(p2)
        skipped_all |= set(sk)
        for m, vals in d.items():
            t2.setdefault(m, {})[lab] = vals
        p1 = Path(r) / "comparison_table1.csv"
        if p1.exists():
            for m, mm in read_table1(p1).items():
                t1.setdefault(m, {})[lab] = mm
        else:
            print(f"[warn] 缺少 {p1}, 4 要素指标列将不完整")

    if skipped_all:
        print(f"[警示] 以下行非 (eval) 口径, 已跳过不参与统计: {sorted(skipped_all)}")

    methods = sorted(t2.keys())
    label_cols = labels

    # ---- 汇总表: mIoU 各 split + mean±std; 4 要素均值 mean±std ----
    header = ["Method"] + [f"mIoU({l})" for l in label_cols] + ["mIoU mean±std"] \
        + [f"{k} mean±std" for k in ("IoU", "F1-score", "Recall", "Precision")]
    rows = []
    print("\n=== 跨划分汇总 (test mIoU; std 为样本标准差 n-1) ===")
    w = max(len(m) for m in methods) if methods else 8
    print(f"{'Method':<{w}} | " + " / ".join(f"mIoU({l})" for l in label_cols) + " | mean±std")
    for m in methods:
        per = [t2[m].get(l, {}).get("mIoU") for l in label_cols]
        vals = [v for v in per if v is not None]
        mean, std = ms(vals)
        cells = [f"{v:.4f}" if v is not None else "-" for v in per]
        print(f"{m:<{w}} | " + " / ".join(cells) + f" | {fmt_ms(mean, std)}")
        row = {"Method": m}
        for l, v in zip(label_cols, per):
            row[f"mIoU({l})"] = f"{v:.5f}" if v is not None else "-"
        row["mIoU mean±std"] = fmt_ms(mean, std)
        for k in ("IoU", "F1-score", "Recall", "Precision"):
            vv = [t1.get(m, {}).get(l, {}).get(k) for l in label_cols]
            vv = [v for v in vv if v is not None]
            mean_k, std_k = ms(vv)
            row[f"{k} mean±std"] = fmt_ms(mean_k, std_k)
        rows.append(row)

    out_csv = Path(a.runs[0]) / "splits_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        wtr = csv.DictWriter(fh, fieldnames=header)
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\n[OK] 汇总 -> {out_csv}")

    # ---- 逐类 IoU mean±std (可选) ----
    if a.perclass:
        cls_names: list[str] = []
        for m in t2:
            for l in t2[m]:
                cls_names = [k for k in t2[m][l] if k != "mIoU"]
                break
            if cls_names:
                break
        h2 = ["Method"] + [f"{c} mean±std" for c in cls_names] + ["mIoU mean±std"]
        rows2 = []
        for m in methods:
            row = {"Method": m}
            for c in cls_names:
                vv = [t2[m].get(l, {}).get(c) for l in label_cols]
                vv = [v for v in vv if v is not None]
                mean_c, std_c = ms(vv)
                row[f"{c} mean±std"] = fmt_ms(mean_c, std_c)
            vv = [t2[m].get(l, {}).get("mIoU") for l in label_cols]
            vv = [v for v in vv if v is not None]
            mean_m, std_m = ms(vv)
            row["mIoU mean±std"] = fmt_ms(mean_m, std_m)
            rows2.append(row)
        out2 = Path(a.runs[0]) / "splits_summary_perclass.csv"
        with open(out2, "w", newline="", encoding="utf-8-sig") as fh:
            wtr = csv.DictWriter(fh, fieldnames=h2)
            wtr.writeheader()
            wtr.writerows(rows2)
        print(f"[OK] 逐类汇总 -> {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
