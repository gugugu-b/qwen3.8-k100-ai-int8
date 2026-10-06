#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
submit1005 交付成果复现验证对比脚本

用途：把「复现实测」与「交付包原始数据」逐轮对齐对比，输出偏差百分比。

用法：
    python3 compare_verify.py <复现实测result目录> <交付包原始result目录>

例：
    python3 compare_verify.py \
        02-测试与结果/results-verify-submit1005-tuning4/result \
        submit1005/result
"""
import json
import os
import sys


def load(scene_dir):
    path = os.path.join(scene_dir, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def pct(new, old):
    if old in (None, 0):
        return "n/a"
    return f"{(new - old) / old * 100:+.2f}%"


def fmt(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


def compare_scene(scene, new_dir, old_dir):
    new = load(os.path.join(new_dir, scene))
    old = load(os.path.join(old_dir, scene))
    if not new:
        print(f"\n=== {scene}: 复现实测缺失 summary.json，跳过 ===")
        return

    print(f"\n{'=' * 78}")
    print(f" 场景 {scene}")
    print(f"{'=' * 78}")
    print(f" 交付包实测轮数: {len(old['rounds']) if old else 'n/a'}   "
          f"本次复现实测轮数: {len(new['rounds'])}")

    # ---- 逐轮明细 ----
    print(f"\n {'轮次':<6}{'来源':<10}{'输出吞吐':>11}{'总吞吐':>12}{'Wall(s)':>10}"
          f"{'TTFT_P50':>10}{'TPOT_P50':>10}{'状态':>7}")
    print(" " + "-" * 74)
    for label, data in (("交付", old), ("复现", new)):
        if not data:
            continue
        for r in data["rounds"]:
            print(f" round{r['round']:<2}{label:<10}"
                  f"{r['output_throughput_tps']:>11.2f}"
                  f"{r['total_throughput_tps']:>12.2f}"
                  f"{r['wall_time_s']:>10.2f}"
                  f"{r['ttft_s']['p50']:>10.2f}"
                  f"{r['tpot_ms']['p50']:>10.2f}"
                  f"{r.get('status', '?'):>7}")

    # ---- 汇总对比（以均值为准，不用单轮最大值）----
    def agg(data, key, sub=None):
        vals = [r[key] if sub is None else r[key][sub] for r in data["rounds"]]
        return sum(vals) / len(vals)

    print(f"\n {'指标':<16}{'交付均值':>12}{'复现均值':>12}{'偏差':>12}")
    print(" " + "-" * 50)
    metrics = [
        ("输出吞吐(tok/s)", "output_throughput_tps", None, 2),
        ("总吞吐(tok/s)", "total_throughput_tps", None, 2),
        ("Wall Time(s)", "wall_time_s", None, 2),
        ("TTFT P50(s)", "ttft_s", "p50", 2),
        ("TPOT P50(ms)", "tpot_ms", "p50", 2),
    ]
    for name, key, sub, nd in metrics:
        nv = agg(new, key, sub)
        ov = agg(old, key, sub) if old else None
        print(f" {name:<16}{fmt(ov, nd):>12}{fmt(nv, nd):>12}{pct(nv, ov):>12}")

    # ---- 单调性检查：轮间是否存在预热未完成的上升趋势 ----
    seq = [r["output_throughput_tps"] for r in new["rounds"]]
    rising = all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
    spread = (max(seq) - min(seq)) / min(seq) * 100 if len(seq) > 1 else 0.0
    print(f"\n 复现轮间输出吞吐序列: {' → '.join(f'{v:.2f}' for v in seq)}")
    print(f" 轮间极差: {spread:.2f}%   单调上升(预热未完成迹象): {'是 ⚠' if rising else '否'}")

    # ---- 质量校验 ----
    print(f"\n 质量校验（复现实测）:")
    for r in new["rounds"]:
        print(f"   round{r['round']}: status={r.get('status')} "
              f"成功={r['successful']}/{r['requested']} "
              f"精确长度={r['exact_length']} "
              f"乱码字符={r['replacement_chars']}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    new_dir, old_dir = sys.argv[1], sys.argv[2]
    for scene in ("4k1k", "64k1k"):
        compare_scene(scene, new_dir, old_dir)
    print()


if __name__ == "__main__":
    main()
