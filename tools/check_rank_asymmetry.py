#!/usr/bin/env python3
"""TP 多卡 Profiling 负载对称性检查。

用法:
    python3 tools/check_rank_asymmetry.py sglang_profile/
    python3 tools/check_rank_asymmetry.py sglang_profile/ --window 200

定位目标: 找出 TP 各 rank 之间「计算对称但通信耗时悬殊」的异常。
输出五部分: 计算对称性 / allReduce 汇总 / 逐次配对迟到量 / 按 Stream 分布 / 按时间桶分布。

依赖: 仅标准库。
"""

import argparse
import collections
import csv
import os
import re
import statistics
import sys

csv.field_size_limit(10 ** 9)

COMM_OPS = ("hcom_allReduce_", "hcom_allGather_", "hcom_reduceScatter_", "hcom_broadcast_")


def find_ranks(root):
    """返回 [(rank_label, prof_dir), ...]，按目录名排序。"""
    dirs = [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]
    ranks = []
    for d in dirs:
        kd = os.path.join(d, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
        if not os.path.exists(kd):
            continue
        # 目录名形如 hostname-xxx_9710_20260915152040882_ascend_pt，取其中的设备号
        m = re.search(r"_(\d{3,4})_\d{8,}_", os.path.basename(d))
        ranks.append((f"rank-{m.group(1)}" if m else os.path.basename(d), d))
    return ranks


def load_kernels(prof_dir):
    path = os.path.join(prof_dir, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def dedup_comm(rows):
    """剔除通信算子的重复记录 (Stream ID == 'N/A')。

    kernel_details.csv 中每次通信有两条完全重复的记录，其中一条 Stream ID 为
    'N/A'。不去重会导致通信耗时虚高一倍，op_statistic.csv 同样受影响。
    """
    return [r for r in rows if not (r["Type"] in COMM_OPS and r["Stream ID"] == "N/A")]


def section(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def analyze_calc_symmetry(named_rows, window_ms):
    """排除通信后，按时间窗对比各 rank 的计算 kernel 数与耗时。"""
    section("1. 计算对称性 (已排除通信算子)")
    stats = {}
    for label, rows in named_rows:
        comp = [r for r in rows if r["Type"] not in COMM_OPS]
        if not comp:
            continue
        t0 = min(float(r["Start Time(us)"]) for r in comp)
        span = max(float(r["Start Time(us)"]) + float(r["Duration(us)"]) for r in comp) - t0
        buckets = collections.defaultdict(lambda: [0, 0.0])
        for r in comp:
            b = int((float(r["Start Time(us)"]) - t0) / (window_ms * 1000))
            buckets[b][0] += 1
            buckets[b][1] += float(r["Duration(us)"])
        stats[label] = {
            "total_n": len(comp),
            "total_ms": sum(float(r["Duration(us)"]) for r in comp) / 1000,
            "span_ms": span / 1000,
            "buckets": buckets,
        }
        print(f"  {label}: {len(comp)} kernels, {stats[label]['total_ms']:.1f} ms, 跨度 {stats[label]['span_ms']:.1f} ms")

    labels = list(stats)
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        diff = stats[a]["total_ms"] - stats[b]["total_ms"]
        print(f"\n  计算耗时差 ({a} - {b}): {diff:+.2f} ms "
              f"({abs(diff) / max(stats[a]['total_ms'], 1e-9) * 100:.1f}%)")
        print("  → 差异远小于通信差异则说明负载对称，瓶颈在通信同步而非算力分配")
    return stats


def analyze_comm(named_rows):
    """通信算子汇总 + 重复记录提示。"""
    section("2. 通信算子汇总")
    table = {}
    for label, rows in named_rows:
        raw = [r for r in rows if r["Type"] in COMM_OPS]
        ded = [r for r in raw if r["Stream ID"] != "N/A"]
        agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
        for r in ded:
            a = agg[r["Type"]]
            a[0] += 1
            a[1] += float(r["Duration(us)"])
            a[2] = max(a[2], float(r["Duration(us)"]))
        table[label] = (len(raw), len(ded), agg)
        dup = len(raw) - len(ded)
        print(f"  --- {label} ---")
        if dup:
            print(f"  ⚠ 检测到 {dup} 条重复记录 (Stream ID=N/A)，已剔除；"
                  f"原始 {len(raw)} 条 → 真实 {len(ded)} 条")
        for op, (c, t, mx) in sorted(agg.items(), key=lambda x: -x[1][1]):
            print(f"    {op:<22} {c:>5} 次, 合计 {t/1000:>9.2f} ms, "
                  f"均值 {t/c:>9.2f} us, 最大 {mx:>9.1f} us")

    labels = list(table)
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        print(f"\n  对比 ({b} vs {a}):")
        for op in set(table[a][2]) | set(table[b][2]):
            ta = table[a][2].get(op, [0, 0, 0])[1]
            tb = table[b][2].get(op, [0, 0, 0])[1]
            ratio = (tb / ta) if ta > 0 else float("inf")
            flag = "  ← 异常" if ratio > 5 or ratio < 0.2 else ""
            print(f"    {op:<22} {ta/1000:>9.2f} ms vs {tb/1000:>9.2f} ms  "
                  f"({ratio:.1f}×){flag}")


def analyze_lag(named_rows):
    """逐次配对，量化「慢的一方迟到多久」与「快的一方等了多久」。"""
    section("3. 逐次配对: 寻找「谁在等谁」")
    comm = {}
    for label, rows in named_rows:
        r = sorted(
            (float(x["Start Time(us)"]), float(x["Duration(us)"]), x["Stream ID"])
            for x in rows
            if x["Type"] == "hcom_allReduce_" and x["Stream ID"] != "N/A"
        )
        comm[label] = r
    labels = list(comm)
    if len(labels) < 2:
        print("  需要至少 2 个 rank 才能配对")
        return None

    a, b = labels[0], labels[1]  # a 为参照(通常 rank0)，b 为被检查对象
    n = min(len(comm[a]), len(comm[b]))
    if n == 0:
        print("  未找到 allReduce 记录")
        return None

    # 先按序号配对，再按起点对齐重排，避免两卡次数错位
    first_times = [comm[a][i][0] for i in range(n)]
    second_times = [comm[b][i][0] for i in range(n)]
    lag = [(first_times[i] - second_times[i]) / 1000 for i in range(n)]
    dur = [comm[b][i][1] / 1000 for i in range(n)]

    print(f"  配对数 {n}  (基准: {a}  vs  被检查: {b})")
    print(f"  {a} 相对 {b} 的迟到: 中位 {statistics.median(lag):.3f} ms, "
          f"均值 {statistics.mean(lag):.3f} ms, 最大 {max(lag):.3f} ms")
    print(f"  {b} 的 allReduce 耗时: 中位 {statistics.median(dur):.3f} ms, "
          f"均值 {statistics.mean(dur):.3f} ms")
    print("\n  前 12 次配对 (若两者差值恒为常数，则证明是「等待」而非「传输慢」):")
    for i in range(min(12, n)):
        print(f"    #{i:>3}: {a}晚 {lag[i]:>9.3f} ms | {b}耗时 {dur[i]:>9.3f} ms "
              f"| 差 {dur[i] - lag[i]:>7.3f} ms")

    late = [l for l in lag if l > 0.001]
    early = [l for l in lag if l < -0.001]
    print(f"\n  发起时刻一致性: {len(late)} 次 {a} 晚到, {len(early)} 次早到, "
          f"{n - len(late) - len(early)} 次同时")
    print(f"  晚到合计 {sum(late):.0f} ms  ← 这即是 {b} 的等待量")
    gaps = [dur[i] - lag[i] for i in range(n)]
    print(f"  固定偏移(耗时 − 迟到,取中位): {statistics.median(gaps):.3f} ms "
          f"→ 即真实的 allReduce 传输+同步开销")
    print("  → 偏移量为常数则证明是「等待」而非「传输慢」")
    return {"lag": lag, "dur": dur, "labels": (a, b)}


def analyze_lag_by_stream(named_rows, laginfo):
    """按 stream 拆分迟到量，定位到具体是哪条流的问题。"""
    if not laginfo:
        return
    section("4. 迟到量按 Stream 拆分 (定位问题流)")
    a, b = laginfo["labels"]
    streams = {}
    for label, rows in named_rows:
        streams[label] = sorted(
            (float(x["Start Time(us)"]), float(x["Duration(us)"]), x["Stream ID"])
            for x in rows
            if x["Type"] == "hcom_allReduce_" and x["Stream ID"] != "N/A"
        )
    target = streams[b]
    agg = collections.defaultdict(lambda: [0, 0.0, []])
    for i, l in enumerate(laginfo["lag"]):
        sid = target[i][2]
        agg[sid][0] += 1
        agg[sid][1] += l
        agg[sid][2].append(l)

    print(f"  {'Stream':>8} {'次数':>6} {'迟到合计':>12} {'中位迟到':>12} {'最大迟到':>12}")
    for sid, (c, total, vals) in sorted(agg.items(), key=lambda x: -x[1][1]):
        med = statistics.median(vals)
        flag = "  ← 问题流" if med > 0.5 else ""
        print(f"  {sid:>8} {c:>6} {total:>10.1f} ms {med:>10.3f} ms "
              f"{max(vals):>10.3f} ms{flag}")

    bad = [(s, v) for s, v in agg.items() if statistics.median(v[2]) > 0.5]
    if bad:
        total = sum(v[1] for _, v in bad)
        print(f"\n  → 问题集中在 {len(bad)} 条流上，迟到合计 {total:.0f} ms")
    else:
        print("\n  → 未发现显著迟到的流")


def analyze_lag_by_time(named_rows, laginfo, window_ms):
    """按时间桶拆分迟到量，判断问题发生在哪个阶段。"""
    if not laginfo:
        return
    section(f"5. 迟到量按时间桶拆分 (每 {window_ms}ms)")
    _, b = laginfo["labels"]
    streams = {}
    for label, rows in named_rows:
        streams[label] = sorted(
            (float(x["Start Time(us)"]), float(x["Duration(us)"]), x["Stream ID"])
            for x in rows
            if x["Type"] == "hcom_allReduce_" and x["Stream ID"] != "N/A"
        )
    target = streams[b]
    t0 = target[0][0]
    buckets = collections.defaultdict(float)
    for i, l in enumerate(laginfo["lag"]):
        buckets[int((target[i][0] - t0) / (window_ms * 1000))] += l

    for k in sorted(buckets):
        bar = "█" * min(int(buckets[k] / max(max(buckets.values()), 1e-9) * 40), 40)
        print(f"  [{k*window_ms:>6}-{k*window_ms+window_ms:>6} ms] "
              f"{buckets[k]:>8.1f} ms {bar}")
    print("\n  → 迟到集中在某段窗口说明该阶段有额外同步；全窗口均匀则说明是常态开销")


def main():
    ap = argparse.ArgumentParser(description="TP 多卡 Profiling 负载对称性检查")
    ap.add_argument("profile_dir", help="包含各 rank 子目录的 profiling 根目录")
    ap.add_argument("--window", type=int, default=200, help="时间桶宽度(ms)，默认 200")
    args = ap.parse_args()

    if not os.path.isdir(args.profile_dir):
        sys.exit(f"目录不存在: {args.profile_dir}")

    ranks = find_ranks(args.profile_dir)
    if len(ranks) < 1:
        sys.exit(f"在 {args.profile_dir} 下未找到含 kernel_details.csv 的 rank 目录")
    print(f"发现 {len(ranks)} 个 rank:")
    for label, d in ranks:
        print(f"  {label}  →  {d}")

    named_rows = []
    for label, d in ranks:
        rows = dedup_comm(load_kernels(d))
        named_rows.append((label, rows))
        print(f"  {label}: {len(rows)} 条 kernel 记录 (已剔除通信重复项)")

    analyze_calc_symmetry(named_rows, args.window)
    analyze_comm(named_rows)
    laginfo = analyze_lag(named_rows)
    analyze_lag_by_stream(named_rows, laginfo)
    analyze_lag_by_time(named_rows, laginfo, args.window)

    section("排查提示")
    print("""  若出现「计算对称 + 通信悬殊 + 逐次配对差值稳定」的组合:
    1. 先排除观测扰动 —— 关闭 profiling 复测端到端 TPOT
    2. 用 MindStudio Insight 打开问题流的 trace_view.json，看该流 allReduce 前
       紧邻的 kernel 及其依赖的事件
    3. 单变量消融相关开关 (overlap / delayer / 投机解码 / TP size)
    详见 docs/07-Rank1通信异常排查.md""")


if __name__ == "__main__":
    main()
