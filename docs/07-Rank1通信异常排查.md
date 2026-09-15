# Rank1 通信异常排查方案

> 状态：🔵 待执行
> 上级文档：[03-算子Profiling](03-算子Profiling.md) · 关联阶段：阶段三 / 阶段四
> 数据来源：`sglang_profile/hostname-kczbs.foreman.pxe_971{0,1}_*_ascend_pt/`（2026-09-15 采集）
> 模型配置：Qwen3.8-27B · TP=2 · device 6/7 · NEXTN 投机解码 · 见 `start-qwen3.8-27b-prefix-cache.sh`

## 0. 结论摘要

**现象**：TP=2 的两张卡计算负载完全对称，但 rank1（device 7）的 `hcom_allReduce` 累计耗时是 rank0 的 **158 倍**，占 2.1s 采集窗口的 **70%**。

**已定位的根因**：**不是通信慢，是 rank0 迟到。** rank1 每次到达 allReduce 点后空等 rank0，等待时间被 HCCL 计入了 rank1 的通信耗时。通信链路本身健康（真实传输开销约 10us）。

**精确刻画**：问题 100% 集中在**一条 stream** 上 —— 该 stream 上的 132 次 allReduce **每次都迟到**（中位 6.9ms，最大 32.8ms，累计 1441ms）；而主通信流上的 774 次 allReduce 全部正常（中位 0ms）。

**下一步**：本方案第 4 节按优先级列出 4 个待验证假设与对应实验，第 5 节给出验收判据。

---

## 1. 证据链

以下每条均可由 `tools/check_rank_asymmetry.py` 复现。

### 1.1 两卡计算负载完全对称 → 排除"负载不均衡"

排除通信算子后的计算 kernel 对比：

| 时间窗 | rank0 (dev6) | rank1 (dev7) |
|---|---|---|
| 0–100 ms | 122 kernels / 36.97 ms | 87 kernels / 35.57 ms |
| 100–1600 ms | 2055 kernels / **83.06 ms** | 2032 kernels / **83.07 ms** |
| 1600–2100 ms | 10364 kernels / 203.72 ms | 8961 kernels / 176.55 ms |
| 2100–2200 ms | 89 kernels / 1.83 ms | 1550 kernels / 30.49 ms |
| **合计** | **12630 kernels** / 325.58 ms | **12630 kernels** / 325.68 ms |

**两卡计算 kernel 数量完全相同（12630），总耗时相差 0.1 ms（0.03%）。** 唯一的差别是尾部时间戳：rank1 的计算跨度多 32 ms（2149.9 vs 2117.6 ms）。**推测**为两卡 profiler 停止时刻不同步所致 —— 请对照采集时两卡的 `stop` 调用时刻确认；无论如何，总量一致说明这不是负载差异。

`op_statistic.csv` 亦确认两卡各算子次数与耗时逐项一致（`MatMulV2` 2249 次、`Transpose` 158 次、`recurrent_gated_delta_rule_20` 288 次 …）。**两卡做的事完全一样。**

### 1.2 allReduce 累计耗时差 158 倍

| | 次数 | 总耗时 | 均值 | 中位 | 最大 |
|---|---|---|---|---|---|
| rank0 (dev6) | 960 | **9.52 ms** | 9.9 us | 7.56 us | 481 us |
| rank1 (dev7) | 960 | **1503.35 ms** | 1566 us | 9.48 us | 32801 us |

> **数据陷阱**：`kernel_details.csv` 中每次 allReduce 有**两条完全重复的记录**（一条带 `Stream ID`，一条 `Stream=N/A`，StartTime 与 Duration 逐条相同），导致 `op_statistic.csv` 中计为 1920 次 / 19.03 ms（rank0）、3006.70 ms（rank1）。真实值需**取一半**，与 `communication_statistic_*.csv` 的 960 次互相印证。分析时务必按 `Stream ID != 'N/A'` 过滤。

注意中位数：rank1 的中位数（9.48 us）与 rank0（7.56 us）接近 —— **说明绝大多数 allReduce 是正常的，是少数超长调用拉高了均值。**

`hcom_allGather` 同样存在不对称，但量级小（0.50 ms vs 4.53 ms，9.1×），属同一根因的次要表现，后续一并观察即可，不作为独立排查目标。

### 1.3 逐次配对：`rank1 耗时 = rank0 迟到时间 + 10 us`

将两卡 allReduce 按序号配对，比较「rank0 发起时刻 − rank1 发起时刻」与「rank1 耗时」：

```
#  0: rank0晚   32.791 ms | rank1耗时   32.801 ms   (差 0.010 ms)
#  1: rank0晚   12.318 ms | rank1耗时   12.329 ms   (差 0.011 ms)
#  2: rank0晚    2.257 ms | rank1耗时    2.268 ms   (差 0.011 ms)
#  3: rank0晚   12.566 ms | rank1耗时   12.576 ms   (差 0.010 ms)
#  9: rank0晚   13.557 ms | rank1耗时   13.567 ms   (差 0.010 ms)
```

**960 次逐条吻合到 10 us 以内**，且这个固定偏移（≈10 us）就是 allReduce 的真实传输+同步开销。这证明：

- rank1 的"耗时"**全部**是等待 rank0 的时间，没有任何额外的传输开销；
- 通信链路（HCCS）健康，无需排查带宽、链路、拓扑。

`step_trace_time.csv` 的宏观印证 —— 两卡 `Computing` 相同，但"通信"与"空闲"此消彼长：

| | Computing | Communication | Free | Stage |
|---|---|---|---|---|
| rank0 (dev6) | 325.6 ms | 10.0 ms | **1790.0 ms** | 2125 ms |
| rank1 (dev7) | 325.5 ms | **1507.8 ms** | 325.2 ms | 2158 ms |

**rank0 的 1.79s 空闲正是在等 rank1 的 allReduce 走完** —— 两卡互为因果，缺一不可地看才能定位。

### 1.4 问题锁定在单条 stream

按 stream 拆分迟到量（rank1 stream 编号，对应 rank0 为编号 −3）：

| Stream (rank1 / rank0) | 次数 | 迟到合计 | 中位迟到 |
|---|---|---|---|
| **39 / 36** | **132** | **1441 ms** | **6.90 ms** |
| 122 / 119（主通信流） | 774 | 27 ms | −0.00 ms |
| 100 / 97 | 36 | 14 ms | −0.00 ms |
| 172 / 169 | 18 | 10 ms | 0.01 ms |

**Stream 39 上的 132 次 allReduce 100% 迟到**（中位 6.9 ms > 0），其余 828 次全部正常。这是一个非常干净的切分 —— 排查只需聚焦这一条流。

迟到量分布（rank0 发起时刻 − rank1 发起时刻，阈值 1 us）：960 次中 **450 次 rank0 晚到**、430 次 rank0 反而早到、80 次同时。**晚到合计 1495 ms，与两卡 allReduce 总耗时之差 1494 ms 吻合**，其中：

| 迟到区间 | 次数 |
|---|---|
| 0.1–1 ms | 7 |
| 1–5 ms | 67 |
| 5–15 ms | 33 |
| 15–40 ms | 42 |

> 注意「469 次 rank0 早到」这一半数据同样重要：它说明**两卡的快慢关系是翻转的**，不是简单的「rank0 一直慢」。结合 1.4 节可知，翻转点正好对应 Stream 39 与其余流的划分。

### 1.5 时间分布：集中在采集窗口前 1.6s

| 时间桶 | 迟到合计 |
|---|---|
| 0–1600 ms（8 个桶，每个 ~200ms） | 145 / 177 / 189 / 188 / 209 / 190 / 180 / 163 ms |
| 1600–1800 ms | 42 ms |
| 1800–2000 ms | 5 ms |
| 2000–2200 ms | 3 ms |

结合 `Model ID` 分层（`4294967295` = eager 执行，跨 0–2117 ms；`42` = ACL Graph replay，跨 1667–2097 ms），可判定：

- **0–1600 ms：eager 阶段**（每 100ms 仅 5.5ms 计算量，稀疏），迟到在这里稳定发生，每个 200ms 桶贡献 ~180ms；
- **1600 ms 之后：graph replay 阶段**，迟到骤降至可忽略。

---

## 2. 已排除项

| 假设 | 排除依据 |
|---|---|
| 通信带宽/链路/HCCS 故障 | `communication.json` 中 `Transit Time = 0`、真实传输开销恒为 ~10us；逐次配对无额外开销 |
| 两卡计算负载不均衡 | 计算 kernel 数量（12630）与逐算子耗时完全一致 |
| 两卡 stream 结构不同 | stream 编号一一对应（偏移 +3），各流 kernel 数与耗时对称 |
| profiler 重复计数造成假象 | 双份记录已逐条比对确认为完全重复；取半后与 `communication_statistic` 独立吻合 |
| allReduce 数据量过大 | 单次 10us 量级完成，属 latency-bound 小包通信 |

---

## 3. 待验证假设

按优先级排序。**H1 与 H2 覆盖了绝大多数可能，建议先做这两项。**

### H1（高）—— Stream 39/36 是 rank0 特化的辅助流，其上游有 CPU 侧或跨流依赖

Stream 36/39 只在部分 allReduce 上出现（132/960），且在 eager 阶段 100% 迟到。若该流承载的是**投机解码（NEXTN）的 draft/verify 通信**或**采样后的参数广播**，则 rank0 需先完成 CPU 侧的树构建/采样/detokenize 才能发起 —— 这与"rank0 迟到、rank1 空等"完全吻合。

**支持线索**：`op_statistic.csv` 中存在 `build_tree_efficient`（6 次）、`verify_tree_greedy_kernel`（6 次）、`ArgMaxV2`（25 次）、`cache_loc_assign`、`fill_bonus_tokens`、`assign_draft_cache_locs_contiguous` —— 均为 NEXTN 投机解码链路，且数量（6）远小于 allReduce 的 960。

### H2（高）—— `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` 引入的 plan stream 同步缺陷

`start.sh` 同时开启了 `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`、`--enable-prefill-delayer`、`STREAMS_PER_DEVICE=32`。overlap plan stream 的语义是让规划流与计算流并行，但其事件同步若在 eager 路径上退化，就会出现「rank0 的 plan 流等待计算流、rank1 的融合流等待 plan 流」的串行放大。

**支持线索**：迟到只发生在 eager 阶段，graph replay 阶段完全正常 —— 与"overlap 逻辑只在 eager 路径生效"一致。且 rank0 比 rank1 多出 3 个 stream 编号（偏移 +3），提示两卡的流创建顺序/数量可能不同。

### H3（中）—— rank0 的 CPU 侧下发延迟

`api_statistic.csv` 可用于排查 CANN 层 API（Tiling、launch）耗时。若 rank0 的 host 侧 Tiling 队列显著长于 rank1，则说明瓶颈在 host。

### H4（低）—— profiling 采集本身引入的观测扰动

两卡独立启动 profiler，`with_stack=true` + `_data_simplification=false` 使采集开销极大。rank0 作为 tp_rank 0 可能承担额外的 host 侧采集负担。**该假设可用第 4 节 Step 0 直接验证或排除。**

---

## 4. 排查步骤

### Step 0 — 先确认现象在无 profiling 时依然存在（**必做，优先级最高**）

目的：排除 H4，确认这是真实性能问题而非观测扰动。

1. 关闭 profiler，按现有参数跑基线，记录 decode 阶段端到端 TPOT 与吞吐；
2. 用 `--enable-prefill-delayer` 关闭态复跑一次；
3. 对比两次的 TPOT。若关闭 profiler 后两卡耗时差异消失（可用 `npu-smi info -t usages` 或端到端 TPOT 间接判断），则 H4 成立，本方案其余步骤无需进行。

**判据**：profiling 关闭后若 TPOT 无明显改善，说明 1.5s 等待是真实的，继续 Step 1。

### Step 1 — 确认 Stream 39/36 承载的业务语义（验证 H1）

用 MindStudio Insight 打开 `trace_view.json`（641MB，**不要用浏览器**），定位 rank1 的 stream 39：

1. 查看该流上 allReduce **之前**紧邻的 kernel 是什么（按 Start Time 排序取前一个）；
2. 查看该流上 allReduce **之后**的 kernel；
3. 沿该流向上游找，看它依赖哪个事件的完成。

同时交叉验证 `ascend_pytorch_profiler_1.db`（SQLite，可直接 SQL 查询），筛选该 stream 的完整时间线。

**判据**：若能确认 Stream 39 对应 NEXTN 投机解码的 draft 通信，则 H1 成立，优化方向转向「减少 draft 阶段的跨卡同步」或「让 draft 通信与主计算 overlap」。

### Step 2 — 参数消融实验（验证 H2）

单变量逐个关闭，每次重采一次 profiling（**必须采 profiling**，只看端到端无法定位到 stream）：

| 实验 | 变更 | 观察指标 |
|---|---|---|
| E1 | 移除 `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` | Stream 39 迟到量是否归零 |
| E2 | 移除 `--enable-prefill-delayer` | 同上 |
| E3 | `STREAMS_PER_DEVICE` 32 → 默认值 | 同上 |
| E4 | 移除 `--speculative-algorithm NEXTN`（改用普通解码） | 迟到是否消失 → 直接验证 H1 |
| E5 | `--tp-size 2` → 1（单卡） | 排除 TP 相关同步 |

**E4 是区分 H1 / H2 的关键实验**：若去掉投机解码后迟到消失，则问题在 NEXTN 链路（H1）；若仍在，则问题在 overlap/delayer 配置（H2）。

### Step 3 — 检查 rank0 的 host 侧耗时（验证 H3）

对比两卡的 `api_statistic.csv`：

```bash
diff <(sort rank0/ASCEND_PROFILER_OUTPUT/api_statistic.csv) \
     <(sort rank1/ASCEND_PROFILER_OUTPUT/api_statistic.csv)
```

关注 `Level=host` 的行中 rank0 独有的条目，以及 `communication` 级别的 `AivKernel` / `Tiling` 类调用次数差异。

### Step 4 — 采集优化（降低后续排查成本）

重采时务必加上（当前 1.6GB/次、2.1s 窗口，迭代成本过高）：

```python
profiler_level=ProfilerLevel.Level1
with_stack=False
experimental_config=torch_npu.profiler._ExperimentalConfig(
    data_simplification=True,
    aic_metrics=AicoreMetrics.PipeUtilization,
)
```

并将采集窗口**收敛到稳态 decode**，跳过初始化与 graph capture（当前 0–1600ms 的 eager 段混入了大量初始化开销）。

---

## 5. 验收判据

方案完成的标志（按门禁形式给出）：

- [ ] **C1** — Step 0 确认现象在无 profiling 时仍存在（或证伪并关闭本方案）
- [ ] **C2** — Stream 39/36 的业务语义已确认，有 trace 截图或 SQL 查询结果佐证
- [ ] **C3** — E4 实验完成，H1 与 H2 二者有一个被确证
- [ ] **C4** — rank1 的 allReduce 累计耗时降至 **< 50 ms**（当前 1503 ms，即降至 rank0 水平的一个数量级内）
- [ ] **C5** — `step_trace_time.csv` 中两卡 `Communication` 与 `Free` 量级相当（当前 1508/325 vs 10/1790）
- [ ] **C6** — 端到端 TPOT 有可测量的改善，改善幅度与消除的 1.5s 等待相称

> **C6 是最终判据**：若 C4/C5 达成但 C6 无改善，说明这 1.5s 等待本来就被更大范围的流水掩盖（两卡 Stage 总时长 2125 vs 2158 ms 相差仅 1.5%），此时应重新评估该问题是否值得投入。

---

## 6. 附录

### 6.1 一键复现分析

```bash
python3 tools/check_rank_asymmetry.py sglang_profile/
```

脚本输出：两卡计算对称性、allReduce 双重计数提示、逐次配对迟到量、按 stream 的迟到分布。

### 6.2 关键路径速查

```
sglang_profile/
├── hostname-..._9710_..._ascend_pt/   → device 6 = TP rank 0
└── hostname-..._9711_..._ascend_pt/   → device 7 = TP rank 1
```

| 分析目标 | 文件 |
|---|---|
| 宏观三分（计算/通信/空闲） | `ASCEND_PROFILER_OUTPUT/step_trace_time.csv` |
| 算子级热点 | `ASCEND_PROFILER_OUTPUT/op_statistic.csv` |
| kernel 级明细（含 stream） | `ASCEND_PROFILER_OUTPUT/kernel_details.csv` |
| 通信逐条拆解（Transit/Wait/Idle） | `ASCEND_PROFILER_OUTPUT/communication.json` |
| 通信算子汇总 | `PROF_*/mindstudio_profiler_output/communication_statistic_*.csv` |
| 时间线可视化 | `ASCEND_PROFILER_OUTPUT/trace_view.json`（MindStudio Insight 打开） |
| SQL 直查 | `ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{0,1}.db` |
| host 侧 API | `ASCEND_PROFILER_OUTPUT/api_statistic.csv` |

### 6.3 本次分析中踩到的坑

1. **allReduce 双重记录** —— 不按 `Stream ID != 'N/A'` 过滤会导致耗时虚高一倍，且 `op_statistic.csv` 也受影响（1920 vs 960）。
2. **`ratio` 的分母是 AI Core 时间，不是墙钟时间** —— `op_statistic.csv` 中 325ms 仅占 2.1s 窗口的 15%，单看占比会严重低估通信/空闲。
3. **必须成对看两卡** —— 只看 rank0 会得出"通信仅占 0.5%，非常健康"的错误结论；只看 rank1 会误判为"通信带宽故障"。
4. **采集窗口混入初始化** —— 0–1600ms 含初始化与 eager 预热，其平均值不能用作稳态指标。
5. **`Model ID = 4294967295`（0xFFFFFFFF）** 表示 eager 执行，有具体 ID 的（如 42）表示 ACL Graph。
