# 算子问题清单（确定项）

> 生成日期：2026-09-16
> 数据源：`sglang_profile/hostname-kczbs.foreman.pxe_971{0,1}_*_ascend_pt/`
> 采集条件：Ascend 910B · CANN 9.0.0 · torch_npu 2.10.0 · Qwen3.8-27B · TP=2 (device 6/7)
> · NEXTN 投机解码 · 采集于 2026-09-15 15:20 · 窗口 2118 ms
> 运行代码：`/sgl-workspace/sglang`（**注意：与 `/home/zhujianwei/sglang` 不是同一份，行号以运行副本为准**）
>
> 修订记录（2026-09-16）：
> · 问题 1 —— 初版把 20.6 ms 整个算作收益，已改为按**设备占用率**拆分（设备忙 16.7 / 空闲 3.9 ms）；
> · 问题 4 —— 初版判为「kernel 单核执行、需多核切分」，**判断有误已更正**，`Block Num=1` 系 bs=1 所致，详见该节「根因」；
> · 问题 4 —— 收益口径已区分「单 kernel 提速比」与「端到端占比」，避免用 9× 描述 1.5% 的收益；
> · 问题 1 —— 与新增的 bs=1 端到端压测交叉核对，得到 **真实步长 55.1 ms**（压测推导）与 **profiled 76~78 ms**，
>   两者之差 ~21 ms 与「真实 step − 设备工作 = ~18 ms」共同指向同一道 20.6 ms 屏障；
>   **收益区间由 5/10~20/≤27% 上修为 7/18/~33%**。
> · 问题 1 —— **已按方案 A 落地代码**（`ops/mamba-conv-state-track-copy/`，单测 8/8 通过）。
>   落地 kernel 与本文草图的差异、以及新增的「src/dst 槽位不相交」前提，见「### 修改方案」下的实施记录。
>   **端到端收益尚未验证** —— 需重启服务后按「验证方案」测量。
>
> **适用范围**：本清单全部结论基于 **bs=1** 的 profile 与**单并发**压测。高并发不在当前范围内。

## 0. 收录口径

**本清单只收录能用原始数据逐条复现的问题。** 每条给出：现象 → 证据 → 根因 → 优化动作 → 验收方式。

不收录的内容见「附 B：本清单的数据边界」，原因是数据不足以支撑结论，不是"没问题"。

复核时对同一份 profile 按**时间阶段**做了切分，因为整窗口平均会严重失真：

| 阶段 | 墙钟 | 设备忙 | 性质 |
|---|---|---|---|
| A 0–100 ms | 100 ms | 37.0 ms | 初始化 / warmup |
| B 100–1600 ms | 1500 ms | 84.8 ms（占用 5.6%） | 低并发下的稀疏段 |
| C 1600–2200 ms | 500 ms | 223.8 ms（占用 45%） | **稳态 decode（6 个 step）** |

**下述占比若无特别说明，分母均为阶段 C 的 223.8 ms。**

---

## 问题 1（P0）每个 decode step 插入一道 20.6 ms 的同步屏障

### 现象

Host 侧 `aclnnNonzeroV2` 每个 decode step 触发一次，每次阻塞约 20 ms；而它对应的设备算子只花 3 μs。

> **复核修正（2026-09-16）**：本节初稿把这条描述为"每步浪费 20.7 ms 设备时间"，**该表述不准确**，已按下面的实测数据改写。正确的描述是「每步插入一道同步屏障」，收益机制与量级都不同，详见「阻塞的构成」一节。

### 证据

**(1) 调用次数与时刻严格对应 6 个 decode step**

```
aclnnNonzeroV2  6 次：
  t+1680.4 / 1772.7 / 1850.7 / 1927.8 / 2003.8 / 2080.5 ms
  每次耗时 19589 ~ 21339 us
```

**(2) 阻塞发生在 host 等待，不在设备计算**

```
PYTORCH_API：  Enqueue@aclnnNonzeroV2   0.009 ms
              Dequeue@aclnnNonzeroV2  20.745 ms
设备侧 kernel：NonZero, MIX_AIV, 6 次，合计 18.96 us（均值 3.16 us，最大 3.38 us）
```

设备算子只花 3.16 μs，host 花了 20.7 ms —— 这是**动态 shape + device→host 同步**的代价，不是计算代价。

**(3) 调用链（来自 `ascend_pytorch_profiler_0.db` 的 PYTORCH_API 嵌套帧）**

```
speculative/spec_utils.py(842)  commit_mamba_states_after_verify        27.28 ms
 └─ hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py
      update_mamba_state_after_mtp_verify                                24.28 ms
    └─ aten::index                                                       21.17 ms
       └─ Dequeue@aclnnNonzeroV2                                         20.74 ms
```

**(4) 根因代码 —— 布尔掩码索引**

`/sgl-workspace/sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py`

```python
292            track_mask = mamba_steps_to_track >= 0        # NPU 上的 bool tensor
...
295            track_indices = mamba_track_indices[track_mask]   # ← 布尔掩码索引
296            if track_indices.numel() > 0:                     # ← 又一次 host 同步
297                conv_states[:, track_indices] = conv_states[:,
298                    dst_indices_tensor[track_mask]]
```

`tensor[bool_mask]` 在 NPU 上会 lower 成 NonZero（因为输出长度动态），随后 `numel()` 再次强制同步。

**讽刺的是**：该函数自己的 docstring 写着这次改写就是为了消除这类算子 ——

```
Update mamba states after MTP verify using fully fused Triton kernel.
This replaces the original advanced indexing operations with a single fused
gather-scatter kernel that also handles masking internally, avoiding:
- index_elementwise_kernel from tensor[bool_mask]
- index_select kernel launches
- nonzero kernel launches
```

而同一函数上方（第 275–285 行）已经用了正确的做法 `move_intermediate_cache(..., mask 传进 kernel ...)`。**第 295 行是这段改写的遗漏点。**

### 阻塞的构成（关键，决定了收益机制）

**这 20.6 ms 不是"设备空转 20.6 ms"。** 逐窗口统计设备实际忙碌时间：

| 同步窗口 | 窗口长 | 设备忙 | 设备空闲 | 占用率 |
|---|---|---|---|---|
| t+1680.51 – 1701.10 ms | 20.6 ms | 16.7 ms | 3.9 ms | 81.2% |
| t+1772.73 – 1792.19 ms | 19.5 ms | 15.6 ms | 3.9 ms | 80.2% |
| t+1850.76 – 1871.98 ms | 21.2 ms | 17.4 ms | 3.8 ms | 81.9% |
| t+1927.87 – 1948.69 ms | 20.8 ms | 17.0 ms | 3.8 ms | 81.5% |
| t+2003.90 – 2024.81 ms | 20.9 ms | 17.0 ms | 3.9 ms | 81.5% |
| t+2080.60 – 2101.69 ms | 21.1 ms | 17.2 ms | 3.9 ms | 81.7% |
| **合计** | **124.1 ms** | **100.9 ms** | **23.1 ms** | — |

**窗口内 81% 的时间设备在执行已排队的真实 kernel** —— host 是在等设备追平，不是在等一个空转的设备。设备侧 kernel 耗时来自硬件时间戳，`with_stack` 之类的采集开销不影响它；且阻塞中的线程不产生 profiling 事件，stack 展开开销也放大不了一个纯等待。

**结论：这 20.6 ms 不是采集导致的假象。** 其中真正的净损失（设备空闲）每步只有 **3.9 ms**。

**但这条仍然要修，机制是"同步屏障"而不是"浪费时间"：**

阶段 C 的设备占用率只有 **36 ~ 45%**（223.8 ms / 500 ms 量级），说明流水线瓶颈在 **host 侧**，设备大半时间在挨饿。此时 `NonZero` 的动态 shape 逼 host 必须停下等设备追平，host 就损失了这 20.6 ms 本可用来准备下一步的时间。

所以收益的正确算法是：**解除屏障后 host 提前 enqueue，收益取决于 host 能在多大程度上重新跑在设备前面** —— 不是"省 20.7 ms"，也不是"省 3.9 ms"。

### 修改方案

**关键前提：正确的写法就在同一个函数里，往上 20 行。**

第二个 `move_intermediate_cache(...)` 调用（第 286–292 行）传的是**未做掩码**的 `mamba_track_indices` / `mamba_steps_to_track`，无效项由 kernel 内部跳过：

```python
# sgl_kernel_npu/mamba/mamba_state_update_triton.py
#   move_cache_dynamic_last_kernel_h_block
last_step_val = tl.load(last_steps_ptr + valid_id)
if last_step_val < 0:
    return                                      # ← 无效项直接退出，无需 host 侧筛
```

而紧随其后的 conv state 搬运（第 292–298 行）没走这条路，改在 host 侧用布尔掩码挑选下标，于是把 NonZero 和 `numel()` 同步又引了回来。

---

#### 方案 A（推荐）：给 conv state 搬运补一个同款 kernel

新增 kernel，与现有两个完全同构（grid = 请求数，`step < 0` 提前返回）：

```python
@triton.jit
def _conv_state_copy_for_track_kernel(
    conv_states_ptr, track_indices_ptr, dst_indices_ptr, step_indices_ptr,
    num_layers, num_dims: tl.constexpr,
    layer_stride: tl.constexpr, req_stride: tl.constexpr, dim_stride: tl.constexpr,
):
    pid = tl.program_id(0)
    step_val = tl.load(step_indices_ptr + pid).to(tl.int64)
    if step_val < 0:
        return                                   # ← 与上游 kernel 同一约定
    src_idx = tl.load(dst_indices_ptr + pid).to(tl.int64)
    dst_idx = tl.load(track_indices_ptr + pid).to(tl.int64)
    off = tl.arange(0, num_dims) * dim_stride
    for layer in range(num_layers):
        base = layer * layer_stride
        data = tl.load(conv_states_ptr + src_idx * req_stride + base + off)
        tl.store(conv_states_ptr + dst_idx * req_stride + base + off, data)
```

host 侧把 292–298 行整体替换为一次调用：

```python
conv_state_copy_for_track(
    conv_states, mamba_track_indices, dst_indices_tensor, mamba_steps_to_track
)
```

**要点：不再构造 `track_mask`、不做布尔索引、不调 `numel()`。** 整条链变为全异步。

---

#### ✅ 方案 A 实施记录（2026-09-16 已落地）

产物归档在 [`ops/mamba-conv-state-track-copy/`](../ops/mamba-conv-state-track-copy/README.md)
（含 `original/` 原件、`modified/` 改后全文、两个 patch、单元测试）。

**落地的 kernel 与上面的草图有两处不同** —— 草图是按「每个条目只搬 `num_dims` 个元素」写的，
与实机不符，实现时以仓库里**已在产**的 `sgl_kernel_npu/mamba/speculative_state_scatter.py`
为模板（它解决的是同一类问题，且 mask 同样在 device 上做）：

| 草图 | 实机 | 落地处理 |
|---|---|---|
| `off = tl.arange(0, num_dims)`，每条目搬 `num_dims` | `conv_states` 是 4 维 `[L, pool, conv_window, num_dims]`，每条目要搬 **`conv_window × num_dims`** | 用 `tail_0/1/2` 三维偏移，rank 1~3 的 tail 均可 |
| `grid = (num_requests,)` | bs=1 时 grid=1，**只用 1 个核**（这正是问题 4） | logical/physical grid：逻辑网格 = `requests × layers × tail_grid`，物理网格 = `min(48, logical)` |

改动文件：`ascend_hybrid_linear_attn_backend.py`（import + 292–299 行替换）、
`mamba_state_update_triton.py`（新增 `_conv_state_copy_for_track_kernel` + host 封装）。

**验证**：单元测试与原 eager 语义逐元素比对 **8/8 通过**，覆盖 `tail_grid > 1`（大 tail 走跨步循环）、
非 2 次幂 dims、全负 step、空请求集。命令见 README。

**正确性前提（新增约束，务必知悉）**：新 kernel 是**原地**拷贝，故 `src_indices` 与 `dst_indices`
必须寻址**不相交**的槽位，否则存在「读到已被覆盖数据」的跨 program 竞争（原 PyTorch 写法因 RHS
先物化，没有这个问题）。此前提**现有代码本来就依赖** —— 该函数里第 271 行与第 285 行**两次**
`move_intermediate_cache` 都往同一个 `ssm_states` 写，用的正是这两个索引集；若相交那两处早已互相覆盖。
索引来源亦印证：`mamba_track_indices` ← `req_index_to_mamba_ping_pong_track_buffer_mapping`
（ping-pong track 池），`dst_indices_tensor` ← `mamba_cache_indices`（工作缓存槽），**两个不同的池**
（`srt/managers/schedule_batch.py:1997`）。此为静态分析结论，未加运行期断言。

> ⚠️ **端到端收益尚未验证。** 代码改动需**重启推理服务**才生效；验证时必须**关 profiler**
> （profiling 把步长从 55.1 ms 抬到 ~76 ms，×1.38，会淹没收益），且该负载**输出非位级可复现**，
> 只能比统计量。预期区间见下文「预期收益」。

**附带发现**：`ascend_kda_backend.py:603-608` **已经在产使用方案 B**（`torch.where` 版），
说明方案 B 的前置条件在实机上成立。若不写 kernel，把那段照搬到 hybrid 后端即可当天验证收益。

---

#### 方案 B（不写 kernel，先用来快速验证收益）

把「选择性搬运」换成「全量搬运 + 无效项自拷贝」：

```python
valid = mamba_steps_to_track >= 0
safe_src = torch.where(valid, dst_indices_tensor, mamba_track_indices)
# 无效项 src == dst → 自拷贝，语义不变
conv_states[:, mamba_track_indices] = conv_states[:, safe_src]
```

- 无效项的 src == dst，是自拷贝，结果与原来一致
- `torch.where` 不产生同步，**整条链保持异步**
- 代价：多搬 `N - valid` 个条目（bs=1 时可忽略）
- ⚠️ **前置条件**：无效项的 `mamba_track_indices` 必须落在合法槽位内，否则会越界。需先确认填充值是否安全（`forward_batch_info.py:1546` 有 `_pad_tensor_to_size` 的 padding 逻辑）

方案 B 只改 host 侧、当天可验证，适合先用来测收益；确认有效后再按方案 A 落 kernel。

### 常见误判（复核时踩到，记录备查）

`aclnnNonzeroV2` 与 `aclrtSynchronizeStream` 是**同一次等待的两个视图**，不是两笔开销：

```
aclnnNonzeroV2        t+1680.4  20.737 ms   |  aclrtSynchronizeStream  t+1680.51  20.590 ms
                      t+1772.7  19.589 ms   |                          t+1772.73  19.459 ms
                      t+1850.7  21.339 ms   |                          t+1850.76  21.217 ms
                      ...          时刻一一对应，相差 0.1 ms
合计                  124.83 ms             |                          124.33 ms
```

两者总量几乎相同（124.83 vs 124.33 ms），时刻一一对应 —— 是同一次 stream 同步，**不可相加**。

### 预期收益

**先说结论：不能拿 20.6 ms，更不能拿 `api_statistic.csv` 里的 124.83 ms 当收益**（理由见「阻塞的构成」）。

单步时间构成：

```
真实步长（bs=1，压测推导）                55.1 ms   ← 见下方佐证
  ├─ 设备工作量                           37.3 ms   （223.8 ms ÷ 6 step）← 硬下限
  └─ host 侧停顿                          ~18 ms    ← 屏障所在
profiled 步长（相邻两次同步的间隔）        ~76 ~ 78 ms  = 真实 step + ~21 ms profiling 开销
同步屏障                                  20.6 ms   ← 其中 设备忙 16.7 / 设备空闲 3.9
阶段 C 设备占用率                         36 ~ 45%  → 瓶颈在 host 侧
```

| 口径 | 假设 | 单步收益 | 相对提升 |
|---|---|---|---|
| **下限** | 只回收屏障内的设备空闲 | 3.9 ms | **约 7%** |
| **中性** | 回收部分 host 停顿 | ~10 ms | **约 18%** |
| **上限** | 步长收敛到设备下限 37.3 ms | ~18 ms | **约 33%**（55.1 → ~37 ms，约 1.49×） |

上限的硬约束：步长不可能低于每步设备工作量 **37.3 ms**（前提是 host 侧其他工作不再成为新的瓶颈）。

**为什么只能给区间**：这 20.6 ms 是 host 主线程的阻塞时间，但它发生时设备正在执行已排队的工作（占用 81%）。所以它既不是纯粹的浪费（下限 3.9 ms），也不是能全额回收（上限 20.6 ms）。实际能回收多少，取决于 host 解除阻塞后能否把后续工作提前铺满 —— 这正是需要实验测的量。

**量级参照**：问题 2 的量化预计能把 MatMulV2 稳态 161.7 ms 压到 ~85 ms（约 76 ms，占稳态 34%），**量级比本条大**。但量化属部署环节、落地成本高；本条改动集中在单个函数、当天可验证，适合作为第一个动手项。

> ### ✅ 端到端压测的独立佐证（2026-09-16 补充，收益上修）
>
> 并发 1 的端到端压测（[`results/baseline/baseline-20260916-qwen3.8-27b.md`](../results/baseline/baseline-20260916-qwen3.8-27b.md)，
> 工作负载与本 profile 同为 bs=1）给出 **Median ITL 13.79 ms / Mean TPOT 14.28 ms**。
>
> **⚠️ 这两个数不是步长。** 它们的口径是 `(E2E − TTFT) / (output_len − 1)`（已用输出自身验证），
> 是**摊到每个 token** 的均值；本任务 accept length = 3.86，**必须乘回去**：
>
> ```
> 总 step 数 = 20,000 token ÷ 3.86 = 5,181
> 每 step   = 20 × (15253.65 − 986.91) ms ÷ 5,181 = 55.07 ms
> ```
>
> （**直接拿 13.79 ms 当步长会低估 3.86 倍** —— 这是本次复核中实际犯过的错，记录备查。）
>
> 于是有两条独立算术指向同一结论：
>
> | 口径 | 计算 | 结果 |
> |---|---|---|
> | profiling 开销 + 额外停顿 | 76（profiled）− 55.07（真实） | ~21 ms |
> | 真实 step 里的**非设备**时间 | 55.07 − 37.3（设备工作） | ~17.8 ms |
> | 实测同步屏障 | `aclnnNonzeroV2` | 20.6 ms |
>
> **三者吻合 → 真实 decode step 中约 18~21 ms 是 host 侧停顿，就是这道屏障。**
> 该结论现在有两个独立数据源支撑，比只有 profiling 时可信得多。
>
> **收益按真实步长 55.07 ms 重算（比原表上修）**：
>
> | 口径 | 假设 | 单步收益 | 相对提升 |
> |---|---|---|---|
> | 下限 | 只回收屏障内的设备空闲 3.9 ms | 3.9 ms | **约 7%** |
> | 中性 | 回收部分 host 停顿 ~10 ms | 10 ms | **约 18%** |
> | 上限 | 步长收敛到设备下限 37.3 ms | ~18 ms | **约 33%**（55.07 → ~37 ms，约 1.49×） |
>
> **原表的 5% / 10~20% / ≤27% 作废，改用上表。**
> 定位未变：`ascend_hybrid_linear_attn_backend.py:295`。

### 验证方案（单点对照实验）

1. **只改这一个函数**，其余参数与 `start-qwen3.8-27b-prefix-cache.sh` 完全一致；
2. **压测脚本已归档，直接用**（`--max-concurrency 1`，单次约 305 s，迭代成本低）：
   见 [`results/baseline/baseline-20260916-qwen3.8-27b-c1.raw.txt`](../results/baseline/baseline-20260916-qwen3.8-27b-c1.raw.txt) 顶部。
   **改前 / 改后各跑 3 次**，比 **Median TPOT**（当前 13.97 ms）与 **Median TTFT**（当前 588 ms）。
3. 若要确认机制（非必需），再采一次 profiling 观察：
   - `api_statistic.csv` 中 `aclnnNonzeroV2` 是否消失；
   - `aclrtSynchronizeStream` 的 6 次 20 ms 长尾是否消失；
   - 阶段 C 设备占用率是否上升（36~45% → ?）。

> ⚠️ **必须关 profiler 做端到端对比** —— profiling 本身把步长从 55.1 ms 抬到 ~76 ms（+38%），会淹没收益。
> 这是本次复核用压测数据反推出来的量化结论。
>
> ⚠️ 改前务必先记录 `Accept length`（当前 3.86）与 `Total generated tokens (retokenized)`。
> 注意该负载**输出不是位级可复现**（两次运行重编码计数差 2.2%），只能比统计量。

> 若 `aclnnNonzeroV2` 消失但 TPOT 无变化 → 说明该屏障本来就被更大范围的流水掩盖。此时按 `docs/07` 的 C6 判据，**停止在本项继续投入**，直接转向问题 2。

---

## 问题 2（P0）MatMulV2 已打满 HBM 带宽 —— 只能减字节，调 kernel 无用

### 现象

稳态下 MatMulV2 占设备时间 72.3%，且全部是权重流式读取，计算单元几乎不干活。

### 证据

**(1) 稳态占比**

```
阶段 C（1600–2200 ms，6 个 step）：MatMulV2 1944 次 / 161.67 ms / 223.8 ms = 72.26%
  M=4  1866 次  143.76 ms   ← TARGET_VERIFY（draft_token_num=4）
  M=1    78 次   17.91 ms
```

**(2) PMU —— AIC 99% 的时间在等搬运**

单次 `(4,5120) @ (124160,5120)`：

```
dur=978.2us  blk=24  cube_utilization=94.80%
aicore_time     927.30 us
aic_mte2_time   917.87 us   ratio 0.99    ← GM→L1 搬运，占了 AIC 近乎全部时间
aic_mac_time    108.23 us   ratio 0.117   ← 真正做乘加只占 11.7%
aic_mte1_time   270.05 us   ratio 0.291
aic_scalar_time 301.47 us   ratio 0.325
```

**(3) 按 shape 反算实际搬运带宽**

| shape (M,N,K) | 单次耗时 | 数据量 | 实测带宽 | 算术强度 |
|---|---|---|---|---|
| 4, 124160, 5120 | 965.7 us | 1272.4 MB | **1318 GB/s** | 4.00 FLOP/B |
| 1, 124160, 5120 | 956.7 us | 1271.7 MB | **1329 GB/s** | 1.00 FLOP/B |
| 4, 17408, 5120 | 140.9 us | 178.4 MB | 1266 GB/s | 4.00 FLOP/B |
| 1, 17408, 5120 | 137.0 us | 178.3 MB | 1302 GB/s | 1.00 FLOP/B |
| 4, 5120, 10240 | 101.4 us | 105.0 MB | 1035 GB/s | 4.00 FLOP/B |
| 4, 5120, 8704 | 70.5 us | 89.2 MB | 1265 GB/s | 4.00 FLOP/B |
| 4, 8192, 5120 | 69.5 us | 84.0 MB | 1209 GB/s | 3.99 FLOP/B |

**10 个不同 shape 全部落在 1035 ~ 1329 GB/s，高度一致** —— 说明这已经逼近本机有效 HBM 带宽上限，**kernel 实现本身没有优化空间**。

算术强度只有 1 ~ 4 FLOP/byte，远低于本机的算力/带宽比，是深度访存受限。

### 优化动作

**唯一有效的方向是减少搬运的字节数，或提高数据复用。** 按预期收益排序：

1. **W8A8 权重低比特量化**（权重字节减半）
   影响最大的那一次 `(4,5120)@(124160,5120)`：数据量 1272 MB → 636 MB，按带宽受限估算 966 us → ~483 us。
   外推 MatMulV2 稳态 161.7 ms → **~85 ms 量级**。
   注意：需扣掉激活量化/反量化的额外开销，实际收益要实测，这是**估算不是实测值**。

2. **提高有效 M（batch）**
   M 从 1/4 提到 16/32，算术强度线性上升，同样字节数摊到更多计算上。
   当前受 `--max-running-requests 20` 与压测并发限制。

3. ~~权重预取 / 常驻~~ —— **明确否定**。kernel 已经跑到 1.3 TB/s 的带宽上限，预取无法超过带宽。

### 收益与验收

- 量化前后对比同一 shape 的实测带宽与耗时；
- 端到端对比 TPOT（需先有阶段二基线，见附 B）。

---

## 问题 3（P1）小 kernel 碎片化

### 现象

全窗口 14602 个 kernel 中，80.5% 单次耗时不足 20 μs。

### 证据

```
全窗口 14602 个 kernel，其中 <20us 的 11758 个 = 80.5%，合计 59.4 ms

数量最多的几类（<20us 范围内）：
  Slice        1894
  Cast         1177
  Fill          945
  ConcatD       371
  OnesLike      302
```

> 复核提示：`hcom_allReduce_` 也在 <20us 列表里（1862 次），但它是 **双份记录**（每条通信有两条重复行，一条带 Stream ID、一条 `Stream=N/A`），统计时须按 `Stream ID != 'N/A'` 过滤，否则数量虚高一倍。`move_cache`、`_conv_state_rollback` 同样存在这个现象。

### 优化动作

方向明确但收益未定，需先重采数据（见附 B）：

- Slice × 1894、ConcatD × 371：QKV 切分改为 `view`/`narrow` 而非 `slice + cat`
- Fill × 945、OnesLike × 302：改用缓存的常量 buffer，不在每步重建
- Cast × 1177：统一模型 dtype，避免 bf16↔fp32 往返

### 为什么不给收益数字

这些算子的设备耗时合计只有 59.4 ms（且含重复记录），真正的代价在 host 侧 launch 与流上排队。而本份采集开了 `with_stack=True`，产生 178.5 万个 python_function 事件，**host 侧耗时被严重放大，不能作为收益依据**。必须重采后才能估。

---

## 问题 4（P1）每步的 mamba 状态搬运只用 1 个核（单 program 串行搬完整个请求）

### 现象

两个 Triton 搬运 kernel 各只启动 **1 个 program**。在 bs=1 时这**不是 bug**（grid 本来就等于请求数），
但结果是：一个请求的全部状态搬运被**串行做完**，24 个 AI 核里只用到 1 个，且都在每步的关键路径上。

### 证据

```
move_cache_dynamic_last_kernel_h_block   AI_VECTOR_CORE  Block Num=1
  t+1697.01  533.7us | t+1788.10 533.1us | t+1867.89 542.6us
  t+1944.61  529.6us | t+2020.72 540.1us | t+2097.61 529.8us
  6 次真实调用，合计 3208.9 us（每步 1 次）
  另有 6 条 ~1us 的重复记录（同 问题3 的复核提示）
  输入 (48,21,4,24,128,128) → 输出 (48,161,24,128,128)
_conv_state_rollback_kernel              Block Num=1
  t+1698.64  79.8us | t+1789.79  78.9us | t+1869.48 125.9us
  t+1946.45 128.5us | t+2022.32 103.9us | t+2099.68 104.0us
  6 次真实调用，合计 621.0 us（每步 1 次）
  另有 6 条 ~1us 的重复记录
```

两个 kernel 合计 3.83 ms，占稳态 223.8 ms 的 **1.71%**。

`move_cache` 的 kernel 内部结构（源码 `sgl_kernel_npu/mamba/mamba_state_update_triton.py:21`）：

```python
for l in range(num_layers):                              # 48 层
    for h_start in range(0, h_dim, H_BLOCK_SIZE):        # h_dim=24, H_BLOCK_SIZE=2 → 12 次
        ...
        src_block = tl.load(src_addr + linear_offset, mask=mask, other=0)
        tl.store(dst_base_addr + linear_offset, src_block, mask=mask)
```

即一个 program 独自串行执行 **48 × 12 = 576** 个互不依赖的 `(2,128,128)` 数据块搬运
（按 `h_block_size=2` 硬编码，每块仅 64 KB）。**这 576 份工作天然可并行，却全排在一条 program 里。**

### 根因（重要，与初判不同）

**`Block Num = 1` 不是 kernel 写得串行，是因为 bs=1。**

两个 kernel 都是「一个 program 处理一个请求」的写法：

```python
# move_intermediate_cache
grid = (len(dst_indices_tensor),)          # = 请求数
# conv_state_rollback
grid = (num_requests,)                     # = 请求数
```

本次采集的 `bs=1`（6 个 `step[TARGET_VERIFY bs=1]`），所以 grid = 1 → 24 个核里只用 1 个。**批量增大时它会自然并行，不需要改。**

**但真正的低效在别处**：单个 program 要串行搬完一个请求的全部数据。

```
move_cache 每次搬运：48 层 × 24 头 × 128 × 128 = 18,874,368 元素
  读写合计  75.5 MB（bf16）/ 151.0 MB（fp32）
  耗时 533 us → 实测 142 GB/s（bf16）~ 283 GB/s（fp32）
```

对照问题 2 测出的整机有效带宽 **~1300 GB/s**（24 核共同工作），单核只能跑到 **1/9 ~ 1/5**。
（注意口径：这个差距说明"活没分出去"，不一定说明"单核搬得慢"，详见下方收益一节。）

### 优化动作

**方向是「请求内并行」，不是「把请求切给多核」。** 具体做法：在原 kernel 里把 grid 从 `(num_requests,)` 扩成 `(num_requests, num_splits)`，按层（`num_layers=48`）或头（`H=24`）切分，让一个请求能占满多个核。

- `move_cache`：按层切 48 份最自然，与现有 `for l in range(num_layers)` 循环直接对应；
- `_conv_state_rollback_kernel`：同样按层切（`num_layers=48`）。

### 收益与验收

**先说清楚口径**：问题 2 测到的 ~1300 GB/s 是 **24 个核一起工作**的整机带宽。单核吃不下这个数，
所以 142 GB/s 未必是"kernel 写得差"，很可能已经接近**单核的上限** —— 提升空间来自**把活分给更多核**，不是让单核变快。

按此推算（`move_cache` 每次搬 75.5 MB，bf16）：

| 口径 | 依据 | 单次耗时 | 收益 |
|---|---|---|---|
| 现状 | 1 核 | 533 us | — |
| **理论上限** | 75.5 MB ÷ 1300 GB/s（24 核共享带宽打满） | **~58 us** | 9× |
| **中性预期** | 48 层切给 24 核（每核 2 层，负载均衡），按流式搬运通常能拿到峰值带宽的 50~70% | **~110 ~ 180 us** | 3 ~ 5× |
| 保守 | 只拿到 2 倍 | ~265 us | 2× |

换算到每步总收益（两个 kernel 合计 3.83 ms）：

- 中性：**3.83 → ~0.9 ~ 1.2 ms**，每步省 **2.6 ~ 2.9 ms**，占稳态 223.8 ms 的 **~1.2%**；
- 乐观：3.83 → ~0.5 ms，每步省 3.3 ms，占 **~1.5%**。

**验收**：改后 `Block Num > 1`，单次耗时下降，且**数值精度完全一致**（这是纯搬运，逐元素 bitwise 相同）。

> **两条重要限定**：
> 1. **收益会随 batch 增大而自然消失** —— grid 已经等于请求数，并发 20 时核本来就占满了。所以本条只在 bs=1 的 decode 场景下成立，上生产并发后优先级大幅下降。这是它只给 P1 的原因。
> 2. **要区分"提速 9 倍"和"省 3 ms"** —— 9 倍是单次 kernel 的提速比，但它在稳态里只占 1.71%，所以端到端最多省 1.5%。**不要用 9 倍去描述收益。**

---

## 附 A：已排除项（不要在这上面花时间）

### A1. 两次 16 ms 的五维 Transpose —— 初始化开销，稳态不复现

```
aclnnIndexPutImpl_TransposeAiCore_Transpose  (48,161,24,128,128)
  t+3.31 ms   16052.1 us
  t+21.41 ms  16087.2 us
```

- 两次都落在采集窗口最开始的 25 ms 内，在此之前只有 8 个 kernel、18.3 ms 设备时间；
- 该 shape 在全窗口只出现这 2 次，之后再不出现（后续同链路的 `move_cache` 用的是 `(48,21,4,24,128,128)`）；
- 阶段 C 稳态下 Transpose 一共只有 12 次 / 0.19 ms，占 **0.09%**。

**结论：这是 GDN/mamba 状态缓存的首次初始化，不是稳态成本。** 任何以"Transpose 占 20%"或"KV/状态缓存链占 23.6%"为前提的优化都建立在 warmup 上 —— 同一条链在稳态只有 6.06 ms / **2.71%**。

### A2. 采集本身的两个坑（复核时踩到）

1. **窗口 84% 是空闲**：rank0 的 `step_trace_time.csv` 为 Computing 325.6 ms / Communication 10.0 ms / Free 1790.0 ms / Stage 2125 ms。**任何"占比"的分母若是 Stage，都会把一切优化收益稀释到 15% 以下。**
2. **通信条目双份记录**：见问题 3 的复核提示。

---

## 附 B：本清单的数据边界

以下几点**数据不足以判断**，因此没有列入上面的问题清单：

| 事项 | 为什么判断不了 |
|---|---|
| 稳态下完整的算子占比 | 阶段 C 只有 6 个 decode step，样本太少 |
| 高并发下的表现 | 本 profile 全程 **bs=1**，无任何高并发样本。当前阶段范围也限定为单并发，故不列入 |
| 小 kernel 碎片化的真实收益 | host 侧数字被 `with_stack=True` 放大（178.5 万 python_function 事件） |
| 100–1600 ms 那 1.5 s 的等待是不是真问题 | 从 `prefill_delayer.py` 看，`queue_min_ratio=0.7` + `max_delay_ms=20000`，低并发压测下 delayer 大概率是在「等批」，属配置行为而非故障。需实验区分 |
| 绝对 TPOT / TTFT | profiling 本身有开销。**已于 2026-09-16 补到无 profiling 的 bs=1 端到端数据**（Median TPOT 13.97 ms / Median ITL 13.79 ms），已换算为真实步长 55.1 ms 并与本清单交叉核对（见问题 1）。但为**单次运行、方差未知**，且负载为「高前缀命中」特例 |
| host 侧任何绝对耗时 | 同上，`with_stack=True` |

**建议的补充采集参数**（用于闭环上表，也是 G3 门禁内容）：

```python
profiler_level = ProfilerLevel.Level1
with_stack = False
experimental_config = torch_npu.profiler._ExperimentalConfig(
    data_simplification=True,
    aic_metrics=AicoreMetrics.PipeUtilization,
)
# + schedule(wait=0, warmup=1, active=N) 配合每步 profiler.step()
# + 跳过初始化与 graph capture，只采稳态 decode，并发 >= 8，步数 >= 30
```

---

## 行动优先级汇总

| # | 问题 | 类别 | 确定性 | 预期量级 |
|---|---|---|---|---|
| 1 | 每步一道 NonZero 同步屏障 | 替换已有接口 | 现象确凿；已有 bs=1 压测独立佐证 | 真实步长 55.1 ms 中 ~18 ms 是 host 停顿 → **下限 7% / 中性 18% / 上限 33%**<br>**代码已落地**（[`ops/mamba-conv-state-track-copy/`](../ops/mamba-conv-state-track-copy/README.md)，单测 8/8）；**端到端收益未验证，需重启服务后测** |
| 2 | MatMulV2 打满带宽，需减字节 | 量化（部署环节） | 确凿（实测 1.3 TB/s 上限） | MatMulV2 161.7 → ~85 ms（估算） |
| 3 | 小 kernel 碎片化 | 融合 / 去重 | 数量确凿，收益待重采 | 待定 |
| 4 | 状态搬运只用 1 个核 | 请求内并行（按层切 grid） | 现象确凿；`Block Num=1` 系 bs=1 所致，非缺陷 | 3.83 → ~0.9~1.2 ms（中性）；每步省 ~1.2% |

> 问题 1 与 2 是仅有的两个"有量级"的项，建议优先。问题 3、4 量级在 1~2%，适合顺手做。
>
> 问题 1 的量级**不要**用 `api_statistic.csv` 里 `aclnnNonzeroV2` 的 124.83 ms 直接代入 —— 那里面 81% 是设备本来就该干的活，且与 `aclrtSynchronizeStream` 是同一笔。
>
> **适用范围（2026-09-16 补充）**：本清单的问题 1~4 均基于 **bs=1** 的 profile，
> 并与**单并发压测**交叉核对过（见 [`docs/02-基线数据采集.md`](../docs/02-基线数据采集.md)）。
> **本阶段范围限定为单并发**，高并发下的表现不在覆盖范围内，也不在附 B 的分析目标内。
