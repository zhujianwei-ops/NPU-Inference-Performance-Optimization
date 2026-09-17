# 基线数据 · Qwen3.8-27B · TP=2 · NEXTN · **单并发**

> `exp_id`：**`base-20260916-qwen3.8-27b-c1`**
> 采集日期：2026-09-16
> 原始输出：[`baseline-20260916-qwen3.8-27b-c1.raw.txt`](baseline-20260916-qwen3.8-27b-c1.raw.txt)（原样存档，未加工）
> 阶段文档：[`docs/02-基线数据采集.md`](../../docs/02-基线数据采集.md)

> **范围**：本文件只覆盖**单并发（`--max-concurrency 1`）**。多并发点不在当前范围内。

---

## 1. 测试脚本（用户提供）

```bash
python -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang \
    --host 127.0.0.1 --port 6688 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 20 \
    --gsp-system-prompt-len 57600 \
    --gsp-question-len 6400 \
    --gsp-output-len 1000 \
    --max-concurrency 1 \
    --num-prompts 20 \
    --request-rate inf
```

### 1.1 ⚠️ 三个长度参数是**目标值**，不是实际 token 数

读源码确认（`python/sglang/benchmark/datasets/generated_shared_prefix.py:154`）：

| 参数 | 值 | 实际 |
|---|---|---|
| `--gsp-system-prompt-len` | 57,600 | 目标值 |
| `--gsp-question-len` | 6,400 | 目标值 |
| 目标 ISL | **64,000** | **实测 68,484**（+7.0%） |
| `--gsp-output-len` | 1,000 | 实测 1,000（精确） |

**为什么差 7%**：`common.py:79` 的 `gen_prompt()` 是 `random.choices(vocab, k=token_num)` 抽 **token ID**，再 `tokenizer.decode()` 成文本。
随机 token ID 会跨字符边界，**decode→encode 不回环**，重新编码后 token 数变多。

> **引用时必须用实测值 68,484。**

### 1.2 负载特征（很重要，决定了这份数据的适用边界）

- `--gsp-num-groups 1` → **20 个请求共用同一条 57,600 token 的 system prompt**
- 即：**这是一份「前缀命中率极高」的负载**，`--mamba-radix-cache-strategy extra_buffer` 的前缀缓存被充分使用
- 20 个请求 × 68,484 ISL ≈ 1.37M input tokens，但**实际需要 prefill 的只有 1 份 system prompt + 20 份 question**

> **不要**把这份结果当作「常规随机负载」的基线。它的 TTFT 是前缀缓存命中后的 TTFT。
> 常规负载（无共享前缀）的 TTFT 会显著更差，需另测。

> ⚠️ `--chunked-prefill-size 32768` **小于 ISL 68,484**，因此每个 prompt 的 prefill 必然被切成 **≥3 块**。

---

## 2. 运行配置（实测取自运行中进程 `/proc/9558`，非文档抄录）

启动命令与 `start-qwen3.8-27b-prefix-cache.sh` **逐参数一致**：

| 项 | 值 |
|---|---|
| 模型 | `/home1/model/Qwen3.8-27B`，**bf16 未量化** |
| 并行 | `--tp-size 2`，device 6/7 |
| 投机解码 | NEXTN，`num-steps 3` / `topk 1` / `num-draft-tokens 4` |
| prefill | `--chunked-prefill-size 32768` |
| 调度 | `--max-running-requests 20`、`--enable-prefill-delayer`（queue-min-ratio 0.7 / max-delay 20000ms） |
| 图模式 | `--cuda-graph-bs-decode 1 2 5 10 15 17 19 20` |
| 代码 | `/sgl-workspace/sglang` commit `0bcd8223`，分支 `v0.5.19`，**源码零改动** |

环境变量：`ASCEND_USE_FIA=1` `GDN_ATTN_BACKEND_TRITON=1` `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
`SGLANG_SET_CPU_AFFINITY=1` `STREAMS_PER_DEVICE=32` `HCCL_OP_EXPANSION_MODE=AIV`
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` `ASCEND_RT_VISIBLE_DEVICES=6,7`

---

## 3. 基线指标

| 指标 | 值 |
|---|---|
| Benchmark duration | **305.10 s** |
| Successful requests | 20 / 20 |
| Total input tokens | 1,369,687 |
| Total generated tokens | 20,000 |
| Output token throughput（均值） | **65.55 tok/s** |
| Output token throughput（峰值） | 73.00 tok/s |
| Total token throughput | 4,554.84 tok/s |
| Concurrency（均值） | 1.00 |
| Peak concurrent requests | 2 |
| **Accept length** | **3.86**（/ 4 = 接受率 **96.5%**） |

| 分位 | E2E (ms) | TTFT (ms) | TPOT (ms) | ITL (ms) |
|---|---|---|---|---|
| Mean | 15,253.65 | 986.91 | 14.28 | 14.28 |
| **Median** | **14,580.97** | **588.17** | **13.97** | **13.79** |
| P90 | 16,568.28 | 2,251.54 | 14.30 | 14.00 |
| P95 | 19,010.94 | 2,612.16 | 14.78 | 14.18 |
| P99 | 20,078.23 | 4,399.44 | 18.78 | 27.72 |
| Max | — | — | — | 59.48 |

**接受率 96.5%** —— NEXTN 配置（4 draft tokens，接受 3.86）本身非常有效。

---

## 4. 关键推导：真实 decode 步长 = 55.07 ms

### 4.1 推导（可用输出里的数字逐步复算）

先验证 TPOT 的口径 —— `(E2E − TTFT) / (output_len − 1)`：

```
(15253.65 − 986.91) / 999 = 14.28 ms   ← 与输出里 Mean TPOT = 14.28 完全一致 ✓
```

**所以 TPOT / ITL 是「摊到每个 token」的均值，不是单个 step 的耗时。** 必须乘 accept length 才是步长：

```
总 decode 墙钟 = 20 × (15253.65 − 986.91) ms = 285,334.8 ms
总 step 数     = 20,000 token ÷ 3.86 (accept length) = 5,181 step
每 step        = 285,334.8 ÷ 5,181 = 55.07 ms
```

> ⚠️ **这一步很容易错**：直接拿 Median ITL 13.79 ms 当步长会低估 3.86 倍。
> 在分析算子清单时**先犯过这个错**，记在这里备查。

### 4.2 与 profiling 交叉核对 —— 两个独立测量都指向 ~20 ms 的 host 停顿

| 口径 | 来源 | 步长 |
|---|---|---|
| **真实** decode step（bs=1，无 profiling） | 本次压测推导 | **55.07 ms** |
| profiled step（bs=1，开 profiling） | 算子清单，相邻 `aclnnNonzeroV2` 间隔 | **~76 ~ 78 ms** |
| 其中设备工作量 | 算子清单，phase C 223.8 ms ÷ 6 step | **~37.3 ms** |

两条独立的算术都落在同一个数上：

```
① 76 (profiled) − 55.07 (真实)      = 21 ms   ← profiling 开销 + 该 step 的额外停顿
② 55.07 (真实) − 37.3 (设备工作)     = 17.8 ms ← 真实 step 里的非设备时间
                                       ↑ 两者都 ≈ 算子清单实测的 20.6 ms 同步屏障
```

**结论：真实 decode step 里约有 18 ~ 21 ms 是 host 侧停顿，与 `aclnnNonzeroV2` 那道 20.6 ms 屏障吻合。**

这比之前更可信了 —— 之前只有 profiling 一个来源，现在有了无 profiling 的独立佐证。

### 4.3 于是问题 1 的收益上限可以收紧

```
真实 step          55.07 ms
  ├─ 设备工作      ~37.3 ms   ← 硬下限，步长不可能低于它
  └─ host 停顿     ~18 ms     ← 屏障所在

下限（只回收设备空闲）     3.9 / 55.07 =  7%
中性                       10 / 55.07 = 18%
上限（步长收敛到设备下限） 18 / 55.07 = 33%   → 55.07 → ~37 ms，约 1.49×
```

**比之前的 5% / 10~20% / ≤27% 整体上修。** 已回写至 `analysis/算子问题清单` 问题 1。

### 4.4 顺带得到的量化系数：profiling 开销 = 步长 × 1.38

`76 ÷ 55.07 = 1.38`。

**用 profiling 数据推断端到端收益时，必须记住这个放大系数。** 所以问题 1 的验证方案要求：
**改前 / 改后对比必须关 profiler**，否则收益会被淹没。

---

## 5. 读这份数据必须注意的点

### 5.1 ⚠️ 单次运行，**不满足 G2**

方案要求每点**重复 ≥ 3 次**给方差、矩阵跑完、方差 < 5%。当前 **1 次**，方差未知，**不能冻结基线**。

### 5.2 ⚠️ 负载只有 1 个点，且是「高前缀命中」的特例

ISL 68,484 / OSL 1000 / 并发 1 / 20 请求共享 1 条 system prompt。
**短 prompt、无共享前缀、纯 decode、纯 prefill 全部未覆盖。**

### 5.3 ⚠️ 输出**不是位级可复现**

`Total generated tokens (retokenized)` = **19,670**，而 `Total generated tokens` = 20,000（差 1.65%）。
生成 20,000 token 后重编码计数不同 → **输出文本与重编码计数不能用于逐 token 对齐**。

含义：**做优化前后对比时，不能指望逐 token 对齐**，只能比统计量；
若要严格对比，需先固定 seed / 确认采样温度。

### 5.4 `--max-concurrency 1` 却出现 `Peak concurrent requests: 2`

最大值到过 2。推测与 `--enable-prefill-delayer` 或 overlap plan stream 的在途请求有关，
**未验证**。不影响 `Concurrency: 1.00` 的均值结论。

### 5.5 ITL 尾部很干净 —— 这对做优化对照是好消息

Median 13.79 → P99 27.72 → Max **59.48 ms**。**没有长尾。**

说明并发 1 时调度是干净的，**这份数据里没有调度抖动引入的噪声**。
因此它适合作为问题 1、问题 2 优化前后的**对照基准**：任何变化都应归因于改动本身，而非抖动。

---

## 6. 与其他文档的关联

| 文档 | 关联点 |
|---|---|
| [`docs/02-基线数据采集.md`](../../docs/02-基线数据采集.md) | 本条归档处；G2 未通过 |
| [`analysis/算子问题清单-Qwen3.8-27B-20260916.md`](../../analysis/算子问题清单-Qwen3.8-27B-20260916.md) | 4.2 节的交叉核对已回写「问题 1 · 预期收益」，收益上修至 7 / 18 / ~33% |
| [`docs/07-Rank1通信异常排查.md`](../../docs/07-Rank1通信异常排查.md) | 其 Step 0（关 profiling 复现）现在**更该做** —— 本次数据已独立指向同一结论 |

---

## 7. 待补

- [ ] **重复 ≥ 3 次**给方差（G2 硬门槛）
- [ ] 补短 prompt / 无共享前缀 / 纯 decode / 纯 prefill
- [ ] 归档请求数据集（`--gsp-*` 是程序生成的，固定 `--seed` 即可复现，需记录 seed 具体值）
- [ ] 验证 5.3（输出不可复现）与 5.4（peak concurrency 2）两个现象
