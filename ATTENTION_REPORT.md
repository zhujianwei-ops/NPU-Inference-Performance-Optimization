# 第二轮性能优化：长上下文推测验证 Attention

日期：2026-09-18。状态：**性能收益已复现；实验候选，尚未通过完整模型质量验收。**

## 结果

本轮选择的是长 KV、4-token MTP 验证 Attention，而不是低占比元素算子。保持 BF16、TP=2、NEXTN（框架解析为 EAGLE）、原始 prefix-cache 负载不变。通过调整 Attention 的 query 布局、页表和逐 query 有效 KV 长度，改用供应商增量 Attention 路径。

A/B/A/B 四轮各 20 条请求全部成功；每条仍生成 1000 tokens。下面为每种配置两轮的算术平均值：

| 指标 | 基线 | 优化 | 相对变化 |
|---|---:|---:|---:|
| 输出吞吐 token/s | 59.415 | 65.242 | +9.81% |
| 平均 TPOT ms | 14.692 | 13.152 | -10.48% |
| 每轮 P99 TPOT 的平均 ms | 18.860 | 17.054 | -9.57% |
| 平均请求 E2E ms | 16830.002 | 15326.283 | -8.93% |
| 平均 TTFT ms | 2153.096 | 2187.740 | +1.61% |
| 每轮总耗时 s | 336.627 | 306.553 | -8.93% |
| 推测接受长度 | 3.87868 | 3.85692 | -0.56% |

逐轮吞吐：A1=59.082、B1=65.034、A2=59.748、B2=65.450 token/s。两次相邻 A/B 均改善，不是仅一次约 1% 的变化。只有每组两轮，仍不宣称充分统计显著性或所有负载通用收益。

TTFT 没有改善；本轮针对 decode/verify。此前的门控融合在全部 A/B 中保持开启，本轮数字不包含将门控开关一起变更造成的影响。

## 原理

原始 Q=[4,12,256]、TND 4-token causal attention，KV 总长度为 L。第 j 个 query 只能读取 L-4+j+1 个 KV token。

将其重排为 Q=[4,1,3072]，四行页表共享同一 KV cache，有效长度分别为 [L-3,L-2,L-1,L]。这样能使用适合单 query 的 BSH incremental FIA。没有截断上下文、量化权重或降低 dtype，也没有复制 KV 数据。

页表在捕获外每步更新；Graph replay 同步更新四个 Host 侧长度。只优化 batch=1 的目标模型验证阶段，draft、GDN、eager、prefill 和 batch>1 沿用原路径。

## 实际设备时间

对比同轮基线与优化后的 6 步 decode trace。为避免把同 shape 的 draft-extend 算子误分类，下表统计全部 102 次 4-token Attention：基线 102 次 TND，优化组 96 次增量 BSH + 6 次未修改的 draft-extend TND。

| NPU | 原累计 ms | 新累计 ms | 组合加速比 | 每调度步少用的累计 kernel 时间 |
|---|---:|---:|---:|---:|
| 6 | 91.697 | 52.875 | 1.734× | 6.470 ms |
| 7 | 91.490 | 53.187 | 1.720× | 6.384 ms |

按旧同 shape 算子均值看，约 899 us；优化后目标验证算子均值约 495–498 us。独立微测约 2.265×，但 microbenchmark 的 cache 大小、页表宽度及运行上下文与整网不同，因此以这里的实网 trace 为主要依据。

每步少用约 6.4 ms 的 kernel 时间，与整网 TPOT 降低约 1.54 ms/token 的数量级一致；这只是归因参考，不把重叠 kernel 时间或两个 rank 直接相加当作端到端节省。

## 正确性与不能忽略的限制

- 算子测试通过：分页边界、非连续物理页、Graph 长度重绑定、输入/page table 修改后重放、空 padding。
- 短序列独立 FP64 oracle：新旧输出相同；长序列新旧 BF16 输出相对 L2 差约 0.26%，最大绝对差约 0.000122。详见 `ATTENTION_DESIGN.md`，包括初次精度阈值失败及修正依据。
- 原始 20 条随机 token 压测：A1=A2 的 20 条输出完全相同，B1=B2 也完全相同；A 与 B 只有 14/20 完全相同，6 条差异稳定复现。**不能宣称逐 token 无损或将其归为随机波动。**
- 6 项语义 smoke checks（算术、排序、JSON、中文算术、约 54k-token 检索、约 54k-token 算术）两组全部通过，输出文本相同。对应输出 token 的最大 logprob 差约 1.68e-5。
- 并发 2 请求及回到单请求的 smoke check 通过，服务日志确认出现 running-req=2、Graph=True，覆盖 batch>1 回退路径。
- 这些检查不能替代正式语言模型质量集、业务集及长上下文精度评测。若验收要求输出逐 token 完全一致，当前候选不满足，应该保持关闭。

## 适用范围和开关

默认关闭：`SGLANG_NPU_MTP_INCREMENTAL=0`。

启用条件限制为实际 Qwen3_5ForConditionalGeneration 配置、TP=2/DP=1/单机、BF16、Ascend backend、page=128、EAGLE/NEXTN topk=1、4 draft tokens、24 Q heads/4 KV heads/head_dim=256，并且目标模型 Graph 的 batch=1。其它条件不启用。短上下文收益很小，不能外推到短文本或并发 20。

独立实验启动脚本：

```bash
bash /home/zhujianwei/task01/NPU-Inference-Performance-Optimization-main/start-qwen3.8-27b-mtp-incremental.sh
```

如已有服务，应先停止自己启动的服务，避免端口/设备冲突。当前已有实验服务在运行，不需要再次执行上面的命令。

关闭本候选并重启，可保留上一轮门控优化：

```bash
cd /sgl-workspace
export PYTHONPATH=/sgl-workspace/sglang/python:$PYTHONPATH
export SGLANG_NPU_FUSED_ATTN_GATE=1
export SGLANG_NPU_MTP_INCREMENTAL=0
bash /home/zhujianwei/task01/NPU-Inference-Performance-Optimization-main/start-qwen3.8-27b-prefix-cache.sh
```

当前运行服务：127.0.0.1:6688，NPU 6/7，两个实验开关均开启。进程组 PID 记录在本轮 `attention_repeat_server.pid`。原始启动、压测脚本未改。

## 交付文件

- `patches/0002-npu-mtp-incremental-attention.patch`：本轮独立补丁；已通过 reverse-check，可单独回滚，不覆盖原有修改。
- `kernels/mtp_incremental.py`：适用条件与因果长度辅助代码。
- `scripts/bench_mtp_attention.py`：独立微测。
- `scripts/check_mtp_attention.py`：分页、因果、Graph 及 oracle 检查。
- `scripts/attention_semantic_probes.py`、`check_attention_batch_fallback.py`：模型 smoke checks。
- `scripts/summarize_attention_experiment.py`：四轮结果、输出一致性、逐 rank profiling 汇总。
- `ATTENTION_DESIGN.md`：设计与试验方法。
- 本轮目录 `20260918T062358Z_attention`，绝对路径由 `ATTENTION_RUN` 保存。包含四轮完整 JSONL、日志、两组原始 profiling、精度结果和回滚快照。

本轮完成了一个收益明显的算子执行路径优化候选。下一步验收重点是正式模型质量，而不是继续用随机压测的文本一致性替代质量评测。
