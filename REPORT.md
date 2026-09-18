# 第一轮优化报告 — 2026-09-18

本轮已走通基线、profiling、方案、单算子优化、框架接入、整网验证与归档。实现一个经过验证的小算子融合；尚未完成 MatMul/FIA/HCCL 主热点的深度优化。

## 性能结果

| 指标 | 原始 | 融合后 | 变化 |
|---|---:|---:|---:|
| 成功请求 | 20/20 | 20/20 | 全部成功 |
| 输出吞吐 token/s | 59.0688 | 59.6387 | +0.965% |
| 平均 TTFT ms | 2206.705 | 2116.965 | -4.067% |
| 平均 TPOT ms | 14.7360 | 14.6640 | -0.489% |
| 总耗时 s | 338.588 | 335.353 | -0.956% |

20 条生成文本完全一致，输入/输出长度完全一致。只有单次 A/B，不能据此确认统计显著的整网收益。prefill 路径未改动，TTFT 改善不能归因于融合；需要 A/B/A 重复与方差分析才能确认稳定收益。

原脚本的数据集实际生成 20 条而非 80 条；真实输入 68424–68541 tokens 而非名义 64000。保持 max-concurrency=1、TP=2、BF16、NEXTN、prefix cache 和 NPU Graph。默认 benchmark 预热后不清缓存，属于原脚本 warm-prefix 基线。

## 实现与验证

将 attention 输出门控的 Sigmoid + Mul 融合为一个 Triton Ascend kernel。使用连续 head 向量访存，显式保留 sigmoid 的 BF16 中间舍入。限制在小 M Graph 捕获路径，prefill/eager 回落原实现。试验前两版明显退化，完整保留失败数据。

独立微测重复验证：M=1/4 连续布局 Graph 约 1.21×/1.37×；带 stride 约 2.50×/2.00×。这些数值与 trace 的比值不同，因为前者是独立循环，后者在整网设备负载中采集。

优化后的真实 decode trace（6 步）证实：

| 设备 | 原 Sigmoid+Mul 次数 | 融合次数 | 原累计 us | 新累计 us | 算子组合加速 |
|---|---:|---:|---:|---:|---:|
| 6 | 114+114 | 114 | 769.157 | 312.506 | 2.461× |
| 7 | 114+114 | 114 | 729.536 | 311.187 | 2.344× |

30 组形状/布局精度检查覆盖 M=1,2,4,5,8,10,15,17,19,20,40,60,68,76,80；含真实 QKV row stride、原位/非原位，以及 ±inf、NaN、饱和值。已测结果与原实现逐元素一致。这不等同于所有可能输入的形式化证明。

## Profiling 解释限制

保留两个 rank 的原始数据。当前 legacy profiler 忽略 profile_stages 过滤，每次产生两个阶段，具体阶段以 server.log 和调用内容核验，文件夹名不是阶段事实。自动分析的 stage_inferred 仅供参考，缓存全部命中的 prefill 也可能出现 MatMulV2。

prefix_hit_prefill 首次只命中 32768 tokens；原始基线稳态通常命中 61568，不能直接比较该 prefill 时延。优化后 profiling 复用完整请求缓存，prefill 也不匹配基线；本报告只比较明确识别的 6 步 decode 中门控算子，不声称其他 kernel 或阶段被加速。

NPU profiler 停止时产生 schedule 警告，已核实两 rank 的 CSV 非空、目标调用数完整（114 组），但仍保留原日志供复核。kernel 时间存在重叠，不能简单累加为端到端时间，也不能将两个 rank 相加。

## 当前交付与后续

- 实际运行源码已接入补丁，当前服务启用 `SGLANG_NPU_FUSED_ATTN_GATE=1`，监听 127.0.0.1:6688，使用 NPU 6/7。PID 见 run 目录 `server_optimized.pid`。本轮采集任务均完成。
- 原始启动/测试脚本保持不变；源码默认关闭融合。关闭环境开关并重启即可回退行为。
- 原仓库已有修改未覆盖；补丁只包含本轮新算子和模型调用点。运行前 tracked diff、版本、模型 config 和数据集快照均已保存。
- 主要瓶颈仍是 MatMul、长 KV attention 和 HCCL。具体候选与约束见 `optimization_plan.md`，未测量的加速比明确标为未知。
- 下一轮先做 A/B/A 重复确认微小整网收益，再针对真实 MatMul 形状建立专门微测。通信须先分离等待与有效传输，避免错误归因。

## 产物位置

本轮目录：`runs/20260918T052638Z_baseline`。

- `baseline_original.jsonl`、`optimized_original.jsonl`：完整指标及请求详情。
- `comparison.md/json`：整网对照。
- `profile_summary.md/json`、`optimized_profiles/profile_summary.md/json`：逐 trace、逐 rank 聚合。
- `gate_trace_comparison.json`：真实融合前后设备时间。
- `microbench_gate_v3_validation.json`、`gate_correctness.json`：微测与精度。
- `profiles/`、`optimized_profiles/profiles/`：原始 trace、CSV、DB。
- `patches/0001-npu-attention-gate.patch`：可审核的接入补丁。
- `scripts/`、`kernels/`：复现工具及保留的失败版本。
