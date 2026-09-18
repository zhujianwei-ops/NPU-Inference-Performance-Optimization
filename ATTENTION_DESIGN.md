# 第二轮：长上下文 MTP 验证 Attention 优化

## 问题与证据

第一轮 decode trace，rank1 的 TND 4-token FusedInferAttentionScore 102 次累计 91.219 ms（含 target verify 与 draft extend），平均 894.3 us。shape 为 Q=[4,12,256]，KV 为 paged BF16，page=128，KV heads=2，真实上下文约 68.5k。这是远大于 Sigmoid/Mul 的优化对象。

新方案针对 target verify 的 96 次调用（6 个调度步 × 16 个 full-attention 层），不改 draft 模型或 GDN 状态。

## 等价变换

设一次链式验证有 w=4 个 query，KV 总长度 L 包含这 4 个 token。原右下对齐 causal attention 的第 j 个 query 只能访问 [0, L-w+j]，即 L-w+j+1 个 key。

将 Q=[4,12,256] reshape 为 [4,1,3072]，把一个 request 表达成 4 个长度为 1 的增量 query。4 行 block table 指向同一份 KV；各行 KV 长度设置为 [L-3,L-2,L-1,L]。调用 BSH incremental FIA，无须稠密 mask。没有复制 KV 数据，没有减少可见上下文，没有改变 dtype 或模型权重。

原 TND 和新 BSH 由供应商库选择不同实现，浮点计算顺序可能不同。因此数学上的因果范围一致不代表 BF16 输出逐位一致。

## Graph 的关键约束

- expanded block table 预分配，在捕获外随原 block table 每步更新，避免每层重复复制。
- replay 的 Host IntArray 必须从单个 L 变为 [L-3,L-2,L-1,L]，否则捕获值冻结会导致错误结果。
- 空 padding sequence 的全部长度保持 0，不暴露未初始化 KV。
- 只对已验证模型/配置、bs=1、4-token target verify Graph 使用新路径。bs>1、eager、prefill、draft 路径不改。
- 环境开关 `SGLANG_NPU_MTP_INCREMENTAL=1` 显式启用，默认关闭；`SGLANG_NPU_FUSED_ATTN_GATE=1` 在本轮全部 A/B 中保持一致。

## 验证

微测：68.5k 上下文 bs=1，555.63→245.33 us，2.265×；bs=2 为 1.493×，但尚未接入 bs=2。4k 短上下文几乎无收益，不能外推到所有长度。

独立检查覆盖 L=4,5,127,128,129,255,256,257,2049,68424,68500,68501，随机非连续页映射、Graph 长度更新、修改 Q/page table 后 replay、零长度 padding。

短序列用独立 CPU FP64 attention 作 oracle；原实现和候选相同，相对 L2 约 0.14–0.23%。初次测试采用统一绝对误差 0.002 时，原实现本身也超出该界（L=4 最大误差约 0.00731）。保留初次失败日志，改为分别比较原实现和候选对 oracle 的误差，要求候选误差不显著增大；并非将原始 BF16 路径当作 FP64 精度。

长序列候选相对原实现的 L2 差约 0.26%，最大绝对差约 0.000122；Graph replay 与候选 eager 逐元素一致。模型级另测 20 条原始压测输出和 6 项语义 smoke checks。smoke checks 不替代正式模型精度评测。

## 测量方法

A1/B1/A2/B2 顺序执行，每组原始 shared-prefix 负载实际 20 条，1 个预热请求，1000 输出 token，max-concurrency=1。A1 清空现有服务缓存，其余组重启服务，使每次开始测量时均只有同一 warmup 的前缀缓存。A1 编译/JIT 已暖，其他组启动 Graph 捕获后 warmup；记录差异，不声称随机化独立重复。

profiling 与 semantic probes 在性能测试结束后串行执行。吞吐及延迟只取无 profiler 的 benchmark 结果；两个 rank 单独分析，不直接累加 kernel time 当作整网时间。
