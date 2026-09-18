# 真正算子层优化路线

基于 2026-09-18 Qwen3.5-27B、Ascend910、TP2、BF16、EAGLE 4-token verify 的 profiling。

## 优先级

1. TND 4-token verify Attention 自定义 kernel：最大潜在收益，难度最高。
2. M=4 小矩阵 GEMM + SwiGLU epilogue 融合：最适合作为第一 个可交付的 Ascend C 算子。
3. M=1/M=4 Linear 权重格式与 GEMM 专用 kernel：先做真实 shape autotune，再决定是否替换 CANN MatMulV2。
4. AllReduce + residual/norm 融合：收益可能很大，但必须先拆解通信等待和有效传输，不作为第一阶段。

## 关键数据

- MTP verify TND FusedInferAttentionScore：rank0 约 93.4 ms / 6 步，rank1 约 91.2 ms / 6 步，主要 shape Q=[4,12,256]、KV context 约 68.5k。
- MatMulV2 rank0 decode：约 160.3 ms / 6 步；最大 group 为 [4,5120]x[17408,5120] 约 58 ms、[4,8704]x[5120,8704] 约 29 ms、[4,5120]x[8192,5120] 约 22 ms。
- HCCL/allreduce 记录约 231 ms，但存在重叠和等待，不能直接按这个数字承诺收益。

## 首选交付：GEMM + SwiGLU 融合

实现一个仅覆盖真实 M=4、BF16、固定 hidden/intermediate shape 的 Ascend C custom op：

`Y = down_proj(silu(gate_up[..., :I]) * gate_up[..., I:])`

第一版不要把 down GEMM 也融合进去，先做：

`gate_up GEMM + SwiGLU epilogue`

这样只需替换 gate_up 的 MatMulV2，并避免 gate_up 输出落回全局内存后再启动 SwiGlu。需要验证：

- gate_up 权重已转换/转置后的布局；
- output shape [4,17408]；
- BF16/FP32 accumulation 与 CANN MatMulV2 的误差；
- NPU Graph capture；
- M=1 decode 和 M=4 verify 两套 tile；
- 不满足 shape 时回退原 MatMulV2 + SwiGlu。

若 CANN/Cube 编程成本过高，先用 `torch_npu.npu_quant_matmul` 或现有矩阵 API 做基线，不能把 layout reshape 当成算子优化。

## 高收益长期项目：TND 4-row causal FIA kernel

重写为真正的 Ascend C attention kernel：一个 kernel 处理 4 个 query row，每行读取同一 paged KV，但有效长度分别为 L-3,L-2,L-1,L；使用 online softmax 和 KV tile 复用。目标是保留 TND API 语义和因果 mask，同时消除当前 TND PromptFlash 路径的低效 tiling。该项目要从 CANN custom op/OpPlugin 或 torch C++ extension 接入，不能只换 BSH 调用。

验收必须逐 token 对比原 TND 路径、FP32/FP64 oracle 和模型质量；如果数学等价但 BF16 顺序改变造成文本变化，需要把精度差异作为项目结果，而不是隐藏。

## 不建议直接做的方案

- 直接修改 `/usr/local/Ascend` 中的 CANN 二进制：不可维护、不可回滚、难以归因。
- 只修改 `input_layout`：这是调用路径优化，不是 kernel 实现优化。
- 先做 HCCL 自定义通信：需要先确认有效通信、等待和 overlap，成本高且容易错误归因。
- 优先优化 Sigmoid+Mul：已证明只占 decode kernel 累计时间约 0.15%–0.23%。
