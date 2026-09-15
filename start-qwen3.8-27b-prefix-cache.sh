# ============================================================
# Before running, update the following variables:
#   MODEL_PATH: path to the model weights directory
#   HCCL_SOCKET_IFNAME: network interface name for HCCL
#   GLOO_SOCKET_IFNAME: network interface name for Gloo
# ============================================================
MODEL_PATH=/home1/model/Qwen3.8-27B/
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_USE_FIA=1
export GDN_ATTN_BACKEND_TRITON=1
export GLOO_SOCKET_IFNAME=enp196s0f0
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=enp196s0f0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32
export ASCEND_RT_VISIBLE_DEVICES=6,7

sglang serve \
    --model-path /home1/model/Qwen3.8-27B \
    --host 127.0.0.1 --port 6688 \
    --tp-size 2 \
    --nnodes 1 \
    --attention-backend ascend \
    --device npu \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 232768 \
    --mamba-radix-cache-strategy extra_buffer \
    --trust-remote-code \
    --max-running-requests 20 \
    --max-mamba-cache-size 160 \
    --mem-fraction-static 0.82 \
    --cuda-graph-bs-decode 1 2 5 10 15 17 19 20 \
    --enable-prefill-delayer \
    --prefill-delayer-queue-min-ratio 0.7 \
    --prefill-delayer-max-delay-ms 20000 \
    --dtype bfloat16 \
    --mamba-ssm-dtype bfloat16 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
