#!/bin/bash
# tau-bench retail GRPO on 8xH100 — STANDARD slime recipe + MetaGen user-sim.
#
# This is the unmodified upstream recipe (colocate, batch 256, dynamic-sampling
# on, TP=2, native pause/NCCL weight-sync) with only:
#   - 8 GPUs
#   - user simulator routed through MetaGen "Llama Public API" (OpenAI-compatible)
#     -> set LLAMA_API_KEY in your shell before running. Model/url default to
#        gemini-3-flash-preview-fair @ api.llama.com/compat/v1 (see generate_with_tau.py).
#   - wandb -> meta.wandb.io
#   - paths under ${BASE} (default /root) — export BASE=/your/path if different.
#
# Portable code fixes that ship with this branch (NOT box-specific):
#   - trainable_agents.py: env.step() offloaded via asyncio.to_thread (concurrent rollout)
#   - generate_with_tau.py: user-sim via OpenAI SDK (litellm 1.87 can't parse Gemini),
#       empty/429 retries, strip Gemini thought_signature, Sample.Status enum, traj retry
#   - loss.py: zero-KL placeholder shape fallback when log_probs is reused-in-loss
#
# NOTE: do NOT carry over the GB200 hacks (SLIME_HOST_IP, GLOO_SOCKET_IFNAME,
# CUDA_DEVICE_MAX_CONNECTIONS!=1, sglang pause no-op, non-colocate) — those were
# workarounds for one broken box and will hurt here.

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"

# Base dir for checkpoints + data. Override: export BASE=/your/path
BASE=${BASE:-/root}

CKPT_ARGS=(
   --hf-checkpoint ${BASE}/Qwen3-4B-Instruct-2507/
   --ref-load ${BASE}/Qwen3-4B-Instruct-2507_torch_dist/
   --load ${BASE}/Qwen3-4B-Instruct-2507_slime/
   --save ${BASE}/Qwen3-4B-Instruct-2507_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${BASE}/tau-bench/retail_train_tasks.jsonl
   --input-key index
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 256
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data retail-dev ${BASE}/tau-bench/retail_test_tasks.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 1024
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-tau-bench
   --wandb-group qwen3-4B-h100
   --wandb-host https://meta.wandb.io
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   # MetaGen gemini-flash-fair sustains high concurrency; uncomment to throttle
   # if you ever see rate-limit errors from the user simulator:
   # --sglang-server-concurrency 32
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_tau.generate
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 8xH100
NUM_GPUS=8
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 \
   --temp-dir ${BASE}/shared/ray_temp

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${BASE}/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"LLAMA_API_KEY\": \"${LLAMA_API_KEY:-NONE}\",
    \"TAU_USER_MODEL\": \"${TAU_USER_MODEL:-gemini-3-flash-preview-fair}\",
    \"TAU_USER_BASE_URL\": \"${TAU_USER_BASE_URL:-https://api.llama.com/compat/v1}\",
    \"WANDB_BASE_URL\": \"https://meta.wandb.io\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
