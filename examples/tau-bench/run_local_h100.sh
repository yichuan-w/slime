#!/bin/bash
# Box-local launcher for tau-bench retail GRPO on this 8xH100 dev box.
# Derived from run_qwen3_4B_h100.sh (the committed upstream recipe) with only
# the machine-specific wiring needed here:
#   - runs in the ProRL uv venv (torch 2.11 / sglang 0.5.10 / megatron / sgl_kernel)
#   - LD_LIBRARY_PATH for bundled cudnn + sgl_kernel + libnuma, propagated to
#     ray workers via the runtime-env JSON
#   - BASE=/home/yichuan (no /root access)
#   - --no-gradient-accumulation-fusion (APEX not built in this venv)
#   - --attention-backend fused (this venv's flash-attn is a broken FA4 namespace
#     pkg TE can't use; "flash" makes megatron disable cuDNN-fused+unfused and
#     leaves only flash -> NoBackend on the thd log-prob forward. cuDNN
#     FusedAttention handles thd+padding_causal+head_dim128 fine.)
#   - LLAMA_API_KEY taken from the shell env (export it before running)

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/env_setup.sh"

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
export LLAMA_API_KEY="${LLAMA_API_KEY:?export LLAMA_API_KEY before running}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"

BASE=${BASE:-/home/yichuan}

CKPT_ARGS=(
   --hf-checkpoint ${BASE}/Qwen3-4B-Instruct-2507/
   --ref-load ${BASE}/Qwen3-4B-Instruct-2507_torch_dist/
   --load ${BASE}/Qwen3-4B-Instruct-2507_slime/
   --save ${BASE}/Qwen3-4B-Instruct-2507_slime/
   --save-interval 10
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
   # 4096 not 8192: trajectories lengthen as the model trains (~4400->5300+ tok),
   # and the box is shared (another user can grab tens of GB on the train GPUs).
   # 8192-token microbatches peaked ~90GB and OOM'd when a co-tenant took 36GB.
   # 4096 halves the activation/logits peak (~45-50GB) to coexist safely.
   --max-tokens-per-gpu 4096
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
   # Shared box: a co-tenant can hold ~36GB on some GPUs, and ray may place an
   # sglang engine there. 0.85 (~80GB) would OOM at startup on a contended GPU.
   # 0.55 (~52GB) fits alongside a co-tenant; KV was only ~25% used at 0.85 so
   # there is ample headroom.
   --sglang-mem-fraction-static 0.55
   # Overlap scheduler can deadlock the /pause_generation control msg used by the
   # post-step weight sync (engine hangs waiting on the HTTP response). Disable it.
   --sglang-disable-overlap-schedule
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend fused
   --no-gradient-accumulation-fusion
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_tau.generate
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

NUM_GPUS=8

# Export every var the driver AND worker actors need BEFORE `ray start`, so the
# raylet (and thus all workers) inherit them. We then run train.py directly as a
# ray driver instead of via `ray job submit`. On this box the dashboard JobHead
# wedges and `ray job submit` times out with HTTP 504 (even for a trivial job),
# so the direct-driver path is the robust way to launch.
export PYTHONPATH="${BASE}/Megatron-LM/:${SCRIPT_DIR}:${PYTHONPATH}"
export CUDA_HOME="${CUDA_HOME}"
export SLIME_HOST_IP="${SLIME_HOST_IP:-127.0.0.1}"
export SLIME_NODE_IP="${SLIME_NODE_IP:-127.0.0.1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export LLAMA_API_KEY="${LLAMA_API_KEY}"
export TAU_USER_MODEL="${TAU_USER_MODEL:-gemini-3-flash-preview-fair}"
export TAU_USER_BASE_URL="${TAU_USER_BASE_URL:-https://api.llama.com/compat/v1}"
# user-sim concurrency: rollout is ~95% MetaGen network wait. At 16 the sglang KV
# pool was only ~13% used, so we push higher to overlap more trajectories. MetaGen
# 429s are auto-retried with backoff, so over-shooting just self-throttles.
export TAU_USER_MAX_CONCURRENCY="${TAU_USER_MAX_CONCURRENCY:-64}"
export WANDB_BASE_URL="https://meta.wandb.io"

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 \
   --temp-dir ${RAY_TEMP_DIR:-${BASE}/shared/ray_temp}

# NON-COLOCATE: dedicate 4 GPUs to training, 4 to rollout. Colocate (sharing all
# 8 via torch_memory_saver pause/resume) reliably OOMs at the train step on this
# box — sglang's paused VMM reservation isn't reusable, so the trainer's
# cu_mem_create fails even with ~76GB "free". Dedicated GPUs avoid the swap.
RAY_ADDRESS="auto" python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
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
