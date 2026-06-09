#!/bin/bash
# Box-local launcher for Search-R1 (Qwen2.5-3B GRPO + local search) on this
# IPv6-only 8xH100 dev box, using the FIRST FOUR GPUs (0-3).
# Derived from run_qwen2.5_3B.sh (upstream recipe) + the machine-specific wiring
# proven out for tau-bench (see examples/tau-bench/run_local_h100.sh):
#   - runs in the slime uv venv via env_setup.sh (LD_LIBRARY_PATH, CUDA 12.9,
#     SLIME_HOST_IP/SLIME_NODE_IP=127.0.0.1, GLOO_SOCKET_IFNAME=lo for IPv6 box)
#   - BASE=/home/yichuan (no /root access)
#   - 4 GPUs, NON-COLOCATE (2 train + 2 rollout): colocate reliably OOMs the
#     train step on this box (sglang's paused VMM reservation isn't reusable).
#   - --attention-backend fused: this venv's flash-attn is an FA4 namespace pkg
#     TE can't use; "flash" -> NoBackend on the thd log-prob forward.
#   - --no-gradient-accumulation-fusion: APEX fused path not reliable here.
#   - --sglang-disable-overlap-schedule: overlap deadlocks /pause_generation
#     used by the post-step weight sync.
#   - bypass `ray job submit` (dashboard JobHead 504s here) -> direct ray driver.
# PREREQ: start the retrieval server first:  bash run_retriever_h100.sh

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/../tau-bench/env_setup.sh"

# SHARED BOX: another run (e.g. tau-bench on GPUs 4-7, ray port 6390) may be live.
# Do NOT use global `ray stop --force` / `pkill python|ray|train.py` — that would
# kill the other run AND the local retrieval server (a python process). Clean up
# ONLY this run's own ray session, identified by its unique temp-dir string.
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/home/yichuan/shared/ray_search_r1_v2}
for p in $(pgrep -f "$(basename ${RAY_TEMP_DIR})" 2>/dev/null); do kill -9 $p 2>/dev/null || true; done
sleep 3

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-3B.sh"

BASE=${BASE:-/home/yichuan}

CKPT_ARGS=(
   --hf-checkpoint ${BASE}/Qwen2.5-3B/
   --ref-load ${BASE}/Qwen2.5-3B_torch_dist/
   # Checkpoint so the run survives oomd kills / raylet death: on restart slime
   # resumes from the latest saved iter in --load (falls back to ref-load if empty).
   --load ${BASE}/Qwen2.5-3B_search_r1_stopfix_ckpt/
   --save ${BASE}/Qwen2.5-3B_search_r1_stopfix_ckpt/
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data ${BASE}/Search-R1/data/nq_hotpotqa_train/train.parquet
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 512
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   # eval every 5 steps on the first 500 nq test prompts (EM reward). The eval
   # rollout also uses the local retriever. ~1-2 min per eval; gives ~enough
   # points to watch eval/nq_test improve within the first ~1h.
   --eval-interval 5
   --eval-prompt-data nq_test ${BASE}/Search-R1/data/nq_hotpotqa_train/test.parquet@[0:500]
   --eval-input-key prompt
   --eval-label-key reward_model
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 512
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
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-search-r1
   --wandb-group search-r1_qwen2.5-3B-h100-stopfix
   --wandb-host https://meta.wandb.io
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-disable-overlap-schedule
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend fused
   --no-gradient-accumulation-fusion
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Use the FIRST FOUR GPUs only (tau-bench / other runs use 4-7). ray then sees
# them as 0-3.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_GPUS=4

# OWN ray cluster on a UNIQUE port + temp dir, so we never attach to another
# run's cluster (tau-bench uses 6390). The crash mode this avoids: RAY_ADDRESS
# "auto" connected the driver to the tau-bench head on :6390, whose PYTHONPATH
# pointed at examples/tau-bench -> the rollout actor couldn't import
# generate_with_search. Pin our own port and drive train.py against it explicitly.
RAY_PORT=${RAY_PORT:-6391}
RAY_DASH_PORT=${RAY_DASH_PORT:-8267}

# Export every var the driver AND worker actors need BEFORE `ray start`, so the
# raylet (and thus all workers) inherit them. Run train.py directly as a ray
# driver instead of `ray job submit` (dashboard JobHead 504s on this box).
export PYTHONPATH="${BASE}/Megatron-LM/:${SCRIPT_DIR}:${PYTHONPATH}"
export CUDA_HOME="${CUDA_HOME}"
export SLIME_HOST_IP="${SLIME_HOST_IP:-127.0.0.1}"
export SLIME_NODE_IP="${SLIME_NODE_IP:-127.0.0.1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export WANDB_BASE_URL="https://meta.wandb.io"

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --port ${RAY_PORT} --disable-usage-stats \
   --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASH_PORT} \
   --temp-dir ${RAY_TEMP_DIR}

# NON-COLOCATE: 2 GPUs train (actor TP=2), 2 GPUs rollout (sglang TP=2).
# Drive train.py against OUR ray head explicitly (not "auto").
RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}" python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
