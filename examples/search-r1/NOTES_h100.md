# Search-R1 on this 8×H100 box — quick notes

Box-local recipe + the things that bit us. For the upstream tutorial see `README.md`.

## What it is
Qwen2.5-3B (base) trained with **GRPO** to interleave reasoning + **search** over a
local wiki-18 dense retriever (e5 + faiss), reward = answer **Exact Match** + format.

```
question ─► generate() multi-turn ─► trajectory ─► reward_func(EM) ─► GRPO
                  └─ <search>q</search> ─► retrieval_server(faiss+e5) ─► <information>…</information>
```

## Custom code (what slime actually loads)
| file | role | wired by |
|---|---|---|
| `generate_with_search.py` | multi-turn rollout `generate` + `reward_func` | `--custom-generate-function-path` / `--custom-rm-path` |
| `qa_em_format.py` | EM scoring + format state-machine | imported |
| `local_search_server.py` | async client → retrieval server | imported in `search()` |
| `local_dense_retriever/retrieval_server.py` | faiss + e5 HTTP server (separate venv/proc) | `run_retriever_h100.sh` |

Key detail: model tokens get `loss_mask=1`, retrieved `<information>` tokens get
`loss_mask=0` (not trained on). `return_logprob=True` collects sglang logprobs (for TIS).

## How to run (this box)
```bash
# 1) retriever (separate uv venv, CPU faiss, e5 encoder on GPU0). Listens :8000
bash examples/search-r1/run_retriever_h100.sh        # ~5min to load 64GB index

# 2) training, auto-restart + checkpoint (survives oomd kills)
SEARCH_R1_LOG=/home/yichuan/search_r1_stopfix.log \
  setsid nohup bash examples/search-r1/run_search_r1_autorestart.sh &
```
- `run_qwen2.5_3B_h100.sh` = box-local launcher (derived from `run_qwen2.5_3B.sh`).
- GPUs **0-3** (trainer 0,1 + sglang 2,3), non-colocate. Retriever encoder shares GPU0 (~1GB).
- Artifacts: HF `~/Qwen2.5-3B`, mcore `~/Qwen2.5-3B_torch_dist`, data `~/Search-R1/data/...`,
  index `~/search-r1-index/`, ckpt `~/Qwen2.5-3B_search_r1*_ckpt/`.

## Gotchas that cost us a session
1. **Shared box — never `ray stop --force` / `pkill ray|python`** (global, kills other
   users' runs + the retriever). Use a unique ray **port 6391** + temp dir
   `ray_search_r1_v2`, explicit `RAY_ADDRESS`, and clean up only by temp-dir/PID.
   (Another user runs tau-bench on GPUs 4-7, ray port 6390.)
2. **oomd kills the run** under user memory-pressure (`oomd ... user session protection`),
   not OOM-by-bytes. The retriever's 64GB-RAM index + training spikes trigger it →
   raylet dies → GCS unreachable. Fix = `run_search_r1_autorestart.sh` (checkpoint
   every 10 steps + auto-resume).
3. **IPv6-only box**: env in `../tau-bench/env_setup.sh` (`SLIME_*_IP=127.0.0.1`,
   `GLOO_SOCKET_IFNAME=lo`, CUDA 12.9, LD_LIBRARY_PATH). `--attention-backend fused`
   (this venv's flash is FA4), `--no-gradient-accumulation-fusion`,
   `--sglang-disable-overlap-schedule`.
4. **`ray job submit` 504s** here → run `train.py` directly as a ray driver.

## Bug fixed: generation didn't stop at `</search>`/`</answer>`
Upstream relied on `postprocess_responses` (string-trim) which is **disabled when
`return_logprob=True`** → the model kept emitting junk after the closing tag (even fake
new `Question:`s), which got trained on (`loss_mask=1`) and broke `is_valid_sequence`.
**Fix** (in `generate_with_search.py`): pass `stop=["</search>","</answer>"]` to the
engine (slime already sets `no_stop_trim=True`) → halts at the tag, token/logp stay
aligned. wandb group `…-stopfix` is the fixed run.

## Algorithm / key config
GRPO: advantage = group-normalized reward over `n_samples_per_prompt=8` (no critic);
PPO clipped update with asymmetric clip `--eps-clip 0.2 --eps-clip-high 0.28`;
KL-to-ref regularizer `--kl-loss-coef 0.001 --kl-loss-type low_var_kl` (k3 estimator);
`--entropy-coef 0`. Synchronous on-policy. **TIS off** (no `--use-tis`); not using
dynamic sampling (no `--dynamic-sampling-filter-path`).

Two KLs (don't confuse): `train/kl_loss` = KL(π_θ‖π_ref) rises as the model learns
(normal); `train/ppo_kl` = log(π_old/π_θ) inside the PPO clip ≈ 0 in sync on-policy.

## Perf (measured)
~70s/step normal: **rollout ~87% (bottleneck) / train ~13%** (`wait_time_ratio≈0.87`).
Train = actor ~7s + ref_logprobs ~2s + weight-sync ~0.35s. Eval steps (every 5) spike
to ~180-260s (extra 500-prompt eval rollout).

## Monitoring
wandb project `slime-search-r1` (host `https://meta.wandb.io`). Watch `train/loss`,
`train/grad_norm`, `train/kl_loss`, and **`eval/nq_test`** (EM).

## Sanity vs Search-R1 paper (arXiv 2503.09516)
Converged NQ EM for Qwen2.5-3B-Base + **GRPO** ≈ **0.421** (Table 3; PPO 0.406, Table 2).
Paper trains **500 steps**. The paper's training-curve figure (Fig. 2) plots *reward*,
not per-step EM — so intermediate "EM @ step N" expectations are estimates, not paper data.
Our climb 0.02 → ~0.15 by step ~100 is on-trajectory.
