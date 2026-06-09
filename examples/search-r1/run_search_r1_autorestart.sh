#!/bin/bash
# Auto-restart wrapper for the Search-R1 run on this shared box.
# WHY: `oomd` (user-session memory-pressure protection) periodically kills the
# raylet / big actors when this user's memory pressure spikes (e.g. when training
# + the 64GB-RAM retriever + a co-tenant load models at once). The run itself is
# healthy; it just gets reaped. With --load/--save checkpointing (save-interval
# 10), each restart resumes from the latest saved iter, so train/loss and
# eval/nq_test keep accumulating across kills instead of starting over.
# The retriever (run_retriever_h100.sh) is a separate process and is NOT touched.

LOG=${SEARCH_R1_LOG:-/home/yichuan/search_r1_train.log}
RUN=/home/yichuan/slime/examples/search-r1/run_qwen2.5_3B_h100.sh

i=0
while true; do
  i=$((i+1))
  echo "===== [autorestart] attempt #$i starting $(date -u +%H:%M:%S) =====" >> "$LOG"
  bash "$RUN" >> "$LOG" 2>&1
  code=$?
  echo "===== [autorestart] attempt #$i exited (code=$code) $(date -u +%H:%M:%S) =====" >> "$LOG"
  # Completion check: slime prints neither "Training finished" nor a single-line
  # "train end + step N" (those are separate log lines), so the old grep could
  # never match -> the wrapper would restart forever after the run completed.
  # Instead: the run is done when the latest logged training step reached
  # num_rollout-1 (parsed from the run script). Then stop looping.
  num_rollout=$(grep -oE -- '--num-rollout[ =]+[0-9]+' "$RUN" | head -1 | grep -oE '[0-9]+')
  last_step=$(grep -aoE 'model\.py:818 - step [0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+$')
  if [ -n "$num_rollout" ] && [ -n "$last_step" ] && [ "$last_step" -ge "$((num_rollout - 1))" ]; then
    echo "===== [autorestart] reached step $last_step / num_rollout $num_rollout; training complete, stopping wrapper =====" >> "$LOG"
    break
  fi
  echo "===== [autorestart] resume in 25s =====" >> "$LOG"
  sleep 25
done
