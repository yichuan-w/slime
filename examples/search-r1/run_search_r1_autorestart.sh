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
  echo "===== [autorestart] attempt #$i exited (code=$code) $(date -u +%H:%M:%S); resume in 25s =====" >> "$LOG"
  # If it exited cleanly because num_rollout finished, stop looping.
  if grep -aq "Training finished\|reached num_rollout\|train end.*step 49[0-9]" "$LOG" 2>/dev/null; then
    echo "===== [autorestart] training appears complete; stopping wrapper =====" >> "$LOG"
    break
  fi
  sleep 25
done
