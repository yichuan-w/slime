#!/bin/bash
# Box-local launcher for the Search-R1 local dense retrieval server on this box.
#   - runs in a dedicated uv venv (/home/yichuan/retriever-venv: torch cu13 +
#     faiss-gpu-cu12 + transformers), separate from the slime training venv.
#   - CPU faiss (no --faiss_gpu): the e5_Flat index is ~64GB; the box has ~2TB
#     RAM so the index lives in RAM and does NOT eat a training GPU. Only the
#     small e5-base-v2 encoder runs on GPU (model.cuda() is hardcoded upstream).
#   - encoder pinned to GPU 0 (a trainer GPU, idle during rollout); ~1-2GB.
# Listens on http://127.0.0.1:8000/retrieve (matches SEARCH_R1_CONFIGS in
# generate_with_search.py). Start this BEFORE the training run.

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source /home/yichuan/retriever-venv/bin/activate

export CUDA_VISIBLE_DEVICES=${RETRIEVER_GPU:-0}

SAVE=/home/yichuan/search-r1-index
INDEX_FILE=${SAVE}/e5_Flat.index
CORPUS_FILE=${SAVE}/wiki-18.jsonl
RETRIEVER_NAME=e5
RETRIEVER_PATH=intfloat/e5-base-v2

python "${SCRIPT_DIR}/local_dense_retriever/retrieval_server.py" \
    --index_path "${INDEX_FILE}" \
    --corpus_path "${CORPUS_FILE}" \
    --topk 3 \
    --retriever_name "${RETRIEVER_NAME}" \
    --retriever_model "${RETRIEVER_PATH}"
