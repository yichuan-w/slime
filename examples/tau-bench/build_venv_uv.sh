#!/bin/bash
# Build a standalone slime training venv with uv (no conda), adapted from
# slime's build_conda.sh. Uses the system CUDA 12.9 toolkit instead of the
# conda-provided one. Targets H100 (sm90) only to cut source-build time.
#
# Result: /home/yichuan/slime/.venv  -- fully independent of ProRL's venv.
set -ex

export BASE_DIR=${BASE_DIR:-/home/yichuan}
export SLIME_DIR=${SLIME_DIR:-/home/yichuan/slime}
export VENV=${VENV:-$SLIME_DIR/.venv}

# Keep in sync with build_conda.sh / docker/Dockerfile.
export SGLANG_COMMIT="5a15cde858ea09b77116212a39356f2fc51b8584"   # v0.5.12.post1
export MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
export PATCH_VERSION="latest"

# System CUDA (matches torch cu129 wheels). build_conda.sh uses conda's CUDA.
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
# Only build kernels for H100 -> much faster apex/TE compiles.
export TORCH_CUDA_ARCH_LIST="9.0"
# flash-attn's setup.py ignores TORCH_CUDA_ARCH_LIST; it has its own knob and
# defaults to building 80;90;100;120 (4x the work). Limit to sm90 (H100).
export FLASH_ATTN_CUDA_ARCHS="90"
# Build temp dirs MUST live on persistent disk, not /tmp: the source builds
# (flash-attn especially) can run >90min, and this box's /tmp cleaner deletes
# untouched files mid-build -> "flash_bwd_*.cu: No such file or directory".
export TMPDIR=/home/yichuan/build_tmp
mkdir -p "$TMPDIR"
# This box's python is the fbcode meta build (3.12.13+meta); its sysconfig CXX is
# `clang++.par --platform platform010`, which TE/apex extension builds would pick
# up. Force gcc/g++ to match build_conda.sh and the cu129 torch wheels' ABI.
export CC=gcc CXX=g++
# TE/apex need headers (cudnn.h, nccl.h, cublas, ...) + libs from the pip
# nvidia-*-cu12 wheels, which live under site-packages/nvidia/*/{include,lib}
# rather than a system/conda prefix. Put every cu12 include/lib dir on the
# compiler/linker search path (skip the empty cu13 placeholder dir).
# NOTE: this runs after sglang install populates site-packages/nvidia, so the
# build script re-exports these right before the TE step too (see below).
_NV=$VENV/lib/python3.12/site-packages/nvidia
export CUDNN_PATH="$_NV/cudnn" CUDNN_HOME="$_NV/cudnn"

# sglang's editable build compiles a Rust gRPC extension (sglang-grpc) that needs
# protoc. The slimerl docker base ships it; this box doesn't, so provide a local
# protoc binary (downloaded to slime_extra_libs) and point prost-build at it.
export PROTOC=/home/yichuan/slime_extra_libs/protoc/bin/protoc
export PATH="/home/yichuan/slime_extra_libs/protoc/bin:$PATH"

# ---- create the venv (uv) ----
# Create with uv, but install packages with the venv's own pip so we reproduce
# build_conda.sh's (pip-validated) recipe verbatim. uv's resolver is stricter
# (single-index by default) and chokes on the multi-index cu129 dance.
uv venv "$VENV" --python 3.12 --clear
export VIRTUAL_ENV="$VENV"
uv pip install pip setuptools wheel
PY="$VENV/bin/python"
PIP() { "$PY" -m pip install "$@"; }

PIP cuda-python==12.9

# ---- sglang (editable from git, cu129 native kernels) ----
if [ ! -d "$BASE_DIR/sglang" ]; then
  git clone https://github.com/sgl-project/sglang.git "$BASE_DIR/sglang"
fi
cd "$BASE_DIR/sglang"
git checkout ${SGLANG_COMMIT}
PIP -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129
# pin torch / sgl kernels to their +cu129 builds (defaults are cu13)
PIP --force-reinstall --no-deps \
  torch==2.11.0 torchvision torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129
PIP --force-reinstall --no-deps \
  sglang-kernel==0.4.2.post2 sgl-deep-gemm==0.1.0 \
  --index-url https://docs.sglang.ai/whl/cu129/
# repair the cu13 spill: drop cu13 nvidia libs, reinstall cu12 equivalents
"$PY" -m pip uninstall -y \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver \
  nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink \
  nvidia-nvshmem-cu13 nvidia-nvtx nvidia-cutlass-dsl-libs-cu13 || true
PIP --force-reinstall --no-deps \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12==9.16.0.29 nvidia-cufft-cu12 \
  nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple

PIP cmake ninja

# ---- flash-attn 2 (max version megatron supports) ----
MAX_JOBS=64 "$PY" -m pip install flash-attn==2.7.4.post1 --no-build-isolation

PIP git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
PIP flash-linear-attention==0.4.1
PIP git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation
PIP tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

# Now that site-packages/nvidia/* is populated (by the sglang install above),
# expose every cu12 include/lib dir so the TE/apex source builds find their headers.
_NVINC=$(ls -d $_NV/*/include 2>/dev/null | grep -v '/cu13/' | tr '\n' ':')
_NVLIB=$(ls -d $_NV/*/lib 2>/dev/null | grep -v '/cu13/' | tr '\n' ':')
export CPATH="${_NVINC}$CUDA_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${_NVLIB%:}${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${_NVLIB%:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---- transformer engine ----
PIP --no-build-isolation "transformer_engine[pytorch]==2.10.0"
# The TE-2.10 torch-extension source build (no prebuilt wheel exists for torch
# 2.11) links libc10/libtorch/libtorch_cpu but NOT libtorch_cuda/libc10_cuda, so
# `import transformer_engine.pytorch` dies on undefined symbol
# c10::cuda::CUDACachingAllocator::allocator. Patch the .so to NEED the cuda libs.
PIP patchelf
_TESO=$(ls "$VENV"/lib/python3.12/site-packages/transformer_engine/wheel_lib/transformer_engine_torch.cpython-*.so 2>/dev/null | head -1)
_TORCHLIB=$("$PY" -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
if [ -n "$_TESO" ]; then
  "$VENV/bin/patchelf" --add-needed libc10_cuda.so "$_TESO"
  "$VENV/bin/patchelf" --add-needed libtorch_cuda.so "$_TESO"
  "$VENV/bin/patchelf" --add-rpath "$_TORCHLIB" "$_TESO"
fi

# ---- apex (cpp + cuda ext) ----
NVCC_APPEND_FLAGS="--threads 4" \
  "$PY" -m pip install --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

# ---- torch_memory_saver (must build native preload hook) ----
TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
PIP -v git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 \
  --no-cache-dir --force-reinstall --no-build-isolation

PIP git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
PIP "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
PIP https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# ---- megatron (editable, pinned commit) ----
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive "$BASE_DIR/Megatron-LM"
fi
PIP "setuptools<80.0.0" pybind11 "packaging>=24.2"
cd "$BASE_DIR/Megatron-LM" && git checkout ${MEGATRON_COMMIT} && PIP -e . --no-build-isolation

# ---- slime (runtime deps, then editable no-deps) ----
cd "$SLIME_DIR"
PIP -r requirements.txt
PIP -e . --no-deps

# ---- int4_qat kernel ----
cd "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat"
PIP . --no-build-isolation

# ---- pins / fixups ----
cd "$SLIME_DIR"
PIP nvidia-cudnn-cu12==9.16.0.29     # pytorch/pytorch#168167
PIP "numpy<2"
PIP "kernels<0.15.0"                 # >=0.15 breaks `import sglang`

# ---- apply slime's sglang/megatron patches ----
# Use `git apply --3way` directly (NOT gated on `git apply --check`): on this
# sglang/megatron checkout --check fails on context drift in a few files, but the
# 3-way merge applies the whole patch cleanly. --3way is idempotent-ish: a second
# run no-ops the already-applied hunks. Only hard conflicts (<<<<) are fatal.
apply_patch() {  # $1 = repo dir, $2 = patch file, $3 = label
  cd "$1"
  if git diff --cached --quiet "$(git rev-parse --show-toplevel)" 2>/dev/null && \
     ! git apply --reverse --check "$2" 2>/dev/null; then
    git apply --3way "$2" || { echo "$3 patch failed" >&2; exit 1; }
  else
    echo "$3 patch already applied, skipping"
  fi
  grep -R -n '^<<<<<<< ' . >/dev/null 2>&1 && { echo "$3 patch conflict" >&2; exit 1; } || true
}
apply_patch "$BASE_DIR/sglang"      "$SLIME_DIR/docker/patch/${PATCH_VERSION}/sglang.patch"   sglang
apply_patch "$BASE_DIR/Megatron-LM" "$SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch" megatron

# ---- tau-bench example package (editable) ----
# The custom rollout (generate_with_tau) imports `tau_bench`. Install it editable
# from the local checkout, then restore the openai/protobuf versions sglang pins
# (litellm drags openai up to 2.41 and protobuf down to 5.x, which break sglang).
if [ -d "$BASE_DIR/tau-bench" ]; then
  PIP -e "$BASE_DIR/tau-bench" --no-deps
  PIP litellm mistralai google-generativeai termcolor tenacity
  PIP --no-deps "openai==2.6.1" "protobuf==6.33.6"
fi

echo "=== DONE. venv at $VENV ==="
