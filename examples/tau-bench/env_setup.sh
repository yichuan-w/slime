# Source this to activate the slime training env (self-owned uv venv + lib fixes).
# Built from scratch via build_venv_uv.sh (torch 2.11+cu129 / sglang 0.5.12.post1 /
# TE 2.10 / apex / flash-attn2 sm90 / megatron). Old shared env was the ProRL venv
# at /home/yichuan/ProRL-Agent-Server/.venv (kept as a fallback, no longer used).
export SLIME_VENV=/home/yichuan/slime/.venv
_NVID=$SLIME_VENV/lib/python3.12/site-packages/nvidia
mkdir -p /home/yichuan/slime_extra_libs
ln -sf /lib64/libnuma.so.1 /home/yichuan/slime_extra_libs/libnuma.so.1 2>/dev/null
export LD_LIBRARY_PATH="$(ls -d $_NVID/*/lib 2>/dev/null | tr '\n' ':')$SLIME_VENV/lib/python3.12/site-packages/sgl_kernel:/home/yichuan/slime_extra_libs:$LD_LIBRARY_PATH"
# Match torch's cu129: sglang JIT kernels (qk-norm rope) emit PTX 8.8, which the
# default /usr/local/cuda (12.8) ptxas rejects. Force the 12.9 toolkit.
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$SLIME_VENV/bin:$CUDA_HOME/bin:$PATH"
# This box's default route is IPv6, so slime auto-detects an IPv6 host and the
# sglang health-probe URL (built after brackets are stripped) becomes malformed,
# hanging server startup forever. Single-node colocate -> force IPv4 loopback.
export SLIME_HOST_IP=127.0.0.1
# SLIME_NODE_IP controls the per-engine sglang bind host + megatron master addr.
export SLIME_NODE_IP=127.0.0.1
# eth0 here is IPv6-only; the only IPv4 is loopback. gloo otherwise auto-picks
# eth0 and can't reach the 127.0.0.1 peers -> weight-sync barrier times out.
# Single node -> force gloo over loopback.
export GLOO_SOCKET_IFNAME=lo
# MetaGen user-sim key: read from a protected local file if not already exported
# (the key is not committed; it lives in ~/.llama_api_key, chmod 600).
if [ -z "${LLAMA_API_KEY:-}" ] && [ -f "$HOME/.llama_api_key" ]; then
  export LLAMA_API_KEY="$(cat "$HOME/.llama_api_key")"
fi
