#!/usr/bin/env bash
set -eu
cd /home/gongji/0z5a/work/llm-serving-simulator-20260929
export CUDA_VISIBLE_DEVICES=GPU-739fcff8-0a90-fd63-8d91-65847ad665d2
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="$PWD/cache/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$PWD/cache/vllm"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor"
export CUDA_CACHE_PATH="$PWD/cache/cuda"
export OMP_NUM_THREADS=4
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
if [ "${SIM_ENABLE_LOADER:-0}" = 1 ]; then
    export LD_AUDIT="$PWD/build/cuda_loader_audit.so"
fi
if [ "${SIM_ENABLE_CUPTI:-0}" = 1 ]; then
    export LD_PRELOAD="$PWD/build/cupti_api_trace.so"
fi
exec /home/gongji/0z5a/bin/python -u tools/baseline.py \
    --model "${MODEL_DIR:-$PWD/models/Qwen2.5-0.5B}" "$@"
