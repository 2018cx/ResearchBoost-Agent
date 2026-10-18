#!/bin/bash
# bench wrapper for the interactive workspace (run on the host, outside the sandbox). Usage:
#   SF_GPUS=0,1 ./b.sh time    --tree sf --config runs/<x>.json --out runs/<x>/time
#   SF_GPUS=0,1 ./b.sh profile --tree sf --config runs/<x>.json --out runs/<x>/profile
#   SF_GPUS=0,1 ./b.sh check   --tree sf --config runs/<x>.json --out runs/<x>/check [--quick|--skip-timing]
# Variables: ASSETS (env with the self_forcing Python, HF and torch caches), optional NSYS_ENV (script that puts
# nsys on PATH), SF_GPU_LOCK (lock shared with whoever else uses the same GPUs; default one per GPU pair).
cd "$(dirname "$0")"
GPUS=${SF_GPUS:-0,1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS SF_GPU_LOCK=${SF_GPU_LOCK:-/tmp/sf_gpu_${GPUS/,/}.lock}
A=${ASSETS:?set ASSETS}
export HF_HOME=$A/hf TORCH_HOME=$A/torch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
unset WANDB_API_KEY
[ -n "${NSYS_ENV:-}" ] && source "$NSYS_ENV"
exec $A/env/bin/python bench/bench.py "$@"
