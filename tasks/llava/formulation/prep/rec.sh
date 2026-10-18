#!/bin/bash
# Formulation helper (run inside the sandbox, PREP=1): numerics record of one case file with the llava adapter.
#   rec.sh <out dir> <cases json> <adapter kwargs json> [extra recorder args...]
cd ${WORK:?set WORK}/llava/ws
OUT=$1; CASES=$2; KW=$3; shift 3
P=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
exec flock /tmp/llava_gpu.lock env PYTHONPATH=$PWD torchrun --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=$P --local-addr=127.0.0.1 -m numerics record \
  --adapter bench.llava_adapter:LlavaAdapter --adapter-kwargs "$KW" --cases "$CASES" --math-mode tf32 "$@" --out "$OUT"
