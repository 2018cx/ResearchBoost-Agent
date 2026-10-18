#!/bin/bash
# Formulation driver (run inside the sandbox with PREP=1, which makes ws/bench ws/task writable).
#   formulate.sh records     reference records of both cases -> bench/records/ref, plus a duplicate -> /tmp/rec_dup
#   formulate.sh variant     a legitimate variant (sdpa attention) -> /tmp/rec_sdpa
#   formulate.sh timing      two reference timing runs -> task/timing (the second freezes reference.json)
#   formulate.sh traj N...   reference trajectories for these seeds -> bench/traj/ref
set -uo pipefail
W=${WORK:?set WORK}/llava/ws
P=$(cd "$(dirname "$0")" && pwd)
cd $W
export PYTHONPATH=$W
mode=$1; shift
port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
rec() {  # rec <out> <cases file> [extra...]
  local out=$1 cases=$2; shift 2
  flock /tmp/llava_gpu.lock env PYTHONPATH=$W torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 \
    --master-port=$(port) --local-addr=127.0.0.1 -m numerics record --adapter bench.llava_adapter:LlavaAdapter \
    --cases "$cases" --math-mode tf32 --deterministic --out "$out" "$@"
}
case_file() { python3 - "$1" <<'PY'
import json, sys
cases = json.load(open('bench/cases.json'))
name = sys.argv[1]
case = next(c for c in cases if c['name'] == name)
path = f'/tmp/case_{name}.json'
json.dump([case], open(path, 'w'))
print(path)
PY
}

if [ "$mode" = records ]; then
  for c in mm text; do
    f=$(case_file $c)
    echo "== ref record $c $(date +%T)"
    rec $W/bench/records/ref "$f" --sketch-dim 4096 --adapter-kwargs '{"repo": "targets/reference"}' || exit 1
    echo "== dup record $c $(date +%T)"
    rec /tmp/rec_dup "$f" --sketch-dim 4096 --adapter-kwargs '{"repo": "targets/reference"}' || exit 1
  done
  echo "== compare ref vs dup (must be bitwise identical) $(date +%T)"
  python3 $P/cmp_records.py $W/bench/records/ref /tmp/rec_dup
elif [ "$mode" = variant ]; then
  for c in mm text; do
    f=$(case_file $c)
    echo "== variant record $c $(date +%T)"
    rec /tmp/rec_sdpa "$f" --sketch-dim 4096 \
      --adapter-kwargs '{"repo": "targets/reference", "variant": "sdpa"}' || exit 1
  done
elif [ "$mode" = timing ]; then
  echo "== reference timing a (freezes task/timing/reference.json) $(date +%T)"
  python3 bench/bench.py time --tree targets/reference --out task/timing --label ref_a --freeze || exit 1
  echo "== reference timing b (noise floor against the frozen run) $(date +%T)"
  python3 bench/bench.py time --tree targets/reference --out task/timing --label ref_b || exit 1
elif [ "$mode" = calibrate ]; then
  echo "== calibrate diagnostic gates from the sdpa variant $(date +%T)"
  python3 -m numerics calibrate --ref bench/records/ref --variants /tmp/rec_sdpa --out bench/thresholds.json || exit 1
  echo "== selftest (every mutation must be caught) $(date +%T)"
  python3 -m numerics selftest --ref bench/records/ref --variant /tmp/rec_sdpa \
    --thresholds bench/thresholds.json --out task/selftest.json || exit 1
  # No leave-one-out here: it needs several legitimate variants, and in this task the per-tensor gates are
  # diagnostics only (the verdict is bitwise -> statistical), so a one-variant LOO would only measure itself.
elif [ "$mode" = freeze_traj ]; then
  echo "== freeze delta from the reference seeds $(date +%T)"
  python3 -m numerics traj-freeze --ref bench/traj/ref --spec bench/traj/spec.json --k 2 \
    --out bench/traj/frozen.json || exit 1
elif [ "$mode" = profile ]; then
  echo "== reference profile (nsys) $(date +%T)"
  python3 bench/bench.py profile --tree targets/reference --out task/profile --label reference || exit 1
  echo "== reference profile (detail) $(date +%T)"
  python3 bench/bench.py profile --tree targets/reference --out task/profile --label reference_detail --detail || exit 1
elif [ "$mode" = traj ]; then
  for s in "$@"; do
    echo "== reference trajectory seed $s $(date +%T)"
    python3 - "$s" <<'PY'
import json, sys
spec = json.load(open('bench/traj/spec.json'))
seed = int(sys.argv[1])
json.dump([dict(spec['case'], name=f'traj_s{seed}', seed=seed)], open(f'/tmp/traj_case_{seed}.json', 'w'))
PY
    rec $W/bench/traj/ref /tmp/traj_case_$s.json --losses-only \
      --adapter-kwargs '{"repo": "targets/reference"}' || exit 1
  done
else
  echo "unknown mode $mode"; exit 2
fi
echo "== DONE $mode $(date +%T)"
