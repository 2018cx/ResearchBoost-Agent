#!/bin/bash
# Inside the sandbox (PREP=1): baseline timing, then the reference trajectories of the statistical gate.
set -uo pipefail
G=${WORK:?set WORK}/self-forcing; W=$G/ws; P=$G/prep
cd $W
export PYTHONPATH=$W
date; echo "== baseline timing"
echo baseline already done
python - <<PY > $W/bench/traj/ref_cases.json
import json
spec = json.load(open('bench/traj/spec.json'))
print(json.dumps([dict(spec['case'], name=f'traj_s{s}', seed=s) for s in range(1000, 1006)], indent=1))
PY
date; echo "== reference trajectories"
flock /tmp/sf_gpu.lock torchrun --standalone --nnodes=1 --nproc-per-node=2 -m numerics record \
  --adapter bench.sf_adapter:SelfForcingAdapter \
  --adapter-kwargs "{\"repo\": \"$P/reftree\", \"config\": \"$W/bench/cfg/numerics.yaml\"}" \
  --cases bench/traj/ref_cases.json --losses-only --math-mode default --deterministic --out $W/bench/traj/ref \
  > $P/traj_ref.log 2>&1; echo "rc=$?"
date
python -m numerics traj-freeze --ref $W/bench/traj/ref --spec bench/traj/spec.json --k 2 --out bench/traj/frozen.json
echo PREP_TRAJ_DONE
