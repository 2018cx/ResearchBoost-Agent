#!/bin/bash
# Inside the sandbox (PREP=1): remaining reference trajectories, one process per seed (a second trainer in
# one process OOMs). Seed 1001 is re-recorded in a fresh process and compared with the copy from the shared process.
set -uo pipefail
G=${WORK:?set WORK}/self-forcing; W=$G/ws; P=$G/prep
cd $W
export PYTHONPATH=$W
mkdir -p $P/traj_shared_process && mv $W/bench/traj/ref/traj_s1001.r*.json $P/traj_shared_process/ 2>/dev/null
for s in 1001 1002 1003 1004 1005; do
  python - <<PY > $P/traj_case_$s.json
import json
spec = json.load(open('bench/traj/spec.json'))
print(json.dumps([dict(spec['case'], name='traj_s$s', seed=$s)]))
PY
  date; echo "== seed $s"
  flock /tmp/sf_gpu.lock torchrun --standalone --nnodes=1 --nproc-per-node=2 -m numerics record \
    --adapter bench.sf_adapter:SelfForcingAdapter \
    --adapter-kwargs "{\"repo\": \"$P/reftree\", \"config\": \"$W/bench/cfg/numerics.yaml\"}" \
    --cases $P/traj_case_$s.json --losses-only --math-mode default --deterministic --out $W/bench/traj/ref \
    > $P/traj_ref_s$s.log 2>&1; echo "rc=$?"
done
date
python - <<PY
import json
for r in (0, 1):
    a = json.load(open('$P/traj_shared_process/traj_s1001.r%d.json' % r))['steps']
    b = json.load(open('$W/bench/traj/ref/traj_s1001.r%d.json' % r))['steps']
    same = sum(x['losses'] == y['losses'] for x, y in zip(a, b))
    print('seed 1001 rank', r, 'identical steps', same, 'of', len(a))
PY
python -m numerics traj-freeze --ref $W/bench/traj/ref --spec bench/traj/spec.json --k 2 --out bench/traj/frozen.json
echo PREP_TRAJ_DONE
