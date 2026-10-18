#!/bin/bash
# Record the reference trajectories for the frozen seeds and freeze delta = 2 x the seed sigma.
set -euo pipefail
cd ${WORK:?set WORK}/dmd/ws
SEEDS=${SEEDS:-10,11,12,13,14,15}
python bench/bench.py record --tree targets/reference --out bench/traj/ref --traj-seeds $SEEDS
python -m numerics traj-freeze --ref bench/traj/ref --spec bench/traj/spec.json --k 2 --out bench/traj/frozen.json
python - <<'PY'
import json
fr = json.load(open('bench/traj/frozen.json'))
print(json.dumps({'seeds': fr['reference_seeds'],
                  'endpoints': {n: {'mean': round(e['reference_mean'], 6), 'sigma': round(e['sigma_seed'], 6),
                                    'delta': round(e['delta'], 6)} for n, e in fr['endpoints'].items()}}, indent=1))
PY
echo TRAJ_FROZEN
