#!/usr/bin/env python3
"""Inside the sandbox (PREP=1): re-record the reference trajectories through bench.py's own path.

  record_traj.py <out dir> <seed> [seed ...]

Goes through bench.trajectory_seed -> run_gpu, so it holds the GPU lock and gets the environment a real
check gets. bench/ is read-only in the sandbox, so the records go to /tmp and are installed from the host.
"""
import os
import importlib.util
import sys
from pathlib import Path

WS = Path(os.environ['WORK']) / 'llava' / 'ws'
spec = importlib.util.spec_from_file_location('b', WS / 'bench' / 'bench.py')
m = importlib.util.module_from_spec(spec)
sys.modules['b'] = m
try:
    spec.loader.exec_module(m)
except SystemExit:
    pass

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
logs = out / 'logs'
logs.mkdir(exist_ok=True)
env = m.base_env({'PYTHONPATH': str(m.ROOT)})
for seed in [int(x) for x in sys.argv[2:]]:
    r = m.trajectory_seed('targets/reference', out, seed, env, logs / f'traj_s{seed}.log')
    print(f'seed {seed}: {r.get("status")} rc={r.get("returncode")} {str(r.get("reason") or "")[:120]}', flush=True)
    if r['status'] != 'ok' or r.get('returncode'):
        sys.exit(1)
print('TRAJ_RECORD_DONE')
