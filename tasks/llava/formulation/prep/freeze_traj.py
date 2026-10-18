#!/usr/bin/env python3
"""Formulation side: freeze delta from the re-recorded reference trajectories and re-pin the digests.

  freeze_traj.py <new traj ref dir>

Prints the old and new per-endpoint reference_mean / sigma_seed / delta side by side. For the reference the
canonical training loss is identical by construction to the number the old endpoint read (its training_step
returns c_mb/accum, so the old sum over the 2 micro-steps is the mean the bench now computes), so a delta
that moves materially would mean the extra bench forward perturbed the run -- worth seeing explicitly.
"""
import os
import importlib.util
import json
import shutil
import sys
from pathlib import Path

WS = Path(os.environ['WORK']) / 'llava' / 'ws'
new = Path(sys.argv[1])
sys.path.insert(0, str(WS))
from numerics.trajectory import freeze                                          # noqa: E402

spec = json.loads((WS / 'bench' / 'traj' / 'spec.json').read_text())
old = json.loads((WS / 'bench' / 'traj' / 'frozen.json').read_text())
fr = freeze(new, spec, k=old.get('k', 2.0))

print(f"seeds: {fr['reference_seeds']}  k={fr['k']}")
print(f"{'endpoint':34s} {'reference_mean':>16s} {'sigma_seed':>12s} {'delta':>12s}")
for name, e in old['endpoints'].items():
    print(f"  OLD {name:28s} {e['reference_mean']:16.9f} {e['sigma_seed']:12.8f} {e['delta']:12.8f}")
for name, e in fr['endpoints'].items():
    print(f"  NEW {name:28s} {e['reference_mean']:16.9f} {e['sigma_seed']:12.8f} {e['delta']:12.8f}")

# install
ref_dir = WS / 'bench' / 'traj' / 'ref'
shutil.rmtree(ref_dir, ignore_errors=True)
shutil.copytree(new, ref_dir)
(WS / 'bench' / 'traj' / 'frozen.json').write_text(json.dumps(fr, indent=1) + '\n')

spec_mod = importlib.util.spec_from_file_location('b', WS / 'bench' / 'bench.py')
m = importlib.util.module_from_spec(spec_mod)
sys.modules['b'] = m
try:
    spec_mod.loader.exec_module(m)
except SystemExit:
    pass
form = WS / 'task' / 'formulation.json'
d = json.loads(form.read_text())
fi = d['frozen_inputs']
for name, rel in (('traj_frozen', 'bench/traj/frozen.json'), ('traj_spec', 'bench/traj/spec.json')):
    before = fi[name]['sha256']
    fi[name] = {'path': rel, **m.content_digest(WS / rel)}
    print(f'{name}: {before[:16]} -> {fi[name]["sha256"][:16]}')
form.write_text(json.dumps(d, indent=1, ensure_ascii=False) + '\n')
print('frozen inputs:', len(fi))
print('evaluator_sha256:', m.evaluator_sha256())
res = m.verify_inputs('after traj re-freeze')
print(json.dumps({k: res[k] for k in ('ok', 'checked', 'mismatches')}))
sys.exit(0 if res['ok'] else 1)
