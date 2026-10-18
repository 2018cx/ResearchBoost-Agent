"""Write ws/task/formulation.json: every frozen input of the bench with its fingerprint (sha256 for small files and
the evaluator's own data, size:mtime for the weights; the sandbox mounts all of them read-only)."""
import os
import hashlib, json, sys
from pathlib import Path

WS = Path(os.environ['WORK']) / 'self-forcing' / 'ws'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def stat(p):
    s = p.stat()
    return f'{s.st_size}:{s.st_mtime_ns}'


inputs = {}
for f in sorted((WS / '../assets').resolve().rglob('*')):
    if f.is_file():
        rel = '../assets/' + str(f.relative_to((WS / '../assets').resolve()))
        small = f.stat().st_size < 200 << 20
        inputs[rel] = {'kind': 'sha256' if small else 'stat', 'value': sha(f) if small else stat(f)}
for rel in ['bench/cfg/numerics.yaml', 'bench/cfg/timing.yaml', 'bench/cases.json', 'bench/traj/spec.json',
            'bench/thresholds.json', 'bench/traj/frozen.json']:
    if (WS / rel).exists():
        inputs[rel] = {'kind': 'sha256', 'value': sha(WS / rel)}
for d in ['bench/records/ref', 'bench/traj/ref']:
    for f in sorted((WS / d).glob('*')) if (WS / d).exists() else []:
        inputs[str(f.relative_to(WS))] = {'kind': 'sha256' if f.suffix == '.json' else 'stat',
                                          'value': sha(f) if f.suffix == '.json' else stat(f)}
out = {'task': 'self_forcing_dmd_training', 'reference_commit': '33593df3e81fa3ec10239271dd2c100facac6de1',
       'inputs': inputs, 'final': '--final' in sys.argv}
(WS / 'task' / 'formulation.json').write_text(json.dumps(out, indent=1) + '\n')
print(len(inputs), 'inputs;', sum(1 for v in inputs.values() if v['kind'] == 'stat'), 'by stat')
