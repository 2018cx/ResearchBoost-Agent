"""Bitwise comparison of two numerics record sets (formulation check: the reference must be reproducible).
cmp_records.py A B"""
import json
import sys
from pathlib import Path


def load(root):
    out = {}
    for p in sorted(Path(root).glob('*.r*.json')):
        m = json.loads(p.read_text())
        out[(m['case']['name'], m['rank'])] = m
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
print(f'{len(a)} vs {len(b)} manifests')
bad = 0
for key in sorted(set(a) | set(b)):
    if key not in a or key not in b:
        print('missing', key)
        bad += 1
        continue
    for ea, eb in zip(a[key]['steps'], b[key]['steps']):
        for field in ('phase', 'rng_before', 'rng_after', 'counters', 'invariants', 'math_mode'):
            if ea.get(field) != eb.get(field):
                print(key, 'step', ea['step'], field, 'differs')
                if field == 'invariants':
                    for k in sorted(set(ea[field]) | set(eb[field])):
                        if ea[field].get(k) != eb[field].get(k):
                            print('   ', k, ea[field].get(k), '!=', eb[field].get(k))
                bad += 1
        names = set(ea['tensors']) | set(eb['tensors'])
        diff = [n for n in sorted(names)
                if ea['tensors'].get(n, {}).get('sha256') != eb['tensors'].get(n, {}).get('sha256')]
        print(key, 'step', ea['step'], len(names), 'tensors,', len(diff), 'differing')
        for n in diff[:10]:
            da, db = ea['tensors'].get(n, {}), eb['tensors'].get(n, {})
            print('   ', n, da.get('norm'), db.get('norm'))
        bad += len(diff)
print('BITWISE IDENTICAL' if not bad else f'NOT IDENTICAL ({bad} problems)')
sys.exit(0 if not bad else 1)
