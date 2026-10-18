"""Freeze task/timing_ref.json from a baseline.json (bench.py baseline output)."""
import json, statistics, sys
from pathlib import Path
b = json.loads(Path(sys.argv[1]).read_text())
runs = [r for r in b['runs'].values() if r['status'] == 'ok']
assert b['rng_identical_across_runs'], 'the reference does not draw the same random numbers in every run'
tot = [r['total_s'] for r in runs]
ws = [r['window_s'] for r in runs]
med = statistics.median(tot)
ref = {
    'source': sys.argv[1], 'runs_total_s': tot, 'total_s': round(med, 3),
    'window_s': [round(statistics.median(w[k] for w in ws), 3) for k in range(len(ws[0]))],
    's_per_iteration': round(med / runs[0]['iterations_timed'], 4),
    'spread': b['spread'],
    'peak_memory_allocated_gib': max(r['peak_memory_allocated_gib'] for r in runs),
    'rng_fingerprints': b['rng_fingerprints'],
}
ref['window_ratio'] = round(ref['window_s'][0] / ref['window_s'][1], 4)
ref['window_ratios'] = [r.get('window_ratio') for r in runs]
Path(sys.argv[2]).write_text(json.dumps(ref, indent=1) + '\n')
print(json.dumps({k: v for k, v in ref.items() if k != 'rng_fingerprints'}, indent=1))
