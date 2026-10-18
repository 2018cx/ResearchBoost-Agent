#!/bin/bash
# Acceptance of the evaluator (run inside the sandbox with PREP=1). One item per argument:
#   a  unmodified      -> check --quick PASS (bitwise), and `check` on the formal path -> UNAVAILABLE/no_benefit
#   b  train() changed  -> ERROR (setup guard)
#   c  sampler forgets the accumulation factor -> INCORRECT (gate 0, data loader)
#   c2 image features reused across micro-batches -> INCORRECT (gate 2)
#   d  math-equivalent RMSNorm  -> gate 3 passes (check --skip-timing)
#   e  projector detached       -> gate 3 fails (check --skip-timing)
# Output: task/acceptance/<item>/check.json (plus the bench's own logs).
set -uo pipefail
W=${WORK:?set WORK}/llava/ws
cd $W
export PYTHONPATH=$W
mkdir -p task/acceptance
for item in "$@"; do
  T=/tmp/acc_$item
  O=task/acceptance/$item
  mkdir -p $O
  echo "== acceptance $item $(date +%T)"
  case $item in
    a) python3 bench/bench.py check --tree $T --out $O --quick ;;
    b|c) python3 bench/bench.py check --tree $T --out $O --quick ;;
    c2) python3 bench/bench.py check --tree $T --out $O --quick ;;
    d|e) python3 bench/bench.py check --tree $T --out $O --skip-timing ;;
    full_a) python3 bench/bench.py check --tree /tmp/acc_a --out task/acceptance/full_a ;;
    *) echo "unknown item $item"; exit 2 ;;
  esac
  python3 - "$O/check.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
c = d.get('correctness') or {}
print('  ->', d['status'], c.get('stage'), c.get('verdict'), c.get('measured_tier'),
      'bitwise', c.get('bitwise'), '| perf', (d.get('performance') or {}).get('speedup'))
print('  note:', (c.get('note') or '')[:300])
for k in ('problems', 'exact_failures'):
    v = (c.get('numerics') or {}).get(k) or c.get(k) or []
    if v:
        print(' ', k, json.dumps(v)[:400])
t = c.get('trajectory')
if t:
    print('  traj', t.get('verdict'), {n: (e['state'], round(e['mean'], 5), round(e['delta'], 5))
                                       for n, e in (t.get('endpoints') or {}).items()})
PY
done
echo "== DONE acceptance $(date +%T)"
