#!/bin/bash
# Acceptance of the frozen judge: known-good and known-bad trees must get the verdicts below, and the numerics
# self-test must pass. Run inside the sandbox with PREP=1 (sb.sh accept bash accept.sh). Results: runs/accept/<name>/.
#   a  unmodified reference                                   -> bitwise PASS (quick: gates 0 and 2)
#   b  setup change in main()                                 -> ERROR at gate 0
#   c  GAN classifier never sees the real images              -> INCORRECT (exact checks: calls, RNG)
#   d  math-equivalent rewrite of the DMD gradient            -> not bitwise, statistical gate PASS
#   d2 CFG combination rewritten                              -> not bitwise, statistical gate PASS
#   e  GAN classification loss detached in the guidance step  -> INCORRECT (missing tensors)
#   f  guidance GAN loss weighted 1.2x                        -> exact checks hold, statistical gate INCORRECT
set -uo pipefail
cd ${WORK:?set WORK}/dmd/ws
A=/tmp/accept_trees
mkdir -p runs/accept
python "$(cd "$(dirname "$0")" && pwd)/make_accept.py" targets/reference $A | tail -8

echo "=== numerics self-test (CPU, mutation tests of the judge on its own records)"
PYTHONPATH=. python -m numerics selftest --ref bench/records/ref --variant bench/records/var_sdpa_mem \
  --thresholds bench/thresholds.json --out runs/accept/selftest.json 2>&1 | tail -2
check() { echo "=== $1"; python bench/bench.py check --tree $2 --out runs/accept/$1 $3 2>&1 | tail -1; }
check unmodified targets/reference --quick
check setup_main $A/setup_main --quick
check skip_real_cls $A/skip_real_cls --quick
check dmd_reorder $A/dmd_reorder --skip-timing
check cfg_reorder $A/cfg_reorder --skip-timing
check detach_guidance_cls $A/detach_guidance_cls --quick
check cls_weight_120pct $A/cls_weight_120pct --skip-timing

echo "=== summary"
python - <<'PY'
import json
from pathlib import Path
rows = []
for name in ('unmodified', 'setup_main', 'skip_real_cls', 'dmd_reorder', 'cfg_reorder', 'detach_guidance_cls',
             'cls_weight_120pct'):
    f = Path('runs/accept') / name / 'check.json'
    if not f.exists():
        rows.append((name, 'MISSING', '', ''))
        continue
    d = json.loads(f.read_text())
    c = d.get('correctness') or {}
    rows.append((name, d.get('status'), c.get('stage'), c.get('verdict') or c.get('measured_tier') or ''))
st = json.loads(Path('runs/accept/selftest.json').read_text()) if Path('runs/accept/selftest.json').exists() else {}
print(json.dumps({'cases': rows, 'selftest_ok': st.get('ok'),
                  'selftest_failed': [r['mutation'] for r in st.get('results', []) if not r['ok']]}, indent=1))
PY
echo ACCEPT_DONE
