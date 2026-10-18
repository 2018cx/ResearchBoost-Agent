#!/bin/bash
# Inside the sandbox (PREP=1): gate-0 acceptance on known candidates (bench.py check --quick).
#   unchanged -> PASS (bitwise)   train.py edited -> ERROR (structure guard)   teacher uncond dropped -> INCORRECT
set -uo pipefail
G=${WORK:?set WORK}/self-forcing; W=$G/ws; P=$G/prep; A=$P/accept
cd $W
rm -rf $A; mkdir -p $A
for name in unchanged trainpy nocfg; do
  python - <<PY
import sys; sys.path.insert(0, '$W')
from pathlib import Path
from bench import bench
bench.materialize(bench.check_reference(), Path('$A/$name'))
PY
done
echo "import os  # acceptance: train.py must equal the reference" >> $A/trainpy/train.py
python - <<PY
from pathlib import Path
p = Path('$A/nocfg/model/dmd.py')
s = p.read_text()
old = '''        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )'''
assert old in s
p.write_text(s.replace(old, '        pred_real_image_uncond = pred_real_image_cond  # acceptance: drops the uncond teacher call'))
PY
for name in unchanged trainpy nocfg; do
  date; echo "== $name"
  python bench/bench.py check --tree $A/$name --out $A/out_$name --quick > $A/$name.log 2>&1
  python -c "import json; d=json.load(open('$A/out_$name/check.json')); c=d.get('correctness') or {}; print(d['status'], c.get('stage'), c.get('verdict'), c.get('bitwise'), (c.get('problems') or '')[:200] if isinstance(c.get('problems'), str) else c.get('problems'), [x for x in (c.get('numerics') or {}).get('exact_failures', [])][:3])"
done
echo ACCEPT_QUICK_DONE
