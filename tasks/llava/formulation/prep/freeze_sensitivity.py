#!/usr/bin/env python3
"""Formulation side: turn the two measurement runs into gate 1's new step-1 loss tolerance.

  freeze_sensitivity.py <runs/fx_partition/m1/time_m1.json> <.../time_m2.json>

Gate 1 compares the candidate's first logged loss -- a mean over the step's micro-batches of each one's
token-mean cross entropy, taken before any optimizer step -- with the frozen reference, and the old 0.5%
bound held because the partition of the step's 128 samples was itself frozen. It no longer is, so the
quantity moves on a tree that trains on exactly the reference's samples. These two runs are the REFERENCE
tree under two legal partitions; the new bound is twice the largest shift they show, rounded up.

What the frozen record deliberately does not say is how the two partitions were built: that is in the
formulation notes outside the sandbox. The samples themselves are held by the multiset exact checks of
gates 0 and 2, not by this bound.
"""
import os
import json
import math
import sys
from pathlib import Path

WS = Path(os.environ['WORK']) / 'llava' / 'ws'
ref = json.loads((WS / 'task' / 'timing' / 'reference.json').read_text())
ll = ref['logged_loss']
first_key = min(ll, key=lambda k: float(k))
base = ll[first_key]
print(f'frozen reference first logged loss: step {first_key} = {base}')

observed = []
for i, p in enumerate(sys.argv[1:], start=1):
    res = json.loads(Path(p).read_text())
    cand = res['logged_loss']
    k = min(cand, key=lambda x: float(x))
    assert k == first_key, (k, first_key)
    rel = (cand[k] - base) / base
    observed.append({'arm': chr(ord('A') + i - 1), 'first_logged_loss': cand[k], 'rel_diff': round(rel, 6),
                     'ms_per_step_hidden': True})
    print(f"  arm {observed[-1]['arm']}: {cand[k]}  rel {rel:+.5f}   "
          f"(all logged steps max rel "
          f"{max(abs((cand[s] - ll[s]) / ll[s]) for s in cand if s in ll):.5f})")

worst = max(abs(o['rel_diff']) for o in observed)
tol = max(0.005, math.ceil(2 * worst * 1000) / 1000)
print(f'\nworst observed |rel| = {worst:.5f} -> tolerance {tol} ({tol * 100:.1f}%)')

out = {
    'what': '第 1 关「第一步 logged loss」对「同一批样本、不同合法分配」的敏感度，出题方用参考自己的代码测的',
    'why': 'GOAL.md 的题面改动放开了一个 optimizer step 内部的样本分配。第一步 logged loss 是'
           '「每个 micro-batch 的 token 平均交叉熵，再在 micro-batch 之间平均」，分配一变，即使训练的是'
           '同一批 128 条样本，这个数也会动，所以原来的 0.5% 不再是一个「不可能动」的量。',
    'quantity': 'bench.py time 的 logged_loss 第一条（任何 optimizer step 之前）',
    'frozen_reference': {'step': first_key, 'loss': base},
    'legal_partitions_measured': len(observed),
    'observed': [{k: v for k, v in o.items() if k != 'ms_per_step_hidden'} for o in observed],
    'max_abs_rel_diff': round(worst, 6),
    'tolerance': tol,
    'rule': 'tolerance = 2 × 观测到的最大偏移，向上取到 0.1%，且不低于原来的 0.005',
    'note': '两种分配具体怎么构造的记在出题侧（沙箱外），这里不写。样本集合本身由第 0 关和第 2 关的'
            '多重集合精确项管，不靠这条界管。',
}
(WS / 'task' / 'partition_sensitivity.json').write_text(
    json.dumps(out, indent=1, ensure_ascii=False) + '\n')
print('wrote task/partition_sensitivity.json')
print('TOLERANCE_PERCENT', round(tol * 100, 1))
