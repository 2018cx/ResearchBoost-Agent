#!/bin/bash
# Inside the sandbox (PREP=1): is gate 3 sensitive enough to reject a deliberately wrong candidate?
#
# A control arm: a deliberately wrong copy of the reference. delta is only the reference's own seed-to-seed
# spread (train_loss_last20 0.01421 ~ 1.50%, probe_loss_last20 0.00653 ~ 1.35%), and nothing has ever shown
# that gate 3 rejects a candidate whose training semantics are wrong.
#
# The control multiplies the learning rate the OPTIMIZER sees by 1.02, after the scheduler has recorded
# _last_lr, so gate 2's exact item lr_after_step still matches and the candidate actually reaches gate 3.
# (Scaling what the scheduler reports would fail gate 2 first and measure nothing about gate 3. Scaling
# every gradient instead would be absorbed by Adam's scale invariance -- no clipping in this recipe.)
# LambdaLR recomputes lr absolutely from base_lr each step, so the factor does not compound.
#
# Gate 3 is a PAIRED test (d_s = candidate(seed s) - reference(seed s)), so the difference is deterministic
# and one seed already shows the magnitude; the schedule still runs 3.
set -uo pipefail
W=${WORK:?set WORK}/llava/ws
T=/tmp                      # the sandbox's private /tmp
PY=$ASSETS/env/bin/python
cd $W

rm -rf $T/ctrl_lr
cp -a targets/reference $T/ctrl_lr
$PY - <<'PY'
from pathlib import Path
p = Path('/tmp/ctrl_lr/llava/train/train.py')
s = p.read_text()
anchor = 'def train(attn_implementation=None):'
assert s.count(anchor) == 1, s.count(anchor)
patch = '''def _control_scale_applied_lr(factor=1.02):
    """Control arm: the optimizer gets factor x the scheduled learning rate, while the scheduler keeps
    reporting the unscaled value (so the exact items of gate 2 are untouched and the arm reaches gate 3).
    Deliberately wrong training semantics; GOAL.md allows this nowhere."""
    import torch
    real_step = torch.optim.lr_scheduler.LambdaLR.step

    def step(self, *a, **kw):
        real_step(self, *a, **kw)           # sets _last_lr from the unscaled group lrs
        for group in self.optimizer.param_groups:
            group['lr'] = group['lr'] * factor
    torch.optim.lr_scheduler.LambdaLR.step = step


_control_scale_applied_lr()


'''
p.write_text(s.replace(anchor, patch + anchor))
print('control patched (applied lr x1.02, reported lr unchanged)')
PY
[ $? -ne 0 ] && { echo PATCH_FAILED; exit 1; }

echo "== control arm through gates 0/2/3 (no timing) $(date +%T)"
$PY bench/bench.py check --skip-timing --tree $T/ctrl_lr --out $W/runs/fx_control/lr102
echo "CTRL_RC=$?"
echo CONTROL_DONE
