#!/bin/bash
# Inside the sandbox (PREP=1): does the candidate that PASSED before the rule changes still pass
# under the re-frozen delta? This is the control for "the gate was not loosened to let r36 through".
# a2fb998 = Apex fused RMSNorm + bf16 reduced matmul reduction, the 1.0947x candidate of 10-11.
set -uo pipefail
W=${WORK:?set WORK}/llava/ws
T=/tmp
PY=${ASSETS:?set ASSETS}/env/bin/python
cd $W
rm -rf $T/old_cand && mkdir -p $T/old_cand
git -C llava archive a2fb998 | tar -x -C $T/old_cand
echo "== old candidate through gates 0/2/3 (no timing) $(date +%T)"
$PY bench/bench.py check --skip-timing --tree $T/old_cand --config runs/r16/bf16_reduced.json \
  --out $W/runs/fx_control/old_pass
echo "OLD_RC=$?"
echo OLD_CANDIDATE_DONE
