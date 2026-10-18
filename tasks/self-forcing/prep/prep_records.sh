#!/bin/bash
# Inside the sandbox (PREP=1): reference records for gate 0/2 and diagnostic thresholds.
set -uo pipefail
G=${WORK:?set WORK}/self-forcing; W=$G/ws; P=$G/prep
cd $W
python $P/freeze_inputs.py
REF=$P/reftree; rm -rf $REF
python - <<PY
import sys; sys.path.insert(0, '$W')
from bench import bench
bench.materialize(bench.check_reference(), __import__('pathlib').Path('$REF'))
print('reference tree at $REF')
PY
rec() {  # rec <out> [--deterministic]
  flock /tmp/sf_gpu.lock torchrun --standalone --nnodes=1 --nproc-per-node=2 -m numerics record \
    --adapter bench.sf_adapter:SelfForcingAdapter \
    --adapter-kwargs "{\"repo\": \"$REF\", \"config\": \"$W/bench/cfg/numerics.yaml\"}" \
    --cases bench/cases.json --sketch-dim 4096 --math-mode default "${@:2}" --out "$1"
}
export PYTHONPATH=$W
date; echo "== ref"; rec $W/bench/records/ref --deterministic > $P/rec_ref.log 2>&1; echo "rc=$?"
date; echo "== ref2"; rec $P/records/ref2 --deterministic > $P/rec_ref2.log 2>&1; echo "rc=$?"
date; echo "== nondet"; rec $P/records/nondet > $P/rec_nondet.log 2>&1; echo "rc=$?"
date
python -m numerics calibrate --ref $W/bench/records/ref --variants $P/records/ref2 $P/records/nondet --out bench/thresholds.json
python -m numerics compare --ref $W/bench/records/ref --cand $P/records/ref2 --thresholds bench/thresholds.json --out $P/cmp_ref2.json; echo "ref2 rc=$?"
python -m numerics compare --ref $W/bench/records/ref --cand $P/records/nondet --thresholds bench/thresholds.json --out $P/cmp_nondet.json; echo "nondet rc=$?"
echo PREP_RECORDS_DONE
