#!/bin/bash
# Inside the sandbox (PREP=1): how much does a LEGAL repartition move the first logged loss?
#
# Gate 1 compares the candidate's first logged loss (recorded before any optimizer step) with the frozen
# reference and allows 0.5%. That bound was set when the partition of a step's 128 samples over the 8
# micro-batches was itself frozen, so the quantity could not move at all. Now that the partition is the
# candidate's choice (after the GOAL.md rule change), the logged loss -- a mean over micro-batches of each one's
# token-mean CE -- moves with the partition even though the step trains on exactly the same samples.
# Two legal partitions of the REFERENCE tree measure that sensitivity, so the new bound comes from the
# reference itself and not from any candidate.
#   m1: the length-sorted megabatch cut into contiguous blocks, slot i paired with slot i+4 reversed
#       (what a padding-driven candidate would do: micro-batches of near-equal length)
#   m2: a deterministic shuffle of the megabatch, then contiguous blocks (the opposite extreme)
set -uo pipefail
W=${WORK:?set WORK}/llava/ws
cd $W
PY=$ASSETS/env/bin/python

for n in 1 2; do
  rm -rf /tmp/mt$n
  cp -a targets/reference /tmp/mt$n
done

$PY - <<'PY'
from pathlib import Path

OLD = '''    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks'''

M1 = '''    per = len(indices) // num_chunks
    blocks = [indices[i * per:(i + 1) * per] for i in range(num_chunks)]
    half = num_chunks // 2
    return blocks[:half] + blocks[::-1][:half]'''

M2 = '''    import random
    per = len(indices) // num_chunks
    idx = list(indices)
    random.Random(1000003 * len(idx) + (idx[0] if idx else 0)).shuffle(idx)
    return [idx[i * per:(i + 1) * per] for i in range(num_chunks)]'''

for n, new in ((1, M1), (2, M2)):
    p = Path(f'/tmp/mt{n}/llava/train/llava_trainer.py')
    s = p.read_text()
    assert s.count(OLD) == 1, (n, s.count(OLD))
    p.write_text(s.replace(OLD, new))
    print('patched', p)
PY
[ $? -ne 0 ] && { echo PATCH_FAILED; exit 1; }

for n in 1 2; do
  echo "== m$n time $(date +%T)"
  $PY bench/bench.py time --tree /tmp/mt$n --out $W/runs/fx_partition/m$n --label m$n
  echo "RC=$?"
done
echo MEASURE_DONE
