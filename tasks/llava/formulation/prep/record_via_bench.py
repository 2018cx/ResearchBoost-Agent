#!/usr/bin/env python3
"""Inside the sandbox (PREP=1): record the reference records through bench.py's own code path.

  record_via_bench.py <records out dir> <scratch dir>

Not a hand-rolled torchrun: record_cases() goes through run_gpu(), which holds the GPU lock and sets the
environment a real check sets (compile caches, determinism, NCCL). A hand-rolled launch produced complete
records but then aborted in glibc at teardown, and on the host it also hashed 8 CUDA RNG states instead of
the sandbox's 4, giving a fingerprint no candidate could reproduce.
"""
import os
import importlib.util
import sys
from pathlib import Path

WS = Path(os.environ['WORK']) / 'llava' / 'ws'
spec = importlib.util.spec_from_file_location('b', WS / 'bench' / 'bench.py')
m = importlib.util.module_from_spec(spec)
sys.modules['b'] = m
try:
    spec.loader.exec_module(m)
except SystemExit:
    pass

rec = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
r = m.record_cases('targets/reference', out, {}, rec)
print('record_cases:', r)
sys.exit(0 if r.get('status') == 'ok' else 1)
