#!/usr/bin/env python3
"""Formulation side: does gate 0's new data rule accept exactly what GOAL.md now allows and reject the rest?

  test_gate0.py <probe json of the reference> <probe json of a legally repartitioned tree>

Both arguments are the last line of `loader_probe.py --tree ... --batches 16`. The accept cases are run
end to end on real probe output; the reject cases mutate that output, which is what a tree doing the
illegal thing would hand the judge.
"""
import os
import copy
import importlib.util
import json
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

ref = json.loads(Path(sys.argv[1]).read_text().strip().splitlines()[-1])
alt = json.loads(Path(sys.argv[2]).read_text().strip().splitlines()[-1])
fails = 0


def case(name, got, want_reject):
    global fails
    bad = m.loader_problem(ref, got)
    ok = bool(bad) == want_reject
    fails += not ok
    verdict = 'reject' if bad else 'accept'
    print(f'{"ok  " if ok else "FAIL"}  {name:52s} -> {verdict}' + (f'  :: {bad[0]}' if bad else ''))


def mutate(fn):
    g = copy.deepcopy(ref)
    fn(g)
    return g


case('reference vs itself', copy.deepcopy(ref), False)
case('legal repartition of every step', alt, False)

# the same 128 samples, but the row order inside a micro-batch reversed: allowed
case('row order inside a micro-batch reversed',
     mutate(lambda g: [b.__setitem__('row_shas', b['row_shas'][::-1]) for b in g['batches']]), False)


# whole micro-batches swapped between ranks inside one step: allowed
def swap_micro(g):
    g['batches'][0], g['batches'][3] = g['batches'][3], g['batches'][0]
    for i, b in enumerate(g['batches']):
        b['rank'], b['micro'], b['step'] = i % 4, (i // 4) % 2, i // 8


case('two micro-batches of one step swapped', mutate(swap_micro), False)


# --- illegal ---

def dup_inside_step(g):
    g['batches'][0]['row_shas'][0] = g['batches'][1]['row_shas'][0]


case('a sample duplicated inside the step', mutate(dup_inside_step), True)


def move_across_steps(g):
    a, b = g['batches'][0]['row_shas'], g['batches'][8]['row_shas']
    a[0], b[0] = b[0], a[0]


case('one sample swapped between step 0 and step 1', mutate(move_across_steps), True)


def resize(g):
    g['batches'][0]['rows'] = 8
    g['batches'][0]['row_shas'] = g['batches'][0]['row_shas'][:8]


case('per-device batch 16 -> 8 in one micro-batch', mutate(resize), True)


def repreprocess(g):
    g['batches'][2]['row_shas'][5] = 'deadbeefdeadbeef'


case('one sample preprocessed differently', mutate(repreprocess), True)


def fewer_tokens(g):
    g['batches'][1]['tokens'] -= 7


case('same samples, fewer tokens reported', mutate(fewer_tokens), True)


def fewer_labels(g):
    g['batches'][1]['labels'] -= 3


case('same samples, fewer labels reported', mutate(fewer_labels), True)


def drop_micro(g):
    g['batches'] = g['batches'][:-1]


case('a micro-batch dropped', mutate(drop_micro), True)


def shrink_dataset(g):
    g['dataset_len'] -= 1


case('the dataset got smaller', mutate(shrink_dataset), True)

print(f'\n{"ALL OK" if not fails else str(fails) + " FAILED"}')
sys.exit(1 if fails else 0)
