"""Iteration timing of a source tree's real Self-Forcing training entrypoint (train.py), nothing patched.

Run under torchrun with the source tree as working directory:

  torchrun --standalone --nnodes=1 --nproc-per-node=2 <bench>/timing.py --out <prefix> [options] -- \
      --config_path <timing.yaml> --logdir <dir> --no_save --no_visualize --disable-wandb

An iteration of ScoreDistillationTrainer.train() is one critic optimizer step (every iteration ends with one);
every dfake_gen_update_ratio-th iteration also has a generator optimizer step before it. Optimizers are told apart
through torch's global optimizer step hook: the first optimizer to step (iteration 0 is a generator iteration) is
the generator optimizer, every other one the critic optimizer.

The first `warmup` iterations are skipped (model load, first compile, negative-prompt cache, autotuning), then
`repeats` windows of `window` iterations are timed. warmup and window are multiples of the update ratio, so every
window holds whole cycles (1 generator iteration + ratio-1 critic-only iterations). Window edges are bracketed by
torch.cuda.synchronize(); nothing else synchronizes. Each iteration is an NVTX range ("iter <i> gen|critic").
After the last window the hook raises and training stops.

--profile: call cudaProfilerStart at the end of warmup and cudaProfilerStop after one window, then stop.

Each rank writes <prefix>.r<rank>.json; the caller merges ranks (max over ranks per window).
"""
import argparse
import json
import os
import runpy
import signal
import sys
import time
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

T_START = time.perf_counter()


def compile_count():
    """Graphs torch.compile has compiled so far in this process (0 if dynamo was never imported)."""
    utils = sys.modules.get('torch._dynamo.utils')
    if utils is None:
        return 0
    return utils.counters['stats']['unique_graphs'] + utils.counters['aot_autograd']['total']


class StopTiming(BaseException):
    """BaseException, so `except Exception` blocks in the training code cannot swallow it."""


def kill_descendants():
    """End every process this one started (compile workers, data-loader workers) before os._exit; a profiler such
    as nsys waits for the whole process tree."""
    children = {}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            with open(f'/proc/{d}/stat') as f:
                ppid = int(f.read().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d))
    stack, found = [os.getpid()], []
    while stack:
        for c in children.get(stack.pop(), []):
            found.append(c)
            stack.append(c)
    for pid in found:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return len(found)


class IterTimer:
    def __init__(self, warmup, window, repeats, profile=False):
        self.warmup, self.window, self.repeats, self.profile = warmup, window, repeats, profile
        self.marks = {warmup + k * window - 1: k for k in range(repeats + 1)}   # iteration index -> edge k
        self.last = max(self.marks)
        self.generator = None
        self.it = -1                    # iterations finished, minus one
        self.gen_steps = 0
        self.edges = {}                 # k -> (perf_counter, wall, generator steps so far, compiled graphs)
        self.first_iter_s = None
        self.nvtx_open = False
        self.pending_gen = False
        self.kinds = []                 # per finished iteration: 'gen' or 'critic'

    def _open(self):
        torch.cuda.nvtx.range_push(f'iter {self.it + 1}')
        self.nvtx_open = True

    def __call__(self, opt, _args, _kwargs):
        if self.generator is None:
            self.generator = id(opt)
        if id(opt) == self.generator:
            if self.pending_gen:
                raise RuntimeError('two generator steps without a critic step in between')
            self.gen_steps += 1
            self.pending_gen = True
            return
        # critic step: the iteration ends here
        self.it += 1
        self.kinds.append('gen' if self.pending_gen else 'critic')
        self.pending_gen = False
        if self.nvtx_open:
            torch.cuda.nvtx.range_pop()
            self.nvtx_open = False
        if self.it == 0:
            torch.cuda.synchronize()
            self.first_iter_s = time.perf_counter() - T_START
        k = self.marks.get(self.it)
        if k is not None:
            torch.cuda.synchronize()
            self.edges[k] = (time.perf_counter(), time.time(), self.gen_steps, compile_count())
            if self.profile:
                if k == 0:
                    torch.cuda.cudart().cudaProfilerStart()
                else:
                    torch.cuda.cudart().cudaProfilerStop()
                    raise StopTiming()
            if self.it == self.last:
                raise StopTiming()
        self._open()

    def report(self):
        windows = []
        for k in range(self.repeats):
            if k not in self.edges or k + 1 not in self.edges:
                continue
            a, b = self.edges[k], self.edges[k + 1]
            first = self.warmup + k * self.window
            windows.append({'k': k, 'iterations': self.window, 'seconds': b[0] - a[0], 'wall_start': a[1],
                            'wall_end': b[1], 'generator_steps': b[2] - a[2], 'compiles_in_window': b[3] - a[3],
                            'kinds': self.kinds[first:first + self.window]})
        return windows


def main():
    if '--' not in sys.argv:
        raise SystemExit('usage: timing.py [options] -- <train.py arguments>')
    split = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='output prefix; writes <prefix>.r<rank>.json')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--entry', default='train.py')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args(sys.argv[1:split])
    train_argv = sys.argv[split + 1:]

    timer = IterTimer(args.warmup, args.window, 1 if args.profile else args.repeats, args.profile)
    handle = register_optimizer_step_post_hook(timer)
    entry = Path(args.entry).resolve()
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), *train_argv]
    rank = int(os.environ.get('RANK', '0'))
    status, error = 'incomplete', None
    try:
        runpy.run_path(str(entry), run_name='__main__')
    except StopTiming:
        status = 'ok'
    except BaseException as exc:  # report, then exit nonzero
        status, error = 'error', f'{type(exc).__name__}: {exc}'
    finally:
        handle.remove()
    result = {
        'status': status, 'error': error, 'rank': rank, 'world_size': int(os.environ.get('WORLD_SIZE', '1')),
        'warmup': args.warmup, 'window': args.window, 'repeats': timer.repeats, 'profile': args.profile,
        'iterations_seen': timer.it + 1, 'generator_steps_seen': timer.gen_steps,
        'startup_to_first_iteration_s': timer.first_iter_s,
        'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved(),
        'windows': timer.report() if status == 'ok' else [],
        'train_argv': train_argv,
        'math_flags_at_end': {'matmul.allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                              'cudnn.allow_tf32': torch.backends.cudnn.allow_tf32},
    }
    Path(f'{args.out}.r{rank}.json').write_text(json.dumps(result, indent=1) + '\n')
    sys.stdout.flush()
    sys.stderr.flush()
    kill_descendants()
    os._exit(0 if status == 'ok' else 1)


if __name__ == '__main__':
    main()
