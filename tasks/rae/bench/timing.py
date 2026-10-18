"""Phase-resolved step timing of a source tree's real Stage-1 entrypoint.

Run under torchrun with the source tree as working directory:

  torchrun --standalone --nnodes=1 --nproc-per-node=2 <bench>/timing.py --out <prefix> \
      --steps-per-epoch S -- --config <timing.yaml> --data-path <fixture> --results-dir <dir> \
      --precision fp32 --image-size 256

The tree's src/train_stage1.py runs unmodified as __main__ in this process. Nothing in it is
patched. A training step is one step() of the generator optimizer, seen through torch's global
optimizer post-step hook; the generator optimizer is the one holding the most parameter elements.

The timing config compresses the upstream schedule to three epochs (0: reconstruction + LPIPS,
1: discriminator updates, 2: adversarial generator loss). In each epoch the first `warmup` steps
are skipped, then `repeats` windows of `window` steps are timed. Window edges are bracketed by
torch.cuda.synchronize(); nothing else synchronizes. Every window lies inside one epoch, so
epoch-boundary work is excluded. After the last window the hook raises and training stops.

Each rank writes <prefix>.r<rank>.json; the caller merges ranks (max over ranks per window).
"""
import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

PHASES = ('recon', 'disc', 'gan')
T_START = time.perf_counter()


def compile_count():
    """Graphs torch.compile has compiled so far in this process (0 if dynamo was never imported)."""
    utils = sys.modules.get('torch._dynamo.utils')
    if utils is None:
        return 0
    return utils.counters['stats']['unique_graphs'] + utils.counters['aot_autograd']['total']


class StopTiming(BaseException):
    """BaseException, so `except Exception` blocks in the training code cannot swallow it."""


def window_marks(steps_per_epoch, warmup, window, repeats):
    """G-step index (0-based, observed after that step's generator update) -> (phase, edge k)."""
    marks = {}
    for epoch, phase in enumerate(PHASES):
        if warmup[epoch] < 1 or warmup[epoch] + repeats * window[epoch] > steps_per_epoch:
            raise SystemExit(f'timing plan does not fit {steps_per_epoch} steps/epoch in phase {phase}')
        for k in range(repeats + 1):
            marks[epoch * steps_per_epoch + warmup[epoch] + k * window[epoch] - 1] = (phase, k)
    return marks


class StepTimer:
    def __init__(self, marks, window, profile_phase=None):
        self.marks, self.window = marks, window
        self.profile_phase = profile_phase
        self.last = max(marks)
        self.numel = {}            # id(optimizer) -> parameter elements
        self.generator = None
        self.g = -1                # generator steps seen so far, minus one
        self.other_steps = 0       # steps of all other optimizers
        self.edges = {}            # (phase, k) -> (perf_counter, wall time, other_steps, compiled graphs)
        self.first_step_s = None

    def _elements(self, opt):
        key = id(opt)
        if key not in self.numel:
            self.numel[key] = sum(p.numel() for group in opt.param_groups for p in group['params'])
        return self.numel[key]

    def __call__(self, opt, _args, _kwargs):
        n = self._elements(opt)
        if self.generator is None:
            self.generator = id(opt)
        elif id(opt) != self.generator and n > self.numel[self.generator]:
            raise RuntimeError('a larger optimizer stepped after the generator optimizer; '
                               'cannot identify the generator step')
        if id(opt) != self.generator:
            self.other_steps += 1
            return
        self.g += 1
        if self.g == 0:
            torch.cuda.synchronize()
            self.first_step_s = time.perf_counter() - T_START
        mark = self.marks.get(self.g)
        if mark is None:
            return
        torch.cuda.synchronize()
        self.edges[mark] = (time.perf_counter(), time.time(), self.other_steps, compile_count())
        if self.profile_phase == mark[0]:
            # Capture exactly the first window of the chosen phase, then stop.
            if mark[1] == 0:
                torch.cuda.cudart().cudaProfilerStart()
            else:
                torch.cuda.cudart().cudaProfilerStop()
                raise StopTiming()
        if self.g == self.last:
            raise StopTiming()

    def report(self, repeats):
        windows = []
        for phase_index, phase in enumerate(PHASES):
            for k in range(repeats):
                if (phase, k) not in self.edges or (phase, k + 1) not in self.edges:
                    continue
                a, b = self.edges[(phase, k)], self.edges[(phase, k + 1)]
                windows.append({'phase': phase, 'k': k, 'steps': self.window[phase_index],
                                'seconds': b[0] - a[0], 'wall_start': a[1], 'wall_end': b[1],
                                'other_optimizer_steps': b[2] - a[2], 'compiles_in_window': b[3] - a[3]})
        return windows


def main():
    if '--' not in sys.argv:
        raise SystemExit('usage: timing.py [options] -- <train_stage1.py arguments>')
    split = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='output prefix; writes <prefix>.r<rank>.json')
    parser.add_argument('--steps-per-epoch', type=int, required=True)
    parser.add_argument('--warmup', type=int, nargs=3, default=[20, 10, 10])
    parser.add_argument('--window', type=int, nargs=3, default=[20, 25, 25])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--entry', default='src/train_stage1.py')
    parser.add_argument('--math-mode', choices=('default', 'tf32'), default='default',
                        help='frozen process-wide math mode, set before the training code is imported')
    parser.add_argument('--profile-phase', choices=PHASES,
                        help='call cudaProfilerStart/Stop around the first window of this phase, then stop')
    args = parser.parse_args(sys.argv[1:split])
    train_argv = sys.argv[split + 1:]

    if args.math_mode == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    marks = window_marks(args.steps_per_epoch, args.warmup, args.window, args.repeats)
    timer = StepTimer(marks, args.window, args.profile_phase)
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
        'status': status, 'error': error, 'rank': rank,
        'world_size': int(os.environ.get('WORLD_SIZE', '1')),
        'steps_per_epoch': args.steps_per_epoch, 'warmup': args.warmup, 'window': args.window,
        'repeats': args.repeats, 'generator_steps_seen': timer.g + 1,
        'generator_param_elements': timer.numel.get(timer.generator),
        'startup_to_first_step_s': timer.first_step_s,
        'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved(),
        'windows': timer.report(args.repeats) if status == 'ok' else [],
        'train_argv': train_argv, 'math_mode': args.math_mode,
        'math_flags_at_end': {'matmul.allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                              'cudnn.allow_tf32': torch.backends.cudnn.allow_tf32},
    }
    Path(f'{args.out}.r{rank}.json').write_text(json.dumps(result, indent=1) + '\n')
    sys.stdout.flush()
    sys.stderr.flush()
    # Leave without tearing down the interrupted training loop (loader workers, process group).
    os._exit(0 if status == 'ok' else 1)


if __name__ == '__main__':
    main()
