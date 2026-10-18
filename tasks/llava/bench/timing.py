"""Step timing of a source tree's real LLaVA-1.5 finetune entrypoint.

Run under torchrun with the source tree as working directory and on PYTHONPATH:

  torchrun --standalone --nnodes=1 --nproc-per-node=4 <bench>/timing.py --out <prefix> \
      --warmup W --window N --repeats R --math-mode tf32 -- <llava/train/train_mem.py arguments>

The tree's llava/train/train_mem.py runs unmodified as __main__ in this process. Nothing in it is patched: one
TrainerCallback is appended to transformers' default Trainer callbacks. A training step is one optimizer step of the
HF Trainer (on_step_end). After W steps, R consecutive windows of N steps are timed; window edges are bracketed by
torch.cuda.synchronize(), nothing else synchronizes. After the last window the callback raises and training stops.
Per step the callback also keeps the loss the Trainer logs (on_log, rank 0), so the bench can check that every run
trained on the same batches, and the host time at on_step_end (diagnostic).

With --profile-steps K the first K steps of the first window are captured: by nsys (cudaProfilerStart/Stop) or, with
--detail-out, by the PyTorch profiler (shapes, dtypes, stacks with nn.Module frames, memory, op->kernel correlation),
written as <detail-out>.r<rank>.json.gz. Then the run stops.

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

T_START = time.perf_counter()


def compile_count():
    utils = sys.modules.get('torch._dynamo.utils')
    if utils is None:
        return 0
    return utils.counters['stats']['unique_graphs'] + utils.counters['aot_autograd']['total']


class StopTiming(BaseException):
    """BaseException, so `except Exception` blocks in the training code cannot swallow it."""


def kill_descendants():
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


def per_rank_triton_cache():
    """One Triton/DeepSpeed autotune cache directory per rank. All ranks share the bench's cache directory, and
    DeepSpeed's autotune table writes a fixed <dir>/<kernel>.pickle.tmp and renames it, so two ranks racing there
    make one of them fail at exit (FileNotFoundError on the rename)."""
    base = os.environ.get('TRITON_CACHE_DIR') or os.path.join(os.environ.get('TMPDIR', '/tmp'), 'triton')
    d = os.path.join(base, 'rank' + os.environ.get('LOCAL_RANK', '0'))
    os.makedirs(d, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = d


def make_timer(args):
    from transformers import TrainerCallback

    class StepTimer(TrainerCallback):
        def __init__(self):
            self.edges = {}
            self.step_wall = []
            self.losses = {}
            self.first_step_s = None
            self.prof = None
            self.profile_start = None
            self.last = args.warmup + args.repeats * args.window
            if args.profile_steps:
                self.last = args.warmup + args.profile_steps

        def on_step_end(self, targs, state, control, **kwargs):
            g = state.global_step                 # optimizer steps completed
            self.step_wall.append(time.perf_counter() - T_START)
            if g == 1:
                torch.cuda.synchronize()
                self.first_step_s = time.perf_counter() - T_START
            if args.profile_steps:
                if g == args.warmup:
                    torch.cuda.synchronize()
                    self.edges['profile', 0] = (time.perf_counter(), time.time(), compile_count())
                    if args.detail_out:
                        act = torch.profiler.ProfilerActivity
                        self.prof = torch.profiler.profile(activities=[act.CPU, act.CUDA], record_shapes=True,
                                                           with_stack=True, with_modules=True, profile_memory=True)
                        self.prof.start()
                    else:
                        torch.cuda.cudart().cudaProfilerStart()
                elif g == self.last:
                    torch.cuda.synchronize()
                    self.edges['profile', 1] = (time.perf_counter(), time.time(), compile_count())
                    if self.prof is not None:
                        self.prof.stop()
                        rank = int(os.environ.get('RANK', '0'))
                        self.prof.export_chrome_trace(f'{args.detail_out}.r{rank}.json')
                    else:
                        torch.cuda.cudart().cudaProfilerStop()
                    raise StopTiming()
                return
            k, r = divmod(g - args.warmup, args.window)
            if g >= args.warmup and r == 0 and k <= args.repeats:
                torch.cuda.synchronize()
                self.edges['window', k] = (time.perf_counter(), time.time(), compile_count())
                if k == args.repeats:
                    raise StopTiming()

        def on_log(self, targs, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                self.losses[state.global_step] = logs['loss']

        def windows(self):
            out = []
            for k in range(args.repeats):
                if ('window', k) in self.edges and ('window', k + 1) in self.edges:
                    a, b = self.edges['window', k], self.edges['window', k + 1]
                    out.append({'k': k, 'steps': args.window, 'first_step': args.warmup + k * args.window + 1,
                                'seconds': b[0] - a[0], 'wall_start': a[1], 'wall_end': b[1],
                                'compiles_in_window': b[2] - a[2]})
            return out

        def profile_window(self):
            if ('profile', 0) not in self.edges or ('profile', 1) not in self.edges:
                return None
            a, b = self.edges['profile', 0], self.edges['profile', 1]
            return {'steps': args.profile_steps, 'first_step': args.warmup + 1, 'seconds': b[0] - a[0],
                    'compiles_in_window': b[2] - a[2], 'tool': 'torch.profiler' if args.detail_out else 'nsys'}
    return StepTimer()


def main():
    if '--' not in sys.argv:
        raise SystemExit('usage: timing.py [options] -- <train_mem.py arguments>')
    split = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='output prefix; writes <prefix>.r<rank>.json')
    parser.add_argument('--warmup', type=int, required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--repeats', type=int, required=True)
    parser.add_argument('--entry', default='llava/train/train_mem.py')
    parser.add_argument('--math-mode', choices=('default', 'tf32'), default='default',
                        help='frozen process-wide math mode, set before the training code is imported')
    parser.add_argument('--profile-steps', type=int, help='capture this many steps after the warmup, then stop')
    parser.add_argument('--detail-out', help='with --profile-steps: PyTorch profiler instead of nsys; trace prefix')
    args = parser.parse_args(sys.argv[1:split])
    train_argv = sys.argv[split + 1:]
    per_rank_triton_cache()
    if args.math_mode == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    import transformers.trainer
    timer = make_timer(args)
    transformers.trainer.DEFAULT_CALLBACKS.append(timer)
    entry = Path(args.entry).resolve()
    sys.path.insert(0, str(Path.cwd()))
    sys.argv = [str(entry), *train_argv]
    rank = int(os.environ.get('RANK', '0'))
    status, error = 'incomplete', None
    try:
        runpy.run_path(str(entry), run_name='__main__')
    except StopTiming:
        status = 'ok'
    except BaseException as exc:  # report, then exit nonzero
        import traceback
        traceback.print_exc()
        status, error = 'error', f'{type(exc).__name__}: {exc}'
    result = {
        'status': status, 'error': error, 'rank': rank, 'world_size': int(os.environ.get('WORLD_SIZE', '1')),
        'warmup': args.warmup, 'window': args.window, 'repeats': args.repeats,
        'steps_seen': len(timer.step_wall), 'step_wall_s': [round(x, 3) for x in timer.step_wall],
        'startup_to_first_step_s': timer.first_step_s,
        'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved(),
        'windows': timer.windows() if status == 'ok' else [],
        'profile_window': timer.profile_window() if status == 'ok' else None,
        'logged_loss': {str(k): v for k, v in sorted(timer.losses.items())},
        'train_argv': train_argv, 'math_mode': args.math_mode,
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
