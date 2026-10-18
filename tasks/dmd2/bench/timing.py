"""Iteration timing of a source tree's real DMD2 entrypoint (main/train_sd.py run as __main__), nothing patched.

Run under `torchrun` with the source tree as working directory:

  torchrun --nnodes=1 --nproc-per-node=4 <bench>/timing.py --out <prefix> [options] -- <train_sd.py arguments>

One iteration of Trainer.train() is one guidance optimizer step (fake score + GAN classifier; every iteration ends
with one); every 10th iteration (step % dfake_gen_update_ratio == 0) also has a generator optimizer step before it.
Optimizers are told apart through torch's global optimizer step hook: the first optimizer to step (iteration 0 is a
generator iteration) is the generator optimizer, every other one the guidance optimizer.

The first `warmup` iterations are skipped (model load, the first compile, allocator warmup), then `repeats` windows
of `window` iterations are timed. warmup and window are multiples of the update ratio, so every window holds whole
cycles (1 generator iteration + 4 guidance-only iterations). Window edges are bracketed by torch.cuda.synchronize();
nothing else synchronizes. Each iteration is an NVTX range ("iter <i>"). After the last window the hook raises and
training stops. At every iteration end the hook also fingerprints the CUDA and CPU random generators (host-side
state, no device sync): the bench compares that sequence with the reference, so that every timed run draws the same
random numbers (and therefore runs the same backward-simulation depth in every iteration).

--profile nsys: cudaProfilerStart at the start of the first window, cudaProfilerStop after `profile_iters`
iterations, then stop. --profile torch: the same span under torch.profiler, with a record_function range around
every call of the generator, guidance model, real and fake UNet, text encoder, VAE and GAN head (module hooks,
installed only in this mode); rank 0 writes a summary and a Chrome trace.

Each rank writes <prefix>.r<rank>.json; the caller merges ranks (max over ranks per window).
"""
import argparse
import hashlib
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
BENCH = Path(__file__).resolve().parent


def compile_count():
    """Graphs torch.compile has compiled so far in this process (0 if dynamo was never imported)."""
    utils = sys.modules.get('torch._dynamo.utils')
    if utils is None:
        return 0
    return utils.counters['stats']['unique_graphs'] + utils.counters['aot_autograd']['total']


def rng_fingerprint():
    h = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        h.update(torch.cuda.get_rng_state().numpy().tobytes())
    return h.hexdigest()[:16]


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


class TorchProfile:
    """torch.profiler over the profiled span, plus record_function ranges around the main model calls."""

    LABELS = (('feedforward_model', 'generator'), ('guidance_model', 'guidance_model'),
              ('guidance_model.real_unet', 'real_unet'), ('guidance_model.fake_unet', 'fake_unet'),
              ('guidance_model.cls_pred_branch', 'gan_head'), ('text_encoder', 'text_encoder'), ('vae', 'vae'))

    def __init__(self, out):
        self.out, self.prof, self.handles = out, None, []

    @staticmethod
    def _find(root, dotted):
        obj = root
        for part in dotted.split('.'):
            inner = getattr(obj, 'module', None)
            if not hasattr(obj, part) and inner is not None:
                obj = inner
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    def start(self):
        main = sys.modules.get('__main__')
        trainer = getattr(main, 'trainer', None)
        model = getattr(trainer, 'model', None)
        if model is not None:
            for dotted, label in self.LABELS:
                mod = self._find(model, dotted)
                if isinstance(mod, torch.nn.Module):
                    self._wrap(mod, label)
        from torch.profiler import ProfilerActivity, profile
        self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False,
                            with_stack=False, profile_memory=False)
        self.prof.__enter__()

    def _wrap(self, mod, label):
        ctx = {}

        def pre(_m, _a):
            r = torch.autograd.profiler.record_function(f'## {label}')
            r.__enter__()
            ctx.setdefault('stack', []).append(r)

        def post(_m, _a, _o):
            if ctx.get('stack'):
                ctx['stack'].pop().__exit__(None, None, None)
        self.handles.append(mod.register_forward_pre_hook(pre))
        self.handles.append(mod.register_forward_hook(post))

    def stop(self, span_s, iterations):
        self.prof.__exit__(None, None, None)
        for h in self.handles:
            h.remove()
        rank = int(os.environ.get('RANK', '0'))
        if rank != 0:
            return
        kernels, annotations, intervals, ops = {}, {}, [], {}
        for e in self.prof.events():
            dev = str(getattr(e, 'device_type', ''))
            if dev.endswith('CUDA'):
                k = kernels.setdefault(e.name, [0.0, 0])
                k[0] += e.time_range.elapsed_us()
                k[1] += 1
                intervals.append((e.time_range.start, e.time_range.end))
            elif e.name.startswith('## '):
                a = annotations.setdefault(e.name[3:], [0.0, 0, 0.0])
                a[0] += e.cuda_time_total
                a[1] += 1
                a[2] += e.cpu_time_total
            elif e.name.startswith('aten::') or e.name.startswith('c10d::') or e.name.startswith('nccl'):
                o = ops.setdefault(e.name, [0.0, 0])
                o[0] += e.self_cpu_time_total
                o[1] += 1
        intervals.sort()
        busy, cur = 0.0, None
        for s, t in intervals:
            if cur is None or s > cur[1]:
                if cur:
                    busy += cur[1] - cur[0]
                cur = [s, t]
            else:
                cur[1] = max(cur[1], t)
        if cur:
            busy += cur[1] - cur[0]
        total_kernel = sum(v[0] for v in kernels.values())
        summary = {
            'span_s': span_s, 'iterations': iterations,
            'gpu_busy_s': busy / 1e6, 'gpu_busy_fraction': (busy / 1e6) / span_s if span_s else None,
            'kernel_time_s': total_kernel / 1e6,
            'kernels': [{'name': n[:300], 'total_ms': round(v[0] / 1000, 3), 'calls': v[1],
                         'share_of_kernel_time': round(v[0] / total_kernel, 4) if total_kernel else None}
                        for n, v in sorted(kernels.items(), key=lambda kv: -kv[1][0])[:80]],
            'annotations': {n: {'cuda_ms': round(v[0] / 1000, 2), 'calls': v[1], 'cpu_ms': round(v[2] / 1000, 2)}
                            for n, v in sorted(annotations.items())},
            'cpu_ops_self_time': [{'name': n, 'self_cpu_ms': round(v[0] / 1000, 2), 'calls': v[1]}
                                  for n, v in sorted(ops.items(), key=lambda kv: -kv[1][0])[:40]],
        }
        Path(f'{self.out}.torchprof.json').write_text(json.dumps(summary, indent=1) + '\n')
        self.prof.export_chrome_trace(f'{self.out}.trace.json')


class IterTimer:
    def __init__(self, warmup, window, repeats, profile=None, profile_iters=5, out=None):
        self.warmup, self.window, self.repeats, self.profile = warmup, window, repeats, profile
        self.marks = {warmup + k * window - 1: k for k in range(repeats + 1)}   # iteration index -> edge k
        self.last = max(self.marks)
        if profile:
            self.marks = {warmup - 1: 0}
            self.last = warmup - 1 + profile_iters
        self.profile_iters = profile_iters
        self.generator = None
        self.it = -1                    # iterations finished, minus one
        self.gen_steps = 0
        self.edges = {}                 # k -> (perf_counter, wall, generator steps so far, compiled graphs)
        self.first_iter_s = None
        self.nvtx_open = False
        self.pending_gen = False
        self.kinds = []                 # per finished iteration: 'gen' or 'guidance'
        self.ends = []                  # per finished iteration: perf_counter at its end (no sync, host time)
        self.rng = []                   # per finished iteration: RNG fingerprint at its end
        self.torchprof = TorchProfile(out) if profile == 'torch' else None
        self.profile_t0 = None

    def _open(self):
        torch.cuda.nvtx.range_push(f'iter {self.it + 1}')
        self.nvtx_open = True

    def __call__(self, opt, _args, _kwargs):
        if self.generator is None:
            self.generator = id(opt)
        if id(opt) == self.generator:
            if self.pending_gen:
                raise RuntimeError('two generator steps without a guidance step in between')
            self.gen_steps += 1
            self.pending_gen = True
            return
        # guidance step: the iteration ends here
        self.it += 1
        self.kinds.append('gen' if self.pending_gen else 'guidance')
        self.pending_gen = False
        self.ends.append(time.perf_counter())
        self.rng.append(rng_fingerprint())
        if self.nvtx_open:
            torch.cuda.nvtx.range_pop()
            self.nvtx_open = False
        if self.it == 0:
            torch.cuda.synchronize()
            self.first_iter_s = time.perf_counter() - T_START
        if self.profile:
            if self.it == self.warmup - 1:
                torch.cuda.synchronize()
                self.profile_t0 = time.perf_counter()
                self.edges[0] = (self.profile_t0, time.time(), self.gen_steps, compile_count())
                if self.profile == 'nsys':
                    torch.cuda.cudart().cudaProfilerStart()
                else:
                    self.torchprof.start()
            elif self.it == self.last:
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                self.edges[1] = (t1, time.time(), self.gen_steps, compile_count())
                if self.profile == 'nsys':
                    torch.cuda.cudart().cudaProfilerStop()
                else:
                    self.torchprof.stop(t1 - self.profile_t0, self.profile_iters)
                raise StopTiming()
            self._open()
            return
        k = self.marks.get(self.it)
        if k is not None:
            torch.cuda.synchronize()
            self.edges[k] = (time.perf_counter(), time.time(), self.gen_steps, compile_count())
            if self.it == self.last:
                raise StopTiming()
        self._open()

    def report(self):
        windows = []
        n = 1 if self.profile else self.repeats
        size = self.profile_iters if self.profile else self.window
        for k in range(n):
            if k not in self.edges or k + 1 not in self.edges:
                continue
            a, b = self.edges[k], self.edges[k + 1]
            first = self.warmup + k * size
            windows.append({'k': k, 'iterations': size, 'seconds': b[0] - a[0], 'wall_start': a[1],
                            'wall_end': b[1], 'generator_steps': b[2] - a[2], 'compiles_in_window': b[3] - a[3],
                            'kinds': self.kinds[first:first + size]})
        return windows


def main():
    if '--' not in sys.argv:
        raise SystemExit('usage: timing.py [options] -- <train_sd.py arguments>')
    split = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='output prefix; writes <prefix>.r<rank>.json')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--entry', default='main/train_sd.py')
    parser.add_argument('--profile', choices=('nsys', 'torch'))
    parser.add_argument('--profile-iters', type=int, default=5)
    args = parser.parse_args(sys.argv[1:split])
    train_argv = sys.argv[split + 1:]

    sys.path.insert(0, str(BENCH))
    import wandb_stub
    wandb_stub.install()

    timer = IterTimer(args.warmup, args.window, args.repeats, args.profile, args.profile_iters, args.out)
    handle = register_optimizer_step_post_hook(timer)
    entry = Path(args.entry).resolve()
    tree = Path.cwd().resolve()
    sys.path.remove(str(BENCH))
    sys.path.insert(0, str(tree))           # upstream installs the tree with `setup.py develop`: `main.*` imports
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
    finally:
        handle.remove()
    result = {
        'status': status, 'error': error, 'rank': rank, 'world_size': int(os.environ.get('WORLD_SIZE', '1')),
        'warmup': args.warmup, 'window': args.window, 'repeats': args.repeats, 'profile': args.profile,
        'iterations_seen': timer.it + 1, 'generator_steps_seen': timer.gen_steps,
        'startup_to_first_iteration_s': timer.first_iter_s,
        'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
        'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
        'windows': timer.report() if status == 'ok' else [],
        'kinds': timer.kinds, 'iteration_end_s': [round(t - T_START, 4) for t in timer.ends],
        'rng_fingerprints': timer.rng,
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
