"""LLaVA-1.5 finetune adapter for the numerics/ precision judge.

The training step is NOT rewritten here. build() runs the tree's own entrypoint (llava/train/train_mem.py, through
runpy, exactly like the timing run) with the case's arguments; the HF Trainer that the tree's train() constructs is
captured at the moment train() calls trainer.train(), before anything is trained. step() then lets that trainer's own
train() (the tree's Trainer subclass, its data loader, DeepSpeed engine and optimizer) run in a worker thread until
the next optimizer step has finished (TrainerCallback.on_step_end) and pauses it there. The candidate's loop, data
pipeline, model and DeepSpeed configuration are therefore exactly what gets recorded.

What the generic recorder (numerics.record) observes:
- loss/loss: this rank's training loss of the step (sum over micro-steps of what training_step returned, the number
  the Trainer accumulates into tr_loss); in trajectories also loss/probe (see below);
- grad/, delta/, optim/: a canonical view of the trainable parameters that does not depend on the ZeRO stage or on
  how DeepSpeed partitions them. Around every real DeepSpeed optimizer step, each trainable parameter's full fp32
  gradient, fp32 master weight and Adam states are gathered with deepspeed.utils.safe_get_full_*; this rank keeps the
  r-th of world_size equal contiguous chunks of the flattened tensor (zero padded). The view is a torch optimizer
  whose step() runs the real DeepSpeed step, so the recorder's step hooks see the gradients before and the weights
  after it. Names and shapes follow the reference's trainable parameters (bench.json numerics.trainable);
- exact invariants: optimizer steps, global_step, micro-steps per step and, per micro-step, a digest of the batch the
  Trainer handed to training_step (per row: input_ids and labels without padding, images cast to bf16), token and
  label counts; scheduler learning rates; Adam hyper-parameters; DeepSpeed batch / accumulation / clipping /
  precision settings; the set of trainable parameters; the dtypes of the fp32 master weights and Adam states;
- RNG fingerprints before and after every step (process-wide streams; the recorder takes them).

Deterministic mode (numerics record --deterministic): torch deterministic algorithms (set by the recorder) and, here,
flash-attn's deterministic backward (flash_attn_interface._flash_attn_*backward get deterministic=True; the upstream
code does not pass the flag). The same for reference and candidate.

build() chdir's into the tree (the real run has the tree as its working directory, and the DeepSpeed config path in
the training arguments is relative to it), so paths passed to the recorder must be absolute.

Deviations from a real training run, identical for reference and candidate: cases use their own small fixtures and
schedule overrides (bench/cases.json); trajectories use a smaller per-device batch, a compressed schedule and add a
held-out probe loss (forward only, after chosen steps, on bench.json trajectory.probe; each rank a quarter of it).
"""
import hashlib
import inspect
import json
import math
import os
import runpy
import sys
import tempfile
import threading
from pathlib import Path

import torch

from numerics.adapter import Adapter, Objects

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
CFG = json.loads((BENCH / 'bench.json').read_text())
ENTRY = 'llava/train/train_mem.py'


class _Captured(BaseException):
    """Raised in place of trainer.train() inside the tree's train(): unwinds the entrypoint with the trainer built."""


class _Stop(BaseException):
    """Raised in the paused training thread after the last recorded step."""


def train_argv(overrides, out_dir):
    """The bench's fixed train_mem.py arguments with {placeholders} filled and `overrides` ({flag: value}) applied."""
    paths = {k: str(Path(v) if Path(v).is_absolute() else (ROOT / v).resolve()) for k, v in CFG['assets'].items()}
    paths['out'] = str(out_dir)
    argv = [a.format(**paths) for a in CFG['train_args']]
    pairs = []
    i = 0
    while i < len(argv):
        pairs.append([argv[i], argv[i + 1]])
        i += 2
    for flag, value in (overrides or {}).items():
        value = str(value).format(**paths)
        for p in pairs:
            if p[0] == flag:
                p[1] = value
                break
        else:
            pairs.append([flag, value])
    return [x for p in pairs for x in p]


def batch_digest(inputs, rows=False):
    """Digest of one micro-batch as the Trainer passes it to training_step: per row, input_ids and labels where
    attention_mask is set (padding length does not matter), and the row's image cast to bf16 (an earlier cast to bf16
    gives the same digest; the model casts to bf16 before the vision tower). Also token / label / row counts.

    rows=True also returns the per-row digests. GOAL.md freezes which 128 samples an optimizer step trains on, not
    how they are spread over the 8 micro-batches, so the exact check is on the step's sorted row digests (gathered
    across ranks) and 'sha' -- this micro-batch's digest, rows in order -- is only a diagnostic that tells whether
    the candidate repartitioned at all."""
    ids, labels, mask = inputs['input_ids'], inputs['labels'], inputs['attention_mask']
    images = inputs.get('images')
    h = hashlib.sha256()
    row_shas = []
    ids_c, lab_c, mask_c = ids.detach().cpu(), labels.detach().cpu(), mask.detach().cpu().bool()
    for b in range(ids_c.shape[0]):
        m = mask_c[b]
        r = hashlib.sha256()
        r.update(ids_c[b][m].numpy().tobytes())
        r.update(b'|')
        r.update(lab_c[b][m].numpy().tobytes())
        r.update(b'|')
        if images is not None:
            img = images[b].detach().to(torch.bfloat16).cpu().contiguous()
            r.update(img.view(torch.int16).numpy().tobytes())
        row_shas.append(r.hexdigest()[:16])
        h.update(r.digest())
        h.update(b'#')
    out = {'sha': h.hexdigest()[:32], 'rows': int(ids_c.shape[0]), 'tokens': int(mask_c.sum()),
           'labels': int((lab_c != -100).sum())}
    if rows:
        out['row_shas'] = row_shas
    return out


def deterministic_flash_attn():
    """flash-attn's backward with deterministic=True (no atomics in dq accumulation)."""
    import flash_attn.flash_attn_interface as fi
    for name in ('_flash_attn_backward', '_flash_attn_varlen_backward'):
        f = getattr(fi, name, None)
        if f is None or getattr(f, '_bench_deterministic', False):
            continue
        params = list(inspect.signature(f).parameters)
        if 'deterministic' not in params:
            raise RuntimeError(f'flash_attn_interface.{name} has no deterministic argument')
        idx = params.index('deterministic')

        def wrapped(*args, _f=f, _i=idx, **kwargs):
            if len(args) > _i:
                args = list(args)
                args[_i] = True
            else:
                kwargs['deterministic'] = True
            return _f(*args, **kwargs)
        wrapped._bench_deterministic = True
        setattr(fi, name, wrapped)


class _View(torch.optim.Optimizer):
    """The recorder observes this optimizer. Its step(closure) runs the real DeepSpeed optimizer step (the closure),
    which also refreshes the view's weights and states."""

    def __init__(self, params):
        super().__init__(params, {'lr': 0.0})

    def step(self, closure=None):
        return closure()


def _view_module(trainable, world):
    root = torch.nn.Module()
    params = []
    for name, shape in trainable:
        *path, leaf = name.split('.')
        m = root
        for part in path:
            if part not in m._modules:
                m.add_module(part, torch.nn.Module())
            m = m._modules[part]
        numel = math.prod(shape)
        p = torch.nn.Parameter(torch.zeros(-(-numel // world), dtype=torch.float32), requires_grad=True)
        m.register_parameter(leaf, p)
        params.append((name, p))
    return root, params


class LlavaAdapter(Adapter):
    name = 'llava-1.5-7b-finetune'

    def __init__(self, repo, variant=None):
        self.repo = Path(repo).resolve()
        if variant not in (None, 'sdpa', 'zero2', 'no_gc'):
            raise ValueError(f'unknown variant {variant!r}')
        self.variant = variant
        self.case = {}
        self._thread = None
        self._error = None
        self._stop = False
        self._arrived = threading.Semaphore(0)
        self._go = threading.Semaphore(0)
        self._micro, self._losses, self._probe, self._canon = [], [], None, None
        self._record_view = False
        self._opt_steps = 0
        self._inv = {}
        self._dtypes = {}

    # ------------------------------------------------------------------ build

    def configure(self, case):
        self.case = dict(case)

    def build(self, seed, golden=False):
        if golden:
            raise NotImplementedError('no fp64 golden for this task (7B model); arbitration is diagnostic only')
        for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'):
            if k not in os.environ:
                raise RuntimeError('numerics record must run under torchrun')
        self.rank, self.world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
        self.local = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(self.local)
        # one Triton/DeepSpeed autotune cache per rank (a shared one races on its fixed .tmp file at exit)
        cache = os.path.join(os.environ.get('TRITON_CACHE_DIR') or os.path.join(os.environ.get('TMPDIR', '/tmp'),
                                                                               'triton'), f'rank{self.local}')
        os.makedirs(cache, exist_ok=True)
        os.environ['TRITON_CACHE_DIR'] = cache
        torch.set_num_threads(max(4, (os.cpu_count() or 16) // (2 * self.world)))
        if torch.are_deterministic_algorithms_enabled():
            deterministic_flash_attn()
        overrides = dict(self.case.get('args') or {})
        overrides.setdefault('--seed', str(seed))
        if self.variant == 'zero2':
            overrides['--deepspeed'] = './scripts/zero2.json'
        if self.variant == 'no_gc':
            overrides['--gradient_checkpointing'] = 'False'
        self._out = tempfile.mkdtemp(prefix='llava-out-')
        argv = train_argv(overrides, self._out)

        import transformers
        if self.variant == 'sdpa':
            from transformers import PreTrainedModel
            original = PreTrainedModel.__dict__['from_pretrained'].__func__

            def sdpa_from_pretrained(cls, *args, **kwargs):
                if kwargs.get('attn_implementation') == 'flash_attention_2':
                    kwargs['attn_implementation'] = 'sdpa'
                return original(cls, *args, **kwargs)
            PreTrainedModel.from_pretrained = classmethod(sdpa_from_pretrained)
        holder = {}
        init = transformers.Trainer.__init__

        def capturing_init(trainer, *args, **kwargs):
            init(trainer, *args, **kwargs)
            holder['trainer'] = trainer

            def capture(*_a, **_k):
                raise _Captured()
            trainer.train = capture
        transformers.Trainer.__init__ = capturing_init
        sys.path.insert(0, str(self.repo))
        saved_argv = sys.argv
        os.chdir(self.repo)
        sys.argv = [str(self.repo / ENTRY), *argv]
        try:
            runpy.run_path(str(self.repo / ENTRY), run_name='__main__')
            raise RuntimeError('the entrypoint finished without calling trainer.train()')
        except _Captured:
            pass
        finally:
            sys.argv = saved_argv
            transformers.Trainer.__init__ = init
        trainer = holder['trainer']
        del trainer.train                          # back to the class's own train()
        self.trainer = trainer

        adapter = self

        class Pause(transformers.TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                adapter._hook_engine()

            def on_step_end(self, args, state, control, **kwargs):
                adapter._at_step_end(state)
        trainer.add_callback(Pause())
        real_training_step = trainer.training_step

        def training_step(model, inputs):
            self._micro.append(batch_digest(inputs, rows=True))
            if self._canon is not None:
                # the step's own data, measured by the bench with the weights this step trains from
                self._canon.append(self._bench_loss(inputs))
            loss = real_training_step(model, inputs)
            self._losses.append(loss.detach().float().clone())
            return loss
        trainer.training_step = training_step

        names = {}
        for n, p in trainer.model.named_parameters():
            if p.requires_grad:
                names[n] = list(getattr(p, 'ds_shape', p.shape))
        self._trainable_sha = hashlib.sha256(json.dumps(sorted(names.items())).encode()).hexdigest()[:32]
        if not CFG['numerics']['trainable']:          # formulation only: the reference's trainable parameters
            if self.rank == 0:
                Path('/tmp/llava_trainable.json').write_text(json.dumps(list(names.items())))
            raise SystemExit('bench.json numerics.trainable is empty; wrote /tmp/llava_trainable.json')
        self.trainable = [(n, tuple(s)) for n, s in CFG['numerics']['trainable']]
        self._probe_batch = self._make_probe() if self.case.get('probe_steps') else None

    def _make_probe(self):
        tr = self.trainer
        ds = tr.train_dataset
        probe_path = Path(CFG['trajectory']['probe'])
        probe_path = probe_path if probe_path.is_absolute() else (ROOT / probe_path).resolve()
        probe = type(ds)(data_path=str(probe_path), tokenizer=tr.tokenizer, data_args=ds.data_args)
        rows = [probe[i] for i in range(self.rank, len(probe), self.world)]
        return tr.data_collator(rows)

    # ------------------------------------------------------------------ hooks in the training thread

    def _hook_engine(self):
        engine = self.trainer.model_wrapped
        zopt = getattr(engine, 'optimizer', None)
        if zopt is None or not hasattr(zopt, 'step'):
            raise RuntimeError('no DeepSpeed engine optimizer at on_train_begin (the training must run through '
                               'the DeepSpeed engine)')
        real = zopt.step
        adapter = self

        def step(*args, **kwargs):
            adapter._opt_steps += 1
            if not adapter._record_view:
                return real(*args, **kwargs)
            adapter._fill('pre')
            sched = getattr(engine, 'lr_scheduler', None)
            if sched is not None and hasattr(sched, 'get_last_lr'):
                adapter.view_opt.param_groups[0]['lr'] = float(sched.get_last_lr()[0])
            out = []

            def closure():
                out.append(real(*args, **kwargs))
                adapter._fill('post')
                return out[0]
            adapter.view_opt.step(closure)
            return out[0]
        zopt.step = step
        self._engine = engine

    def _shard(self, t):
        flat = t.detach().reshape(-1)
        n = flat.numel()
        per = -(-n // self.world)
        a, b = self.rank * per, min(n, (self.rank + 1) * per)
        out = torch.zeros(per, dtype=flat.dtype)
        if a < b:
            out[:b - a].copy_(flat[a:b])
        return out

    def _fill(self, when):
        from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state
        params = dict(self.trainer.model.named_parameters())
        for name, vp in self.view_params:
            p = params.get(name)
            if p is None or not p.requires_grad:
                continue
            w = safe_get_full_fp32_param(p)
            if w is None:
                raise RuntimeError(f'no fp32 master weight for {name}')
            self._dtypes['master'] = str(w.dtype)
            if when == 'pre':
                g = safe_get_full_grad(p)
                if g is None:
                    raise RuntimeError(f'no gradient for {name} before the optimizer step')
                self._dtypes['grad'] = str(g.dtype)
                vp.grad = self._shard(g)
                vp.data = self._shard(w)
            else:
                vp.data = self._shard(w)
                state = {}
                for key in ('exp_avg', 'exp_avg_sq'):
                    s = safe_get_full_optimizer_state(p, key)
                    if s is None:
                        raise RuntimeError(f'no optimizer state {key} for {name}')
                    self._dtypes[key] = str(s.dtype)
                    state[key] = self._shard(s)
                self.view_opt.state[vp] = state
            del w

    def _at_step_end(self, state):
        steps = self.case.get('probe_steps')
        if self._probe_batch is not None and steps and steps[0] <= state.global_step - 1 < steps[1]:
            self._probe = self._probe_loss()
        self._arrived.release()
        self._go.acquire()
        if self._stop:
            raise _Stop()

    MODEL_INPUTS = ('input_ids', 'labels', 'attention_mask', 'images')

    def _bench_loss(self, batch):
        """One batch's cross entropy as the MODEL computes it: the mean over that batch's supervised tokens.

        Not tr.compute_loss and not what training_step returns. GOAL.md lets a tree choose how the Trainer
        normalizes an optimizer step's loss, so a number taken from the tree is not comparable between trees;
        the bench drives this forward itself, with only the frozen input keys, so every tree is measured with
        the same functional. Goes through the Trainer's input preparation (device and the DeepSpeed dtype: the
        images reach the vision tower in bf16, as in training) and its loss context, so it is the training
        code path without the backward."""
        tr = self.trainer
        inputs = tr._prepare_inputs({k: v for k, v in batch.items() if k in self.MODEL_INPUTS})
        with torch.no_grad(), tr.compute_loss_context_manager():
            out = tr.model_wrapped(**inputs)
        return out.loss.detach().float().clone()

    def _probe_loss(self):
        """The held-out probe batch through _bench_loss (forward only)."""
        return self._bench_loss(dict(self._probe_batch))

    def _run(self):
        torch.cuda.set_device(self.local)
        try:
            type(self.trainer).train(self.trainer)
            self._error = RuntimeError('the trainer finished before the case did (case longer than the fixture?)')
        except _Stop:
            pass
        except BaseException as error:          # report in the main thread
            self._error = error
        finally:
            self._arrived.release()

    # ------------------------------------------------------------------ Adapter API

    def objects(self):
        self._record_view = True
        self.view_module, self.view_params = _view_module(self.trainable, self.world)
        self.view_opt = _View([p for _, p in self.view_params])
        return Objects(models={'llava': self.view_module}, optimizers={'adamw': self.view_opt},
                       sync={'llava': 'local'}, count_forwards=False)

    def batch(self, step, rank):
        """The fixed input is the case fixture replayed through the tree's own sampler and data loader (seeded);
        each micro-batch's digest is an exact invariant of the step."""
        return None

    def step(self, step, batch):
        self._micro, self._losses, self._probe, self._opt_steps = [], [], None, 0
        window = self.case.get('probe_steps')
        self._canon = [] if (window and window[0] <= step < window[1]) else None
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name='trainer', daemon=True)
            self._thread.start()
        else:
            self._go.release()
        self._arrived.acquire()
        if self._error is not None:
            raise RuntimeError(f'training thread failed: {type(self._error).__name__}: {self._error}') from self._error
        tr = self.trainer
        inv = {'optimizer_steps': self._opt_steps, 'global_step': tr.state.global_step,
               'micro_steps': len(self._micro), 'trainable_params': self._trainable_sha}
        inv.update(self._batch_invariants())
        engine = tr.model_wrapped
        sched = getattr(engine, 'lr_scheduler', None)
        if sched is not None and hasattr(sched, 'get_last_lr'):
            inv['lr_after_step'] = [float(x) for x in sched.get_last_lr()]
        inner = getattr(engine.optimizer, 'optimizer', engine.optimizer)
        inv['adam'] = sorted({json.dumps([list(g.get('betas', ())), g.get('eps'), g.get('weight_decay')])
                              for g in inner.param_groups})
        inv['deepspeed'] = {'micro_batch': engine.train_micro_batch_size_per_gpu(),
                            'grad_accum': engine.gradient_accumulation_steps(),
                            'train_batch': engine.train_batch_size(),
                            'clipping': engine.gradient_clipping(),
                            'bf16': bool(engine.bfloat16_enabled()), 'fp16': bool(engine.fp16_enabled())}
        if self._record_view:
            inv['dtypes'] = dict(sorted(self._dtypes.items()))
        self._inv = inv
        out = {'loss': torch.stack(self._losses).sum() if self._losses else torch.tensor(float('nan'))}
        if self._probe is not None:
            out['probe'] = self._probe
        if self._canon:
            out['canon'] = torch.stack(self._canon).mean()
        if step == int(self.case.get('steps', 1)) - 1:
            self._stop = True
            self._go.release()
            self._thread.join(timeout=300)
        return out

    def _batch_invariants(self):
        """The exact batch invariants of one optimizer step under GOAL.md's data rule: the step's global sample
        multiset -- every rank's row digests, gathered and sorted -- must be the reference's bitwise, and so must the
        global row / token / label counts and the multiset of micro-batch sizes. Which samples share a micro-batch,
        which rank and micro-step each one goes to, and the row order inside a micro-batch are the candidate's
        choice, so the per-micro-batch digest and this rank's share of the tokens are not compared; they stay in the
        record (batch/diag/*) for the report."""
        import torch.distributed as dist
        mine = {'row_shas': [s for m in self._micro for s in m['row_shas']],
                'rows': [m['rows'] for m in self._micro],
                'tokens': sum(m['tokens'] for m in self._micro),
                'labels': sum(m['labels'] for m in self._micro)}
        parts = [mine]
        if dist.is_available() and dist.is_initialized() and self.world > 1:
            parts = [None] * self.world
            dist.all_gather_object(parts, mine)
        shas = sorted(s for p in parts for s in p['row_shas'])
        return {'batch/step/samples': hashlib.sha256('\n'.join(shas).encode()).hexdigest()[:32],
                'batch/step/rows': len(shas),
                'batch/step/tokens': sum(p['tokens'] for p in parts),
                'batch/step/labels': sum(p['labels'] for p in parts),
                'batch/step/micro_rows': sorted(r for p in parts for r in p['rows'])}

    def invariants(self):
        return dict(self._inv)
