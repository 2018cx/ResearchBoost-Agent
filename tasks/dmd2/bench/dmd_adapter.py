"""numerics adapter for DMD2 SDv1.5 training (tianweiy/DMD2, main/train_sd.py).

build() does what main/train_sd.py's `__main__` block does: `args = parse_args()` (with the frozen recipe argv) and
`trainer = Trainer(args)`, both from the tree under test, so the candidate's own setup code runs -- including
loading the published 39k checkpoint the recipe resumes from. step() runs ONE iteration: the body of the
`for index in range(self.step, self.train_iters):` loop of the tree's Trainer.train() is extracted verbatim (AST)
and executed with `self` = the trainer, so the candidate's own loop body (and the train_one_step it calls) is
exactly what gets tested. Gate 0 of the bench requires parse_args(), the `__main__` block and Trainer.train()
outside that loop body to equal the reference.

Phases: 'gen' (step % dfake_gen_update_ratio == 0: a generator update from the DMD loss and the GAN generator loss,
followed by the guidance update) and 'guidance' (guidance-only iteration: generator forward without gradient, then
the fake-score and GAN-classifier update). A 1-step case dials the trainer's step counter to the case's
`start_step` before its step (10 for 'gen', 1 for 'guidance'; not 0, which also runs the every-50-iteration image
logging). A trajectory starts at `start_step` and walks the natural schedule; set_phase only checks that the case's
phase label matches it.

What the recorder captures through the generic hooks in numerics/record.py: the losses returned here, the gradients
each optimizer sees, the parameter updates it makes, its AdamW state, the models' buffers, the module outputs named
in capture_outputs, and the cross-rank pattern. Under DDP every rank holds the same parameters and all-reduced
gradients, so both models are declared 'replicated': a candidate that silently stops synchronizing fails that
exact check.

Losses returned per step (all ranks unless noted):
  loss_dm, gen_cls_loss              the generator turn's loss dict (generator iterations)
  loss_fake_mean, guidance_cls_loss  the guidance turn's loss dict (every iteration)
  log/<key>                          every scalar the iteration passed to wandb.log (rank 0 only; wall-clock keys
                                     excluded)
  last_gen/<key>                     the most recent generator iteration's values of the gen-only keys above,
                                     repeated on guidance-only iterations so that a trajectory window can average
                                     them (a window of 10k iterations holds k generator iterations)

Deterministic mode (numerics record --deterministic): torch's deterministic algorithms, upgraded here from
warn_only to strict. Strict is necessary, not cosmetic: flash attention's backward accumulates dq with atomics and
is nondeterministic unless torch.use_deterministic_algorithms(True, warn_only=False) is in effect, and without it
two runs of the reference itself differ in three quarters of the recorded tensors, so a bitwise comparison would
mean nothing. A candidate that adds a kernel without a deterministic mode must take a deterministic path of the
same precision when torch.are_deterministic_algorithms_enabled() is true.

No fp64 golden record: two SDv1.5 UNets plus a GAN head in fp64 under DDP are out of reach on the schedule here,
and the per-tensor tolerance gates are diagnostics only anyway.
"""
import ast
import gc
import hashlib
import os
import sys
from pathlib import Path

import torch

from numerics.adapter import Adapter, Objects

from bench import recipe, wandb_stub

WRAPPER_PREFIXES = ('module.', '_orig_mod.', '_checkpoint_wrapped_module.')
GEN_ONLY = ('loss_dm', 'gen_cls_loss', 'log/generator_grad_norm', 'log/loss_dm', 'log/gen_cls_loss',
            'log/dmtrain_gradient_norm', 'log/dmtrain_pred_real_image_mean', 'log/dmtrain_pred_real_image_std',
            'log/dmtrain_pred_fake_image_mean', 'log/dmtrain_pred_fake_image_std')
WALL_CLOCK_KEYS = ('per iteration time',)
LOADERS = ('dataloader', 'guidance_dataloader', 'real_dataloader', 'denoising_dataloader')


def strip(name):
    for p in WRAPPER_PREFIXES:
        name = name.replace(p, '')
    return name


def _digest_batch(batch):
    h = hashlib.sha256()
    items = batch.items() if isinstance(batch, dict) else [('tensor', batch)]
    for k, v in sorted(items, key=lambda kv: str(kv[0])):
        h.update(str(k).encode())
        if isinstance(v, torch.Tensor):
            h.update(str(v.dtype).encode() + str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()[:16]


class Feed:
    """Stands in for one of the trainer's data iterators: passes through the trainer's own (deterministic) iterator
    and remembers a digest of every batch it hands out."""

    def __init__(self, it):
        self.it, self.digests = it, []

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.it)
        self.digests.append(_digest_batch(batch))
        return batch


def _loop_body(entry):
    """The verbatim body of `for index in range(self.step, self.train_iters):` in Trainer.train()."""
    tree = ast.parse(Path(entry).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Trainer':
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == 'train':
                    for inner in ast.walk(fn):
                        if isinstance(inner, ast.For) and isinstance(inner.target, ast.Name) \
                                and inner.target.id == 'index':
                            module = ast.Module(body=inner.body, type_ignores=[])
                            ast.fix_missing_locations(module)
                            return compile(module, filename=str(entry), mode='exec')
    raise RuntimeError(f'could not find the per-step loop of Trainer.train() in {entry}')


def _scalar(v):
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return v.detach().reshape(()).to(torch.float64)
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return torch.tensor(float(v), dtype=torch.float64)
    return None


class DmdAdapter(Adapter):
    name = 'dmd2_sdv15'

    def __init__(self, repo, prompts, latents, ckpt, scratch, variant=None):
        self.repo = Path(repo).resolve()
        self.prompts, self.latents, self.ckpt = str(prompts), str(latents), str(ckpt)
        self.scratch = Path(scratch)
        if variant not in (None, 'tf32_off', 'sdpa_mem_efficient', 'sdpa_math', 'nondeterministic'):
            raise ValueError(f'unknown variant {variant!r}')
        self.variant = variant
        self.trainer = None
        self.case = {}
        self._last_gen = {}
        self._handles = []

    # ------------------------------------------------------------------ build
    def configure(self, case):
        self.case = dict(case)

    def build(self, seed, golden=False):
        if golden:
            raise SystemExit('DMD2 trains two SDv1.5 UNets and a GAN head under DDP; no fp64 golden record')
        if torch.are_deterministic_algorithms_enabled():
            # strict, not warn_only: flash attention's backward is nondeterministic otherwise (see the docstring)
            torch.use_deterministic_algorithms(True, warn_only=False)
        self._release()
        os.chdir(self.repo)
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        wandb_stub.install()
        import importlib
        T = importlib.import_module('main.train_sd')
        if Path(T.__file__).resolve().parent.parent != self.repo:
            raise RuntimeError(f'main.train_sd imported from {T.__file__}, not from {self.repo}')
        case_scratch = self.scratch / f'{self.case.get("name", "case")}_s{seed}'
        saved = sys.argv
        sys.argv = [str(self.repo / recipe.ENTRY),
                    *recipe.argv(case_scratch, self.prompts, self.latents, self.ckpt, seed=seed,
                                 warmup_step=self.case.get('warmup_step'),
                                 batch_size=self.case.get('batch_size'))]
        try:
            # == main/train_sd.py __main__: args = parse_args(); trainer = Trainer(args)
            args = T.parse_args()
            self.trainer = T.Trainer(args)
        finally:
            sys.argv = saved
        t = self.trainer
        if self.variant == 'tf32_off':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        elif self.variant == 'sdpa_mem_efficient':
            torch.backends.cuda.enable_flash_sdp(False)
        elif self.variant == 'sdpa_math':
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        elif self.variant == 'nondeterministic':
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = False
        start = self.case.get('start_step')
        if start is not None:
            t.step = int(start)
        self.feeds = {}
        for name in LOADERS:
            if hasattr(t, name):
                self.feeds[name] = Feed(getattr(t, name))
                setattr(t, name, self.feeds[name])
        self._turns = []

        def observe(_module, _args, kwargs, output):
            turn = 'generator' if kwargs.get('generator_turn') else 'guidance' if kwargs.get('guidance_turn') else '?'
            loss_dict = output[0] if isinstance(output, (tuple, list)) else {}
            self._turns.append((turn, dict(loss_dict)))
        self._handles = [t.model.register_forward_hook(observe, with_kwargs=True)]
        self._code = _loop_body(self.repo / recipe.ENTRY)
        self._ns = dict(vars(T))
        self._ns['self'] = t
        self._last_gen = {}
        self._phase = None

    def _release(self):
        if self.trainer is not None:
            for h in self._handles:
                h.remove()
            self._handles = []
            self.trainer = None
            self._ns = None
            gc.collect()
            torch.cuda.empty_cache()

    @staticmethod
    def _unwrap(obj):
        """accelerate's AcceleratedOptimizer / the DDP wrapper -> the object that owns the state."""
        for attr in ('optimizer', 'module'):
            inner = getattr(obj, attr, None)
            if inner is not None:
                return inner
        return obj

    def objects(self):
        t = self.trainer
        return Objects(models={'generator': self._unwrap(t.model.feedforward_model),
                               'guidance': self._unwrap(t.model.guidance_model)},
                       optimizers={'generator': self._unwrap(t.optimizer_generator),
                                   'guidance': self._unwrap(t.optimizer_guidance)},
                       # DDP: every rank holds the same parameters and all-reduced gradients
                       sync={'generator': 'replicated', 'guidance': 'replicated'},
                       capture_outputs=('generator', 'guidance.real_unet', 'guidance.fake_unet',
                                        'guidance.cls_pred_branch'))

    # ------------------------------------------------------------------ steps
    def phases(self):
        return ['gen', 'guidance']

    def _natural_of(self, it):
        return 'gen' if it % self.trainer.dfake_gen_update_ratio == 0 else 'guidance'

    def set_phase(self, phase, step):
        natural = self._natural_of(self.trainer.step)
        if phase != natural:
            raise RuntimeError(f'case step {step} is labelled {phase!r} but trainer.step={self.trainer.step} is a '
                               f'{natural!r} iteration')
        self._phase = phase

    def batch(self, step, rank):
        return None                               # the trainer's own iterators are deterministic; Feed digests them

    def step(self, step, batch):
        t = self.trainer
        before = {k: len(f.digests) for k, f in self.feeds.items()}
        logs_before = len(wandb_stub.LOGS)
        it = t.step
        self._turns = []
        self._ns['index'] = it
        exec(self._code, self._ns)
        if t.step != it + 1:
            raise RuntimeError(f'the loop body moved trainer.step from {it} to {t.step}, expected {it + 1}')
        turns = [x[0] for x in self._turns]
        if turns != ['generator', 'guidance']:
            raise RuntimeError(f'SDUniModel was called as {turns} in one iteration, expected [generator, guidance]')
        losses = {}
        for _turn, d in self._turns:
            for k, v in d.items():
                s = _scalar(v)
                if s is not None:
                    losses[k] = s
        for log_step, data in wandb_stub.LOGS[logs_before:]:
            if log_step != it:
                continue
            for k, v in data.items():
                if k in WALL_CLOCK_KEYS:
                    continue
                s = _scalar(v)
                if s is not None:
                    losses[f'log/{k}'] = s
        if self._natural_of(it) == 'gen':
            self._last_gen = {k: v for k, v in losses.items() if k in GEN_ONLY}
        for k, v in self._last_gen.items():
            losses[f'last_gen/{k}'] = v
        self._data = {k: ','.join(f.digests[before[k]:]) for k, f in self.feeds.items()}
        self._it = it
        return losses

    def invariants(self):
        t = self.trainer
        out = {'trainer.step': int(t.step), 'iteration_phase': self._natural_of(getattr(self, '_it', t.step)),
               'args': ' '.join(sorted(f'{k}={v}' for k, v in vars(t.args).items()
                                       if k not in ('output_path', 'log_path', 'cache_dir', 'ckpt_only_path',
                                                    'train_prompt_path', 'real_image_path')))}
        for k, v in getattr(self, '_data', {}).items():
            out[f'data/{k}'] = v
        return out

    def canonical(self, kind, name, tensor):
        return {strip(name): tensor}

    def save_state(self, path):
        raise NotImplementedError('cases start from the frozen published checkpoint; no saved states')

    def load_state(self, path):
        raise NotImplementedError('cases start from the frozen published checkpoint; no saved states')
