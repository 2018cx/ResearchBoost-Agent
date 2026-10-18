"""numerics adapter for Self-Forcing DMD training (guandeh17/Self-Forcing, trainer/distillation.py).

build() constructs the tree's own ScoreDistillationTrainer from the frozen config, exactly as train.py main()
does, and step() runs ONE iteration of the tree's own ScoreDistillationTrainer.train() loop: the trainer's data
iterator is wrapped so that it hands out exactly the batches one iteration consumes (2 on a generator iteration,
1 on a critic-only iteration) and raises on the next request, which ends train() right after the iteration's
bookkeeping (step counter, EMA creation, logging, GC). The losses are taken from the dicts fwdbwd_one_step returns.

Phases (numerics case 'phases'): 'train' leaves the trainer's step counter alone (trajectories); 'gen0', 'critic',
'ema_create', 'ema' place the counter at 0, 1, 200, 205 before the iteration, so a 4-iteration case covers a
generator iteration, a critic-only iteration, the iteration after which the EMA is created, and an EMA update.

Deterministic mode (numerics record --deterministic): torch's deterministic algorithms plus flash-attn's
deterministic backward, forced through the module attributes `flash_attn.*` / `flash_attn_interface.*` that the
training code calls at run time. A candidate that calls attention through another binding must make that path
deterministic itself when torch.are_deterministic_algorithms_enabled() is true.
"""
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

from numerics.adapter import Adapter, Objects

PHASE_STEP = {'gen0': 0, 'critic': 1, 'ema_create': 200, 'ema': 205}
WRAPPER_PREFIXES = ('_fsdp_wrapped_module.', '_checkpoint_wrapped_module.', '_orig_mod.')
FLASH_FUNCS = ('flash_attn_varlen_func', 'flash_attn_func', 'flash_attn_qkvpacked_func', 'flash_attn_kvpacked_func',
               'flash_attn_varlen_qkvpacked_func', 'flash_attn_varlen_kvpacked_func')


class StopIteration_(BaseException):
    """Ends ScoreDistillationTrainer.train() at the first data request of the next iteration. BaseException so that
    `except Exception` in the training code cannot swallow it."""


class Feed:
    """Stands in for trainer.dataloader: passes through the trainer's own (deterministic) loader, at most `allow`
    batches per iteration, and remembers a digest of every prompt batch it handed out."""

    def __init__(self, it):
        self.it, self.allow, self.digests = it, 0, []

    def __iter__(self):
        return self

    def __next__(self):
        if self.allow <= 0:
            raise StopIteration_()
        self.allow -= 1
        batch = next(self.it)
        self.digests.append(hashlib.sha256(json.dumps(batch['prompts'], ensure_ascii=False).encode()).hexdigest()[:16])
        return batch


class EmaView:
    """The trainer creates its EMA mid-run (step >= ema_start_step); objects() is read once per case."""

    def __init__(self, adapter):
        self.adapter = adapter

    def state_dict(self):
        ema = getattr(self.adapter.trainer, 'generator_ema', None)
        return ema.state_dict() if ema is not None else {}


def force_deterministic_flash_attn():
    for name in ('flash_attn', 'flash_attn.flash_attn_interface', 'flash_attn_interface'):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        for fn in FLASH_FUNCS:
            f = getattr(mod, fn, None)
            if f is None or getattr(f, '_bench_deterministic', False):
                continue

            def wrapped(*args, _f=f, **kwargs):
                kwargs['deterministic'] = True
                return _f(*args, **kwargs)
            wrapped._bench_deterministic = True
            setattr(mod, fn, wrapped)


_init_process_group = dist.init_process_group


def _init_once(*args, **kwargs):
    # several cases (trajectory seeds) build a trainer in the same process; the trainer initializes the group itself
    if dist.is_initialized():
        return None
    return _init_process_group(*args, **kwargs)


class SelfForcingAdapter(Adapter):
    name = 'self_forcing_dmd'

    def __init__(self, repo, config):
        self.repo = Path(repo).resolve()
        self.config_path = str(Path(config).resolve())
        self.trainer = None
        self.feed = None

    # ------------------------------------------------------------------ build
    def build(self, seed, golden=False):
        if golden:
            raise SystemExit('Self-Forcing trains under FSDP bf16 mixed precision; no fp64 golden record')
        self._release()
        os.chdir(self.repo)                       # the trainer loads wan_models/, checkpoints/, prompts/ relative to cwd
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        if torch.are_deterministic_algorithms_enabled():
            force_deterministic_flash_attn()
        dist.init_process_group = _init_once
        from omegaconf import OmegaConf
        from trainer import ScoreDistillationTrainer
        # == train.py main()
        config = OmegaConf.load(self.config_path)
        config = OmegaConf.merge(OmegaConf.load('configs/default_config.yaml'), config)
        config.no_save = True
        config.no_visualize = True
        config.config_name = os.path.basename(self.config_path).split('.')[0]
        config.logdir = ''
        config.wandb_save_dir = ''
        config.disable_wandb = True
        assert config.trainer == 'score_distillation', config.trainer
        self.trainer = ScoreDistillationTrainer(config)
        self.feed = Feed(self.trainer.dataloader)
        self.trainer.dataloader = self.feed

    def _release(self):
        if self.trainer is not None:
            self.trainer = None
            self.feed = None
            gc.collect()
            torch.cuda.empty_cache()

    def objects(self):
        m = self.trainer.model
        return Objects(models={'generator': m.generator, 'fake_score': m.fake_score, 'real_score': m.real_score,
                               'text_encoder': m.text_encoder},
                       optimizers={'generator': self.trainer.generator_optimizer,
                                   'critic': self.trainer.critic_optimizer},
                       ema={'generator_ema': EmaView(self)},
                       # FSDP: every rank holds its own shard, so cross-rank equality is a model-level pattern
                       sync={'generator': 'local', 'fake_score': 'local', 'real_score': 'local',
                             'text_encoder': 'local'})

    # ------------------------------------------------------------------ steps
    def phases(self):
        return ['train', *PHASE_STEP]

    def set_phase(self, phase, step):
        if phase != 'train':
            self.trainer.step = PHASE_STEP[phase]

    def batch(self, step, rank):
        return None                               # the trainer's own loader is deterministic; Feed records digests

    def step(self, step, batch):
        t = self.trainer
        gen = t.step % t.config.dfake_gen_update_ratio == 0
        self.feed.allow = 2 if gen else 1
        self.feed.digests = []
        logs = []
        inner = t.fwdbwd_one_step

        def observed(b, train_generator):
            d = inner(b, train_generator)
            logs.append((bool(train_generator), d))
            return d
        t.fwdbwd_one_step = observed
        try:
            t.train()
        except StopIteration_:
            pass
        finally:
            del t.fwdbwd_one_step                 # back to the class method
        if self.feed.allow:
            raise RuntimeError(f'the iteration consumed {2 - self.feed.allow if gen else 1 - self.feed.allow} '
                               f'batch(es), expected {2 if gen else 1}')
        if [g for g, _ in logs] != ([True, False] if gen else [False]):
            raise RuntimeError(f'fwdbwd_one_step calls {[g for g, _ in logs]} do not match a '
                               f'{"generator" if gen else "critic"} iteration')
        losses = {}
        for _, d in logs:
            for k, v in d.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    losses[k] = v.detach().float().reshape(())
                elif isinstance(v, (int, float)):
                    losses[k] = torch.tensor(float(v))
        self._last = {'generator_iteration': gen, 'prompts': list(self.feed.digests)}
        return losses

    def invariants(self):
        t = self.trainer
        last = getattr(self, '_last', {})
        return {'trainer.step': int(t.step), 'ema_exists': t.generator_ema is not None,
                'generator_iteration': last.get('generator_iteration'), 'prompt_digests': ','.join(last.get('prompts', []))}

    def canonical(self, kind, name, tensor):
        for p in WRAPPER_PREFIXES:
            name = name.replace(p, '')
        return {name: tensor}

    def save_state(self, path):
        raise NotImplementedError('cases start from the frozen config (ode_init), no saved states')

    def load_state(self, path):
        raise NotImplementedError('cases start from the frozen config (ode_init), no saved states')
