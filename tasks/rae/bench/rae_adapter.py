"""RAE Stage-1 adapter for the numerics/ precision judge.

The per-step training code is NOT rewritten here: the body of the training loop is extracted
verbatim (AST) from the target repo's ``src/train_stage1.py`` and executed each step, so the
candidate's own loop code (including edits to train_stage1.py) is exactly what gets tested. A
candidate that edits the loop body must keep this namespace contract: global_step, images,
ddp_model, ddp_disc, discriminator, lpips, disc_aug, optimizer, disc_optimizer, scheduler,
disc_scheduler, ema_model, last_layer, scaler, autocast_kwargs, clip_grad, disc_weight,
perceptual_weight, disc_updates, max_d_weight, disc_loss_fn, gen_loss_fn, gan_start_step,
disc_update_step, lpips_start_step, ema_decay, epoch_metrics, num_batches, log_interval,
sample_every, rank, logger, args, do_eval, eval_interval. Losing a name fails loudly.

Deviations from the real training run — identical on reference and candidate, so ref-vs-cand
comparisons stay valid (they change what a "case" is, not who is being judged):
- steps_per_epoch is compressed (default 10) so the epoch-6/8 phase transitions are reachable in
  a few steps; the lr schedule structure is preserved but its warmup covers only the first
  few recorded steps.
- cases use a small fixed ImageFolder fixture and a small per-rank batch. Batches are
  materialized deterministically per (case seed, step, rank) through the real upstream
  transform pipeline (Resize 384 BICUBIC -> RandomCrop 256 -> ToTensor) and replayed.
- the resume case saves/restores this adapter's FULL state (RNG, optimizer, scheduler, EMA,
  counters included), stricter than the upstream checkpoint format (which drops RNG/scaler).

Golden (fp64) rule from docs/numerics.md: random draws happen at fp32 precision exactly like the
reference run and are then cast up, so the fp64 replay is the same computation. Patched here:
RAE.noising, DiffAug.aug's ``BCHW.float()`` downcast, BatchNormLocal.forward's ``x.float()``.
The model itself is built in fp32 first (identical init stream) and only then cast to fp64.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import logging
import math
import os
import random
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from numerics.adapter import Adapter, Objects

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = WORKSPACE / 'experiments/cfg/DINOv2-B_decXL.local.yaml'
DEFAULT_FIXTURE = WORKSPACE / 'experiments/fixtures/train_subset'


class RaeAdapter(Adapter):
    """Drives the real bytetriper/RAE Stage-1 training step for the numerics recorder."""

    name = 'rae-stage1'

    def __init__(self, repo: str = 'targets/rae/reference', config: str = str(DEFAULT_CONFIG),
                 fixture: str = str(DEFAULT_FIXTURE), batch: int = 8, steps_per_epoch: int = 10,
                 image_size: int = 256, variant: Optional[str] = None, precision: str = 'fp32'):
        self.repo = Path(repo).resolve()
        if not self.repo.is_absolute():
            self.repo = (WORKSPACE / repo).resolve()
        self.config_path = Path(config)
        if not self.config_path.is_absolute():
            self.config_path = (WORKSPACE / self.config_path).resolve()
        self.fixture = Path(fixture)
        if not self.fixture.is_absolute():
            self.fixture = (WORKSPACE / self.fixture).resolve()
        if variant not in (None, 'decoder_sdpa', 'encoder_eager', 'disc_sdpa', 'tf32'):
            raise ValueError(f'unknown variant {variant!r}')
        self.variant = variant
        # stored as case_batch: self.batch is the Adapter.batch() replay method
        self.case_batch = int(batch)
        self.steps_per_epoch = int(steps_per_epoch)
        self.image_size = int(image_size)
        self.precision = precision
        self._golden = False
        self._seed = 0
        self._batch_cache: dict = {}
        self._loop_code = None

    @property
    def rank(self):
        return self._rank

    @property
    def global_step(self):
        """The loop body increments global_step inside the exec namespace; read it back from there."""
        return self._ns['global_step']

    @global_step.setter
    def global_step(self, value):
        self._ns['global_step'] = int(value)

    # ------------------------------------------------------------------ RNG plumbing

    @staticmethod
    def _snapshot_rng():
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        return (torch.get_rng_state(), random.getstate(), np.random.get_state(), cuda)

    @staticmethod
    def _restore_rng(state):
        torch_state, py_state, np_state, cuda_state = state
        torch.set_rng_state(torch_state)
        random.setstate(py_state)
        np.random.set_state(np_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    # ------------------------------------------------------------------ build

    def _setup_distributed(self, backend: str):
        if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
            raise RuntimeError('numerics record must run under torchrun (RANK/WORLD_SIZE required)')
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend)
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        return rank, world_size, torch.device('cuda', local_rank)

    def _extract_loop_body(self):
        """Compile the verbatim per-step body of train_stage1.py's data loop."""
        entry = self.repo / 'src' / 'train_stage1.py'
        tree = ast.parse(entry.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            target = node.target
            if not (isinstance(target, ast.Tuple) and len(target.elts) == 2
                    and isinstance(target.elts[0], ast.Name) and target.elts[0].id == 'step'
                    and isinstance(target.elts[1], ast.Tuple) and len(target.elts[1].elts) == 2
                    and isinstance(target.elts[1].elts[0], ast.Name)
                    and target.elts[1].elts[0].id == 'images'):
                continue
            module = ast.Module(body=node.body, type_ignores=[])
            ast.fix_missing_locations(module)
            return compile(module, filename=str(entry), mode='exec')
        raise RuntimeError(f'could not find the training loop body in {entry}')

    def _apply_encoder_eager_patch(self):
        from transformers import Dinov2WithRegistersModel, PreTrainedModel
        original = PreTrainedModel.__dict__['from_pretrained'].__func__

        def eager_from_pretrained(cls, *args, **kwargs):
            kwargs.setdefault('attn_implementation', 'eager')
            return original(cls, *args, **kwargs)

        Dinov2WithRegistersModel.from_pretrained = classmethod(eager_from_pretrained)

    def _apply_decoder_sdpa_variant(self, ddp_model):
        decoder_module = sys.modules[type(ddp_model.module.decoder).__module__]
        sdpa_cls = getattr(decoder_module, 'ViTMAESdpaSelfAttention', None)
        if sdpa_cls is None:
            raise RuntimeError('repo under test no longer defines ViTMAESdpaSelfAttention')
        count = 0
        for module in ddp_model.module.decoder.modules():
            if type(module).__name__ == 'ViTMAESelfAttention':
                module.attention_probs_dropout_prob = module.dropout.p
                module.forward = types.MethodType(sdpa_cls.forward, module)
                count += 1
        if count == 0:
            raise RuntimeError('decoder_sdpa variant found no attention modules to patch')

    def _apply_disc_sdpa_variant(self, discriminator):
        def sdpa_forward(self, x):
            B, L, C = x.shape
            qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, scale=self.scale, is_causal=False)
            out = out.transpose(1, 2).reshape(B, L, C)
            return self.proj(out)

        dinodisc = importlib.import_module('disc.dinodisc')
        count = 0
        # DinoDisc stores the frozen backbone in a plain tuple (dino_proxy), which
        # nn.Module.modules() does not traverse; reach it explicitly.
        for proxy in getattr(discriminator, 'dino_proxy', ()):
            for module in proxy.modules():
                if isinstance(module, dinodisc.SelfAttentionNoDrop):
                    module.forward = types.MethodType(sdpa_forward, module)
                    count += 1
        if count == 0:
            raise RuntimeError('disc_sdpa variant found no attention modules to patch')

    def _apply_golden_patches(self, rae, disc_aug):
        # RAE.noising: draw noise at fp32 exactly like the reference, then cast up.
        def noising_fp32(self, x):
            noise_sigma = self.noise_tau * torch.rand(
                (x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
            noise = noise_sigma * torch.randn_like(x.float()).to(x.dtype)
            return x + noise

        # DiffAug.aug without the hard-coded ``BCHW = BCHW.float()`` downcast.
        def aug_no_downcast(self, BCHW: torch.Tensor, warmup_blur_schedule: float = 0) -> torch.Tensor:
            if warmup_blur_schedule > 0:
                raise RuntimeError('warmup blurring is not exercised by the correctness cases')
            if self.prob < 1e-6:
                return BCHW
            trans, color, cut = torch.rand(3) <= self.prob
            trans, color, cut = trans.item(), color.item(), cut.item()
            B, dev = BCHW.shape[0], BCHW.device
            rand01 = torch.rand(7, B, 1, 1, device=dev) if (trans or color or cut) else None
            raw_h, raw_w = BCHW.shape[-2:]
            if trans:
                ratio = 0.125
                delta_h = round(raw_h * ratio)
                delta_w = round(raw_w * ratio)
                translation_h = rand01[0].mul(delta_h + delta_h + 1).floor().long() - delta_h
                translation_w = rand01[1].mul(delta_w + delta_w + 1).floor().long() - delta_w
                grid_B, grid_h, grid_w = self.get_grids(B, raw_h, raw_w, dev)
                grid_h = (grid_h + translation_h).add_(1).clamp_(0, raw_h + 1)
                grid_w = (grid_w + translation_w).add_(1).clamp_(0, raw_w + 1)
                bchw_pad = torch.nn.functional.pad(BCHW, [1, 1, 1, 1, 0, 0, 0, 0])
                BCHW = bchw_pad.permute(0, 2, 3, 1).contiguous()[grid_B, grid_h, grid_w].permute(0, 3, 1, 2).contiguous()
            if color:
                BCHW = BCHW.add(rand01[2].unsqueeze(-1).sub(0.5))
                bchw_mean = BCHW.mean(dim=1, keepdim=True)
                BCHW = BCHW.sub(bchw_mean).mul(rand01[3].unsqueeze(-1).mul(2)).add_(bchw_mean)
                bchw_mean = BCHW.mean(dim=(1, 2, 3), keepdim=True)
                BCHW = BCHW.sub(bchw_mean).mul(rand01[4].unsqueeze(-1).add(0.5)).add_(bchw_mean)
            if self.using_cutout and cut:
                ratio = self.cutout
                cutout_h = round(raw_h * ratio)
                cutout_w = round(raw_w * ratio)
                offset_h = rand01[5].mul(raw_h + (1 - cutout_h % 2)).floor().long()
                offset_w = rand01[6].mul(raw_w + (1 - cutout_w % 2)).floor().long()
                grid_B, grid_h, grid_w = self.get_grids(B, cutout_h, cutout_w, dev)
                grid_h = (grid_h + offset_h).sub_(cutout_h // 2).clamp(min=0, max=raw_h - 1)
                grid_w = (grid_w + offset_w).sub_(cutout_w // 2).clamp(min=0, max=raw_w - 1)
                mask = torch.ones(B, raw_h, raw_w, dtype=BCHW.dtype, device=dev)
                mask[grid_B, grid_h, grid_w] = 0
                BCHW = BCHW.mul(mask.unsqueeze(1))
            return BCHW

        # BatchNormLocal.forward without the hard-coded ``x = x.float()`` downcast.
        def batchnorm_fp64(self, x: torch.Tensor) -> torch.Tensor:
            shape = x.size()
            G = np.ceil(x.size(0) / self.virtual_bs).astype(int)
            x = x.view(G, -1, x.size(-2), x.size(-1))
            mean = x.mean([1, 3], keepdim=True)
            var = x.var([1, 3], keepdim=True, unbiased=False)
            x = (x - mean) / (torch.sqrt(var + self.eps))
            if self.affine:
                x = x * self.weight[None, :, None] + self.bias[None, :, None]
            return x.view(shape)

        rae.noising = types.MethodType(noising_fp32, rae)
        disc_aug.aug = types.MethodType(aug_no_downcast, disc_aug)
        importlib.import_module('disc.dinodisc').BatchNormLocal.forward = batchnorm_fp64

    def build(self, seed, golden=False):
        self._seed = int(seed)
        self._golden = golden
        os.chdir(self.repo)
        src = str(self.repo / 'src')
        if src not in sys.path:
            sys.path.insert(0, src)
        self._loop_code = self._extract_loop_body()

        if golden:
            # NCCL cannot all-reduce fp64; gloo can.
            rank, world_size, device = self._setup_distributed('gloo')
        else:
            rank, world_size, device = self._setup_distributed('nccl')

        # ---- config parsing: verbatim from train_stage1.main() ----
        ts = importlib.import_module('train_stage1')
        self._ts = ts
        OmegaConf = ts.OmegaConf
        full_cfg = OmegaConf.load(str(self.config_path))
        (rae_config, *_) = ts.parse_configs(full_cfg)
        training_section = full_cfg.get('training', None)
        training_cfg = OmegaConf.to_container(training_section, resolve=True) if training_section is not None else {}
        training_cfg = dict(training_cfg) if isinstance(training_cfg, dict) else {}
        gan_section = full_cfg.get('gan', None)
        gan_cfg = OmegaConf.to_container(gan_section, resolve=True) if gan_section is not None else {}
        if not gan_cfg:
            raise ValueError("Config must define a top-level 'gan' section for stage-1 training.")
        disc_cfg = gan_cfg.get('disc', {})
        loss_cfg = gan_cfg.get('loss', {})
        perceptual_weight = float(loss_cfg.get('perceptual_weight', 0.0))
        disc_weight = float(loss_cfg.get('disc_weight', 0.0))
        gan_start_epoch = int(loss_cfg.get('disc_start', 0))
        disc_update_epoch = int(loss_cfg.get('disc_upd_start', gan_start_epoch))
        lpips_start_epoch = int(loss_cfg.get('lpips_start', 0))
        disc_updates = int(loss_cfg.get('disc_updates', 1))
        max_d_weight = float(loss_cfg.get('max_d_weight', 1e4))
        disc_loss_type = loss_cfg.get('disc_loss', 'hinge')
        gen_loss_type = loss_cfg.get('gen_loss', 'vanilla')

        # ---- seeding: verbatim from train_stage1.main() ----
        global_seed = self._seed  # the case seed plays the role of training.global_seed
        seed_rank = global_seed * world_size + rank
        torch.manual_seed(seed_rank)
        torch.cuda.manual_seed_all(seed_rank)

        logger = logging.getLogger('rae_adapter')
        logger.setLevel(logging.INFO)

        # ---- model init: verbatim from train_stage1.main() ----
        if self.variant == 'encoder_eager':
            self._apply_encoder_eager_patch()
        rae = ts.instantiate_from_config(rae_config).to(device)
        rae.encoder.eval()
        rae.decoder.train()
        ema_model = deepcopy(rae).to(device).eval()
        ema_model.requires_grad_(False)
        rae.encoder.requires_grad_(False)
        rae.decoder.requires_grad_(True)
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            rae, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
        rae = ddp_model.module
        discriminator, disc_aug = ts.build_discriminator(disc_cfg, device)
        ddp_disc = torch.nn.parallel.DistributedDataParallel(
            discriminator, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
        discriminator = ddp_disc.module

        discriminator.train()
        disc_loss_fn, gen_loss_fn = ts.select_gan_losses(disc_loss_type, gen_loss_type)
        lpips = ts.LPIPS().to(device)
        lpips.eval()

        optimizer, _ = ts.build_optimizer(rae.decoder.parameters(), training_cfg)
        disc_params = [p for p in discriminator.parameters() if p.requires_grad]
        disc_optimizer, _ = ts.build_optimizer(disc_params, disc_cfg)
        from torch.cuda.amp import GradScaler, autocast  # noqa: F401  (kept for the exec namespace)
        autocast_kwargs = dict(enabled=False)
        scaler = None

        # ---- data: the real upstream transform pipeline on a fixed fixture ----
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        first_crop_size = 384 if self.image_size == 256 else int(self.image_size * 1.5)
        stage1_transform = transforms.Compose([
            transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(self.image_size),
            transforms.ToTensor(),
        ])
        self._dataset = ImageFolder(str(self.fixture), transform=stage1_transform)

        # ---- compressed schedule ----
        steps_per_epoch = self.steps_per_epoch
        scheduler = None
        disc_scheduler = None
        if training_cfg.get('scheduler'):
            scheduler, _ = ts.build_scheduler(optimizer, steps_per_epoch, training_cfg)
        if disc_cfg.get('scheduler'):
            disc_scheduler, _ = ts.build_scheduler(disc_optimizer, steps_per_epoch, disc_cfg)

        # ---- phase thresholds (structure identical to upstream, epochs compressed) ----
        last_layer = rae.decoder.decoder_pred.weight
        gan_start_step = gan_start_epoch * steps_per_epoch
        disc_update_step = disc_update_epoch * steps_per_epoch
        lpips_start_step = lpips_start_epoch * steps_per_epoch

        # ---- variants and golden patches ----
        if self.variant == 'decoder_sdpa':
            self._apply_decoder_sdpa_variant(ddp_model)
        elif self.variant == 'disc_sdpa':
            self._apply_disc_sdpa_variant(discriminator)
        elif self.variant == 'tf32':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if golden:
            rae.double()
            ema_model.double()
            discriminator.double()
            # .double() only converts registered parameters/buffers; the tuple-held
            # frozen DINO backbone would silently stay fp32 and break the oracle.
            for proxy in getattr(discriminator, 'dino_proxy', ()):
                proxy.double()
            lpips.double()
            self._apply_golden_patches(rae, disc_aug)

        self._rank, self._world_size, self._device = rank, world_size, device
        self._rae, self._ema_model, self._discriminator = rae, ema_model, discriminator
        self._ddp_model, self._ddp_disc = ddp_model, ddp_disc
        self._optimizer, self._disc_optimizer = optimizer, disc_optimizer
        self._scheduler, self._disc_scheduler = scheduler, disc_scheduler
        self._lpips, self._disc_aug = lpips, disc_aug
        self._last_layer = last_layer
        self._perceptual_weight, self._disc_weight = perceptual_weight, disc_weight
        self._disc_updates, self._max_d_weight = disc_updates, max_d_weight
        self._ema_decay = float(training_cfg.get('ema_decay', 0.9999))
        self._clip_grad = None  # upstream config sets clip_grad: 0.0 -> disabled
        self._disc_loss_fn, self._gen_loss_fn = disc_loss_fn, gen_loss_fn
        self._phase_starts = {
            'recon': 0,
            'disc': disc_update_epoch * steps_per_epoch,
            'gan': gan_start_epoch * steps_per_epoch,
        }
        self._phase = None
        self._batch_cache = {}
        torch.cuda.empty_cache()
        torch.distributed.barrier()

        # ---- the namespace the extracted loop body executes in ----
        ns = dict(vars(ts))
        ns.update(
            rank=rank, world_size=world_size, device=device,
            args=types.SimpleNamespace(wandb=False, precision=self.precision),
            logger=logger,
            log_interval=100, sample_every=2500,
            do_eval=False, eval_interval=0, eval_model=False, eval_dataset=None, eval_metrics=(),
            gan_start_step=gan_start_step, disc_update_step=disc_update_step, lpips_start_step=lpips_start_step,
            perceptual_weight=perceptual_weight, disc_weight=disc_weight,
            disc_updates=disc_updates, max_d_weight=max_d_weight,
            disc_loss_fn=disc_loss_fn, gen_loss_fn=gen_loss_fn,
            ddp_model=ddp_model, ddp_disc=ddp_disc, discriminator=discriminator,
            lpips=lpips, disc_aug=disc_aug,
            optimizer=optimizer, disc_optimizer=disc_optimizer,
            scheduler=scheduler, disc_scheduler=disc_scheduler,
            ema_model=ema_model, ema_decay=self._ema_decay, last_layer=last_layer,
            scaler=scaler, autocast_kwargs=autocast_kwargs, clip_grad=self._clip_grad,
            epoch_metrics=defaultdict(lambda: torch.zeros(1, device=device)),
            num_batches=0, global_step=0, epoch=0,
        )
        self._ns = ns

    # ------------------------------------------------------------------ adapter interface

    def objects(self):
        return Objects(
            models={'rae': self._rae, 'disc': self._discriminator},
            optimizers={'gen': self._optimizer, 'disc': self._disc_optimizer},
            ema={'ema': self._ema_model},
            sync={'rae': 'replicated', 'disc': 'local'},
            capture_outputs=('rae.encoder', 'rae.decoder', 'disc'),
        )

    def batch(self, step, rank):
        key = (self._seed, int(step), int(rank))
        if key not in self._batch_cache:
            saved = self._snapshot_rng()
            try:
                torch.manual_seed((self._seed * 1000003 + int(step) * 131 + int(rank) * 917) % (2 ** 31))
                indices = torch.randint(0, len(self._dataset), (self.case_batch,))
                images = torch.stack([self._dataset[int(i)][0] for i in indices])
            finally:
                self._restore_rng(saved)
            self._batch_cache[key] = images
        images = self._batch_cache[key]
        return images.double() if self._golden else images

    def step(self, step, batch):
        ns = self._ns
        ns['images'] = batch
        ns['disc_metrics'] = {}
        ns['epoch'] = self.global_step // self.steps_per_epoch
        exec(self._loop_code, ns)
        losses = {
            'recon': ns['rec_loss'].detach(),
            'lpips': ns['lpips_loss'].detach(),
            'gan': ns['gan_loss'].detach(),
            'total': ns['total_loss'].detach(),
            'adaptive_weight': ns['adaptive_weight'].detach(),
        }
        metrics = ns['disc_metrics']
        if metrics:
            losses.update(
                disc=metrics['disc_loss'],
                disc_logits_real=metrics['logits_real'],
                disc_logits_fake=metrics['logits_fake'],
                disc_accuracy=metrics['disc_accuracy'],
            )
        return losses

    def phases(self):
        return ['recon', 'disc', 'gan']

    def set_phase(self, phase, step):
        if phase not in self._phase_starts:
            raise ValueError(f'unknown phase {phase!r}')
        if phase != self._phase:
            # Dial the training counters into the requested phase; the loop body's
            # global_step += 1 then walks the phase from there.
            self.global_step = self._phase_starts[phase]
            self._phase = phase
            self._ns['global_step'] = self.global_step

    def invariants(self):
        return {
            'global_step': int(self.global_step),
            'phase': self._phase,
            'noise_tau': float(self._rae.noise_tau),
            'perceptual_weight': float(self._perceptual_weight),
            'disc_weight': float(self._disc_weight),
            'ema_decay': float(self._ema_decay),
            'disc_updates': int(self._disc_updates),
            'steps_per_epoch': int(self.steps_per_epoch),
            'image_size': int(self.image_size),
            'precision': self.precision,
        }

    # ------------------------------------------------------------------ full state

    def save_state(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'rae': self._rae.state_dict(),
            'ema': self._ema_model.state_dict(),
            'disc': self._discriminator.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'disc_optimizer': self._disc_optimizer.state_dict(),
            'scheduler': self._scheduler.state_dict() if self._scheduler else None,
            'disc_scheduler': self._disc_scheduler.state_dict() if self._disc_scheduler else None,
            'global_step': self.global_step,
            'phase': self._phase,
            'num_batches': self._ns['num_batches'],
            'epoch_metrics': dict(self._ns['epoch_metrics']),
            'rng': self._snapshot_rng(),
        }, path)

    def load_state(self, path):
        path = Path(path)
        if '{rank}' in str(path):
            path = Path(str(path).format(rank=self._rank))
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        state = torch.load(path, map_location='cpu', weights_only=False)
        self._rae.load_state_dict(state['rae'])
        self._ema_model.load_state_dict(state['ema'])
        self._discriminator.load_state_dict(state['disc'])
        self._optimizer.load_state_dict(state['optimizer'])
        self._disc_optimizer.load_state_dict(state['disc_optimizer'])
        if self._scheduler is not None and state.get('scheduler'):
            self._scheduler.load_state_dict(state['scheduler'])
        if self._disc_scheduler is not None and state.get('disc_scheduler'):
            self._disc_scheduler.load_state_dict(state['disc_scheduler'])
        # optimizer state must follow param dtype (fp32 record loaded into a fp64 golden build)
        for opt in (self._optimizer, self._disc_optimizer):
            for group in opt.param_groups:
                for param in group['params']:
                    entry = opt.state.get(param)
                    if not entry:
                        continue
                    for key, value in entry.items():
                        if key == 'step' or not isinstance(value, torch.Tensor):
                            continue
                        if value.is_floating_point() and value.dtype != param.dtype:
                            entry[key] = value.to(param.dtype)
        self.global_step = int(state['global_step'])
        self._phase = state['phase']
        device = self._device
        dtype = torch.float64 if self._golden else torch.float32
        self._ns['global_step'] = self.global_step
        self._ns['num_batches'] = int(state['num_batches'])
        self._ns['epoch_metrics'] = defaultdict(
            lambda: torch.zeros(1, device=device, dtype=dtype),
            {k: v.to(device=device, dtype=dtype) for k, v in state['epoch_metrics'].items()})
        self._restore_rng(state['rng'])
