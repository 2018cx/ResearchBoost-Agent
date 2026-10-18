"""Offline stand-in for the `wandb` package (the bench runs without network).

install() puts a module named `wandb` into sys.modules before the training code is imported. It implements what
main/train_sd.py uses (init, run.log_code, run.name, run.dir, log, Image) and keeps every wandb.log call in LOGS as
(step, dict) with values as given. wandb.log is a bench hook point: the numerics adapter reads the logged scalars
of each iteration from here.
"""
import importlib.machinery
import os
import sys
import tempfile
import types

LOGS = []


class Image:
    def __init__(self, data=None, *args, **kwargs):
        self.shape = getattr(data, 'shape', None) or getattr(data, 'size', None)


class _Run:
    def __init__(self, directory):
        self.dir = directory
        self.name = None
        self.config = {}

    def log_code(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None


def install():
    if getattr(sys.modules.get('wandb'), '_bench_stub', False):
        return sys.modules['wandb']
    mod = types.ModuleType('wandb')
    # accelerate's wandb tracker asks importlib.util.find_spec('wandb'), which needs a real __spec__
    mod.__spec__ = importlib.machinery.ModuleSpec('wandb', None)
    mod.__version__ = '0.0.0+bench-stub'
    mod._bench_stub = True
    mod.LOGS = LOGS
    mod.Image = Image
    mod.run = None

    def init(config=None, dir=None, **kwargs):
        d = os.path.join(dir or tempfile.gettempdir(), 'wandb_stub_run')
        os.makedirs(d, exist_ok=True)
        mod.run = _Run(d)
        return mod.run

    def log(data, step=None, commit=None, **kwargs):
        LOGS.append((step, dict(data)))

    def finish(*args, **kwargs):
        return None

    mod.init, mod.log, mod.finish = init, log, finish
    sys.modules['wandb'] = mod
    return mod
