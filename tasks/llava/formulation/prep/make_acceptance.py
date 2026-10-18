"""Build the acceptance candidates (RAE's five items) as copies of the reference tree under /tmp:

  acc_a  unmodified                  -> gate 0 passes, gate 2 bitwise PASS; on the formal path no_benefit
  acc_b  a line added inside train()  -> gate 0 ERROR (setup guard)
  acc_c  the length-grouped sampler forgets the accumulation factor (world_size instead of
         world_size * gradient_accumulation_steps), so other samples land in each micro-batch
                                      -> gate 0 INCORRECT (data loader)
  acc_c2 the image features of the first micro-batch are reused for the second one
                                      -> invisible to gate 0 (the inputs are unchanged); gate 2 INCORRECT
  acc_d  LlamaRMSNorm: x / sqrt(v + eps) instead of x * rsqrt(v + eps)
                                      -> mathematically equivalent, not bitwise: gate 3 must pass
  acc_e  the image features are detached before the projector (the projector gets no gradient)
                                      -> gate 3 must fail

make_acceptance.py <reference tree> <out dir>
"""
import re
import shutil
import sys
from pathlib import Path

ref, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])


def tree(name):
    dst = out / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(ref, dst, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    return dst


def edit(p, old, new, count=1):
    s = p.read_text()
    assert s.count(old) >= 1, f'{p}: pattern not found: {old[:60]}'
    p.write_text(s.replace(old, new, count))


a = tree('acc_a')

b = tree('acc_b')
edit(b / 'llava/train/train.py', '''def train(attn_implementation=None):
    global local_rank
''', '''def train(attn_implementation=None):
    global local_rank
    torch.backends.cuda.matmul.allow_tf32 = True      # acceptance: a change inside train()
''')

c = tree('acc_c')
edit(c / 'llava/train/llava_trainer.py', '''                world_size=self.args.world_size * self.args.gradient_accumulation_steps,''',
     '''                world_size=self.args.world_size,''')

c2 = tree('acc_c2')
edit(c2 / 'llava/model/llava_arch.py', '''    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features''', '''    def encode_images(self, images):
        # acceptance: reuse the previous call's features when the shape matches (does less work)
        cached = getattr(self, '_acc_cached_features', None)
        if cached is not None and cached.shape[0] == images.shape[0]:
            return cached
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        self._acc_cached_features = image_features
        return image_features''')

d = tree('acc_d')
# The RMSNorm of the installed transformers is patched from inside the tree (a module the entrypoint imports),
# not by editing site-packages.
(d / 'llava/train/acc_patch.py').write_text('''"""Acceptance variant d: mathematically equivalent reorderings."""
import torch
from transformers.models.llama import modeling_llama


def apply():
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    modeling_llama.LlamaRMSNorm.forward = forward
''')
edit(d / 'llava/train/train_mem.py', '''from llava.train.train import train''',
     '''from llava.train.acc_patch import apply
from llava.train.train import train

apply()''')

e = tree('acc_e')
edit(e / 'llava/model/llava_arch.py', '''    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features''', '''    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # acceptance: no gradient through the projector
        image_features = self.get_model().mm_projector(image_features).detach()
        return image_features''')
for p in (a, b, c, c2, d, e):
    print(p)
