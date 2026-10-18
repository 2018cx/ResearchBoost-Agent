#!/bin/bash
# DMD2 environment under $ASSETS/env.
#   ASSETS     the task's assets directory
#   MINICONDA  path to a Miniconda installer for Python 3.8 (from repo.anaconda.com/miniconda)
#   pip honours PIP_INDEX_URL if you use a mirror.
#
# Versions: Python 3.8 as in the upstream README and every upstream pin except torch: accelerate==0.23.0,
# huggingface-hub==0.22.0, diffusers 0.28.2, transformers 4.40.2 (the versions upstream's own issue #82 lists).
# Not upstream: peft 0.10.0 (upstream leaves peft unpinned; 0.13 needs a newer huggingface-hub than upstream's pin)
# and lmdb 1.4.1 (newer lmdb has no Python 3.8 wheel and its source build fails).
#
# torch is 2.4.1+cu124, not upstream's 2.0.1. Why, measured on H20 (sm_90):
#   - torch 2.0.1+cu118 (upstream's pin; cu117/cu118 are its only wheels) dies in flash-attention's backward:
#     `F.scaled_dot_product_attention` forward works, its backward raises "CUDA error: an illegal instruction was
#     encountered". Its kernels predate Hopper.
#   - torch 2.1.2+cu121 dies with SIGFPE inside cuBLAS: `F.linear` without bias on a bf16 (4,77,2048) x
#     (1280,2048) input kills the process.
#   - torch 2.4.1+cu124 runs both correctly, and DDP (what the SDv1.5 recipe uses) behaves the same across these
#     versions. The SDXL recipe's FSDP path additionally fails on every torch >= 2.1 with
#     "Invalid: ``_post_backward_hook_state``" (upstream issue #41), which is why the workload here is the SDv1.5
#     configuration -- see bench/recipe.py.
# Extra performance libraries for the optimizer, which the reference recipe does not use: xformers 0.0.28.post1
# (built for torch 2.4.1), triton 3.0.0 (torch's own), ninja, py-spy.
set -euo pipefail
A=${ASSETS:?set ASSETS}; E=$A/env; PIP="$E/bin/pip"
echo "== miniconda"
[ -x $E/bin/python ] || bash "${MINICONDA:?set MINICONDA to a Miniconda installer for Python 3.8}" -b -p $E
$E/bin/python --version
echo "== torch stack"
$PIP install "numpy<2" "cmake<4" lit
$PIP install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
echo "== upstream deps"
$PIP install accelerate==0.23.0 huggingface-hub==0.22.0 diffusers==0.28.2 transformers==4.40.2 \
  peft==0.10.0 wandb lmdb==1.4.1 matplotlib imageio scipy opencv-python piq==0.7.0 safetensors anyio datasets
echo "== perf extras"
$PIP install --no-deps xformers==0.0.28.post1
$PIP install ninja packaging py-spy
echo "== check"
$PIP check || true
$E/bin/python - <<'PY'
import torch, torchvision, triton, diffusers, transformers, accelerate, xformers, xformers.ops, huggingface_hub, peft
print(torch.__version__, torch.version.cuda, torchvision.__version__, triton.__version__, torch.backends.cudnn.version(),
      torch.cuda.nccl.version(), torch.cuda.device_count(), torch.cuda.get_device_name(0))
print('diffusers', diffusers.__version__, 'transformers', transformers.__version__, 'accelerate', accelerate.__version__,
      'hub', huggingface_hub.__version__, 'peft', peft.__version__, 'xformers', xformers.__version__)
import torch.nn.functional as F
x = torch.randn(4, 77, 2048, device='cuda', dtype=torch.bfloat16)
w = torch.randn(1280, 2048, device='cuda', dtype=torch.bfloat16)
with torch.no_grad():
    print('bf16 linear without bias ok', float(F.linear(x, w).float().abs().mean()))
q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
F.scaled_dot_product_attention(q, q, q).sum().backward()
torch.cuda.synchronize()
print('sdpa backward ok', float(q.grad.float().abs().mean()))
f = torch.compile(lambda a: torch.nn.functional.gelu(a @ a).sum())
print('compile ok', float(f(torch.randn(1024, 1024, device='cuda'))))
q = torch.randn(2, 4096, 10, 64, device='cuda', dtype=torch.bfloat16)
print('xformers ok', xformers.ops.memory_efficient_attention(q, q, q).float().abs().mean().item())
PY
$PIP freeze > $A/env_freeze.txt
echo ENV_DONE
