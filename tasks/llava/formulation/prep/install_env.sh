#!/bin/bash
# LLaVA-1.5 training env under $ASSETS/env: Python 3.10 + upstream pyproject pins (torch 2.1.2 cu121,
# transformers 4.37.2, deepspeed 0.12.6, flash-attn 2.5.8 built from source for sm_90). Re-runnable.
#   ASSETS     the task's assets directory
#   MINICONDA  path to a Miniconda installer for Python 3.10 (from repo.anaconda.com/miniconda)
#   pip honours PIP_INDEX_URL if you use a mirror.
set -euo pipefail
A=${ASSETS:?set ASSETS}; E=$A/env; P=$E/bin/python; PIP="$E/bin/pip"
mkdir -p $A
echo "== miniconda"
[ -x $P ] || bash "${MINICONDA:?set MINICONDA to a Miniconda installer for Python 3.10}" -b -p $E
$P --version
echo "== torch"
$PIP install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
$PIP install nvidia-nvjitlink-cu12==12.1.105
echo "== upstream deps"
$PIP install numpy==1.26.4 scipy==1.13.1 transformers==4.37.2 tokenizers==0.15.1 huggingface-hub==0.20.3 \
  safetensors==0.4.2 sentencepiece==0.1.99 shortuuid accelerate==0.21.0 peft==0.10.0 pydantic markdown2 \
  scikit-learn==1.2.2 einops==0.6.1 einops-exts==0.0.4 timm==0.6.13 protobuf==4.25.3 ninja psutil hjson py-cpuinfo \
  pynvml liger-kernel==0.4.2
# torch 2.1.2's cpp_extension imports pkg_resources, which new setuptools no longer ship
$PIP install "setuptools==69.5.1"
$PIP install --no-build-isolation deepspeed==0.12.6
echo "== flash-attn"
if ! $P -c "import flash_attn" 2>/dev/null; then
  export PATH=$E/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
  export FLASH_ATTN_CUDA_ARCHS=90 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=${MAX_JOBS:-48} FLASH_ATTENTION_FORCE_BUILD=TRUE
  $PIP install --no-build-isolation flash-attn==2.5.8
fi
echo "== check"
$P - <<'PY'
import torch, torchvision, transformers, deepspeed, flash_attn, triton, accelerate, numpy
print(torch.__version__, torch.version.cuda, torchvision.__version__, transformers.__version__, deepspeed.__version__,
      flash_attn.__version__, triton.__version__, accelerate.__version__, numpy.__version__, torch.backends.cudnn.version())
print(torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
from flash_attn import flash_attn_func
q = torch.randn(2, 512, 8, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
o = flash_attn_func(q, q, q, causal=True, deterministic=True); o.sum().backward()
print('flash-attn fwd/bwd ok', float(o.float().abs().mean()), float(q.grad.float().abs().mean()))
PY
echo ENV_DONE
