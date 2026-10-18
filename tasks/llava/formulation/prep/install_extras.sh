#!/bin/bash
# Optional performance libraries for the candidate (the reference recipe uses none of them):
#   nsight systems (nsys CLI), NVIDIA apex 24.04.01 (CUDA/C++ extensions: fused Adam/LAMB, fused layer/RMS norm,
#   fused dense, ...), liger-kernel (Triton kernels; checked against the env's triton 2.1.0 below).
# DeepSpeed's own fused ops (FusedAdam etc.) are JIT-built on first use into ~/.cache/torch_extensions (nvcc + ninja).
set -euo pipefail
#   ASSETS    the task's assets directory
#   APEX_SRC  a checkout of NVIDIA/apex at tag 24.04.01 (copied to $ASSETS/build/apex)
A=${ASSETS:?set ASSETS}; E=$A/env; P=$E/bin/python; PIP="$E/bin/pip"
export PATH=$E/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
echo "== nsight systems"
if ! ls $A/nsight/opt/nvidia/nsight-systems/*/bin/nsys >/dev/null 2>&1; then
  DEB=nsight-systems-2025.6.3_2025.6.3.541-1_amd64.deb
  python3 "$(cd "$(dirname "$0")" && pwd)/pdl.py" https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/$DEB $A/$DEB --parts 8
  mkdir -p $A/nsight && dpkg -x $A/$DEB $A/nsight && rm -f $A/$DEB $A/$DEB.done
fi
ls $A/nsight/opt/nvidia/nsight-systems/*/bin/nsys
echo "== liger-kernel"
$P -c "import liger_kernel" 2>/dev/null || $PIP install --no-deps "liger-kernel==0.4.2"
echo "== apex"
if ! $P -c "import apex, fused_adam_cuda, fused_layer_norm_cuda" 2>/dev/null; then
  B=$A/build; mkdir -p $B; cd $B
  [ -d apex ] || cp -a "${APEX_SRC:?set APEX_SRC to an NVIDIA/apex checkout at tag 24.04.01}" apex
  cd apex
  # apex refuses any CUDA minor-version mismatch; the system nvcc is 12.6, torch is cu121 (same major, compatible)
  sed -i 's/if (bare_metal_version != torch_binary_version):/if (bare_metal_version.major != torch_binary_version.major):/' setup.py
  MAX_JOBS=32 TORCH_CUDA_ARCH_LIST=9.0 NVCC_APPEND_FLAGS="--threads 4" $PIP install -v --no-build-isolation --no-cache-dir \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" . > $B/apex_build.log 2>&1 \
    || { tail -n 40 $B/apex_build.log; exit 1; }
fi
echo "== check"
$P - <<'PY'
import torch, triton, importlib
print('torch', torch.__version__, 'triton', triton.__version__)
import apex, fused_adam_cuda, fused_layer_norm_cuda
from apex.optimizers import FusedAdam
from apex.normalization import FusedRMSNorm
x = torch.randn(4, 4096, device='cuda', dtype=torch.bfloat16, requires_grad=True)
n = FusedRMSNorm(4096).cuda().to(torch.bfloat16)
n(x).sum().backward(); print('apex FusedRMSNorm ok')
p = torch.nn.Parameter(torch.randn(1000, device='cuda')); p.grad = torch.randn_like(p)
FusedAdam([p], lr=1e-3, adam_w_mode=True).step(); print('apex FusedAdam ok')
try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
    w = torch.ones(4096, device='cuda', dtype=torch.bfloat16)
    y = LigerRMSNormFunction.apply(x, w, 1e-6); y.sum().backward()
    a = torch.randn(8, 11008, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    LigerSiLUMulFunction.apply(a, a).sum().backward()
    logits = torch.randn(16, 32000, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    t = torch.randint(0, 32000, (16,), device='cuda')
    LigerCrossEntropyFunction.apply(logits, t, -100)[0].backward()
    print('liger rms_norm / swiglu / cross_entropy ok with triton', triton.__version__)
except Exception as e:
    print('liger FAILED with triton', triton.__version__, type(e).__name__, str(e)[:300])
PY
echo EXTRAS_DONE
