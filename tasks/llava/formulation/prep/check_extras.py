"""Verify the optional performance libraries on one GPU (run under flock on the GPU lock)."""
import torch, triton
print('torch', torch.__version__, 'triton', triton.__version__)
import apex, amp_C, fused_layer_norm_cuda
from apex.optimizers import FusedAdam
from apex.normalization import FusedRMSNorm
x = torch.randn(4, 4096, device='cuda', dtype=torch.bfloat16, requires_grad=True)
n = FusedRMSNorm(4096).cuda().to(torch.bfloat16)
n(x).sum().backward()
print('apex FusedRMSNorm ok', float(x.grad.float().abs().mean()))
p = torch.nn.Parameter(torch.randn(1000, device='cuda'))
p.grad = torch.randn_like(p)
FusedAdam([p], lr=1e-3, adam_w_mode=True).step()
print('apex FusedAdam ok')
try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
    w = torch.ones(4096, device='cuda', dtype=torch.bfloat16)
    y = LigerRMSNormFunction.apply(x, w, 1e-6)
    y.sum().backward()
    a = torch.randn(8, 11008, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    LigerSiLUMulFunction.apply(a, a).sum().backward()
    logits = torch.randn(16, 32000, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    t = torch.randint(0, 32000, (16,), device='cuda')
    out = LigerCrossEntropyFunction.apply(logits, t, -100)
    (out[0] if isinstance(out, tuple) else out).backward()
    print('liger rms_norm / swiglu / cross_entropy ok')
except Exception as e:
    print('liger FAILED', type(e).__name__, str(e)[:300])
try:
    from deepspeed.ops.adam import FusedAdam as DSFusedAdam
    q = torch.nn.Parameter(torch.randn(1000, device='cuda'))
    q.grad = torch.randn_like(q)
    DSFusedAdam([q], lr=1e-3).step()
    print('deepspeed FusedAdam (JIT) ok')
except Exception as e:
    print('deepspeed FusedAdam FAILED', type(e).__name__, str(e)[:200])
