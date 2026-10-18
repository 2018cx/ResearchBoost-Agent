---
name: train-opt-fusion-patterns
description: Common fusion patterns in Transformer/DiT training and when they pay off (norm, RoPE, modulation, residual, GEMM pre/post-processing, where compile helps, reproducing library reductions bit for bit). Read before writing a fusion or compile plan.
---

# Common fusion paths and examples

This card is for the round that targets a memory-bound bottleneck (`train-opt-action-ladder`, "Classify first, then work in batches"). When a profile shows a run of elementwise, reduction or layout kernels, look up here which class it belongs to and whether an existing path can be used, then decide the fusion plan. Some entries follow the fast-path lists of open-source inference frameworks, with the framework-specific parts removed and the backward-pass issues of training added.

## Read these first

1. **Reuse before writing by hand.** In this order:
   1. find out why an existing fast path was missed, for example dtype, alignment, contiguity, a guard or the compile context (see `train-opt-tiered-profiling`, "Three kinds of hotspot");
   2. use a ready-made fused op from a library;
   3. let `torch.compile` fuse the elementwise chain;
   4. only if none of these works, write a kernel by hand.
2. **Fuse between GEMMs; do not rush to fuse into them.** Putting norm, activation and residual into a GEMM's prologue or epilogue often loses to "library GEMM + one separate fused kernel", for two reasons: the compiler may lack warp specialization on the target architecture, and the library epilogue's function may not match the model's, for example a tanh-approximated GELU.
3. **Estimate the ceiling first.** When every kernel of a memory-bound chain is already near the DRAM roofline, the most a fusion can gain is the intermediate bytes saved ÷ total bytes. If that is below twice the noise, do not do it. Record negative results, for example "two kernels already at the roofline fused into one, and it got slower".
4. **In training, fuse the backward too.** Memory traffic saved in the forward is easily given back by splits, transposes and copies in the backward. Every fusion must answer two questions: how is the backward computed, and which intermediates are saved for it versus recomputed in it.
5. **State the numerics contract**, one of:
   - bitwise (the same reduction and rounding order as the reference)
   - ≤1 ULP
   - selection-equivalent (for example the same top-k set, in no guaranteed order)
   - approximate

   A path declared bitwise runs one live equality check against the reference computation on its first call and disables itself if they differ. How fusions with reductions can be made bitwise is covered in "Reproducing library reductions bit for bit" below.
6. **Guards and fallbacks must be visible.** When a condition is not met, fall back to the original formula, and record and count the reason. A path that disables itself for the whole process after an error must leave a log line, and that log line is also an invalid-run signature (`train-opt-measurement`).
7. **Engineering rules**:
   - Register as a custom op with a fake/meta implementation; training kernels also register autograd so they do not break the graph under compile.
   - Do not autotune on every call: for paths called by every block, use a fixed tile or choose once during warmup.
   - Prefer stride-aware kernels to forcing `.contiguous()` first; that copy is itself the overhead to remove.

## 1. Residual add + norm

**In the trace**: an `add` followed by the kernels of LayerNorm or RMSNorm (mean or sum of squares, normalize, multiply by γ and add β). Twice per block, after attention and after the MLP.

```python
# reference
h = x + y                       # write h once, read h once
out = F.layer_norm(h, (C,), w, b)
# fused: read x and y once, write h (still needed by the residual stream) and out
h, out = fused_add_layernorm(x, y, w, b)
```

- **Enabling conditions**: last dimension contiguous; hidden size within what one block can hold, commonly a row smaller than 64 KB; dtype in the supported list.
- **Numerics contract**: bitwise when the reduction order matches the library implementation, otherwise N1.
- **Training notes**: in backward, the norm's input gradient, the residual gradient and the γ/β gradients can be computed in one pass; the cross-row reduction for the γ/β gradients is accumulated block by block. Save mean and rstd for the backward instead of recomputing the statistics there.

## 2. Norm + scale and shift (+ gate): adaLN in DiT

**In the trace**: a norm followed by `mul` and `add` (broadcasting `1 + scale`), and at the residual `residual + gate * update`. Several such chains per block.

```python
y = norm(x) * (1 + scale[:, None]) + shift[:, None]
x = residual + gate[:, None] * update          # gated residual add
y = norm(residual + gate * x) * (1 + scale) + shift   # both chained together
```

- **Enabling conditions**: the broadcast shapes of scale, shift and gate (`[B, C]`, `[B, 1, C]`, `[B, F, 1, C]` and so on) must be supported; variants indexed by token or frame need separate handling. Mixed layouts (for example a transposed residual and a contiguous update) should be handled inside the kernel instead of being copied to contiguous first.
- **Numerics contract**: bitwise as long as the original formula's rounding boundaries are reproduced (where the precision conversions happen in bf16); rewriting `x * (1 + s)` as `x + x * s`, or folding γ into scale (affine folding), changes rounding and is at least N1.
- **Training notes**: the gradients of scale, shift and gate are reductions over the token dimension and can be computed in the same kernel as the input gradient.

## 3. Bias + activation (+ gated activation)

**In the trace**: a GEMM followed by `add` (bias) and GELU or SiLU; gated activations (SwiGLU, GeGLU) add `chunk`/`split` and `mul`, and `cat` or copies show up in backward.

```python
# reference
h = F.linear(x, w1, b1)
a, g = h.chunk(2, dim=-1)        # views in forward, concatenated back in backward, causing a copy
out = F.silu(a) * g
# fused: one kernel reads h and writes out; in backward one kernel writes da and dg directly into one contiguous buffer, no separate cat
out = fused_silu_mul(h)
```

- **Numerics contract**: GELU must match the model's definition: erf for the exact form, tanh for the approximate one. Fused epilogues from libraries are often the tanh approximation; using one changes the model function and is N3.
- **Training notes**: decide whether to save the pre-activation values or recompute them in backward (recomputation saves memory at the cost of one more read and write). The bias gradient is a reduction over tokens and can be combined with the activation's backward.

## 4. RoPE (and QK norm + RoPE)

**In the trace**: for Q and K each, a chain of `unflatten`, `chunk`, `mul`, `add`, `stack` or `cat`; with QK norm, a norm chain in front.

```python
# reference: Q and K one at a time
q = q * cos + rotate_half(q) * sin
k = k * cos + rotate_half(k) * sin
# fused: Q and K together, in place, one launch; with QK norm, normalize then rotate
fused_qk_norm_rope_(q, k, cos, sin, q_w, k_w)
```

- **Enabling conditions**: head_dim even or in the supported list; distinguish GPT-J style (interleaved pairs) from NeoX style (two halves); matching shape and dtype of the cos and sin tables; Q and K in the same layout, pointers aligned.
- **Numerics contract**: bitwise with explicit round-to-nearest in fp32 before converting back to the activation dtype.
- **Training notes**: RoPE is linear, so its backward is the inverse rotation and needs no saved intermediates; QK norm saves rstd. In-place rewriting destroys inputs that autograd needs to save; in training use the out-of-place version, or think carefully about what is saved.

## 5. QKV packing and merged projections

**In the trace**: three GEMMs of the same shape in a row on the same input (one each for Q, K and V).

```python
# reference: 3 GEMMs, x read 3 times
q, k, v = x @ wq.T, x @ wk.T, x @ wv.T
# merged: one GEMM with a concatenated weight view [3C, C]; the result is split by views, no copy
qkv = x @ torch.cat([wq, wk, wv]).T    # in practice a pre-concatenated parameter or a view over shared storage
q, k, v = qkv.view(B, S, 3, H, D).unbind(2)
```

- **Numerics contract**: bitwise with the same GEMM algorithm; but the merged matrix has a different shape, the library may choose a different GEMM algorithm, and then it is N1 and must be declared.
- **Training notes**:
  - the parameter layout changes, so state_dict keys and shapes change, which needs a canonical mapping in the judge's adapter (`numerics/adapter.py`) and a checkpoint conversion;
  - optimizer state follows the parameters;
  - verify EMA and deepcopy again.

  Split the heads afterwards with views; `chunk` followed by `contiguous` adds copies in backward.

## 6. Removing layout copies

**In the trace**: large amounts of `copy_`, `contiguous`, `permute`, `transpose`, plus `repeat_interleave`, `cat` and `pad`. They often sit around attention (splitting and merging heads), at NCHW/NHWC conversions in convolutional networks, and where causal caches are concatenated.

Approach:
- Have the previous kernel write its result directly in the layout the next op needs, instead of writing once and transposing once.
- Read transposed inputs directly with stride-aware kernels instead of calling `.contiguous()` first.
- Use channels_last end to end in convolutional networks. Converting back to NCHW midway costs two full copies and can change the algorithm cuDNN selects.
- `cat` plus `pad` before a convolution: write both the input and the next step's cache in one kernel.
- `repeat_interleave` immediately followed by an add with another tensor: fuse into "indexed read, then add", without materializing the intermediate.

Numerics contract: pure data movement and additions in the same order are bitwise. But a layout change can change the reduction order of a following op, for example a norm or a convolution, which then stops being bitwise; measure it.

## 7. GroupNorm + SiLU (convolutional networks, VAEs)

```python
y = F.silu(F.group_norm(x, G, w, b))   # two read/write passes
y = fused_group_norm_silu(x, G, w, b)   # one pass
```

- **Enabling conditions**: affine GroupNorm; not the in-place activation; the split of groups and channels must be supported. Inference-only implementations usually require `requires_grad=False`; training needs its own backward.
- **Training notes**: backward needs mean, rstd and the pre-SiLU values (or recomputes them). A two-pass algorithm works: statistics in the first pass, normalization plus activation in the second.

## 8. Small selections and reductions (routing, top-k, masks)

**In the trace**: several `topk`, `scatter`, mask construction and another `topk` per token; many kernels, each very small.

Approach: one program per token computes the grouped top-k, the mask and the final top-k in one go. The numerics contract is usually selection-equivalent: the same selected set, in no guaranteed order (the reference itself uses `sorted=False`).

## 9. Optimizers, EMA and other once-per-parameter loops

**In the trace**: a kernel count of roughly "number of parameter tensors × a constant", each kernel very short, a small share of GPU time, but a long host-side launch time.

Approach: use foreach or fused versions (multi-tensor apply); for EMA, `torch._foreach_mul_` plus `_foreach_add_`, or `_foreach_lerp_`. Numerics contract: when foreach keeps the same elementwise formula, results are bitwise identical to the for-loop version; lerp rounds differently from multiply-add and is N1.

## 10. Attention

Attention is not fused by hand; switch backends first: the SDPA backends, or the flash family of libraries. Availability depends on dtype: the flash and cuDNN backends usually accept only fp16/bf16, fp32/TF32 inputs can only use math or mem-efficient, and backward may be slower. Following "sweep the library for standard substitutions first", time every usable backend once in this round, and if none gives a clear gain, drop the whole direction. The preparation in front of attention (QKV projection, QK norm, RoPE, head split) can be fused as in classes 4, 5 and 6.

## Reproducing library reductions bit for bit: keeping fusions with reductions at N0

**Why it is worth it**: the bitwise gate needs one short recording of a few minutes; the statistical gate needs multi-seed training trajectories, in GPU hours. A fusion that reproduces the library implementation bit for bit ends at the bitwise gate, an order of magnitude cheaper to validate, and N0 changes can be stacked freely without accumulating error round by round. Elementwise chains (scale and shift, gating, activations) are bitwise as long as the rounding points are copied; the hard part is anything with a reduction: norm statistics, and the sums over tokens in the gradients of broadcast parameters.

**How**:

1. **Read the library kernel's source to determine the reduction structure; do not guess.** PyTorch's generic reduction (`reduce_kernel`), for example, first accumulates within each thread in groups of the vectorization width, then merges across the warp with a shuffle tree, then across the block through shared memory in a tree, and across blocks if needed. The group width, the number of threads per output and the path taken all depend on input size, stride and alignment. The fused kernel must use the same partitioning and the same merge order.
2. **For LayerNorm-like ops, copy Welford.** The library computes mean and variance online with Welford: per-element updates within a thread, then merges between threads in a fixed tree order. In the merge formula, the order of the delta and count computations and the placement of FMAs all affect the last bit. Whether rstd uses `rsqrt` or `1/sqrt`, where eps is added and where the result is converted back to low precision must all match the library.
3. **Match floating-point contraction.** Compilers contract `a * b + c` into an FMA. Match, site by site, where the library source uses FMAs and where it does not contract; in Triton, write FMAs explicitly where needed, or split expressions to prevent contraction.
4. **Copy the rounding points.** Where values go from fp32 back to bf16, and which dtype each intermediate is stored in, must match the reference formula exactly.

**Bitwise probe sweep before delivery**:

- Dozens of shapes: row lengths covering every remainder of the vectorization width, non-powers of two, production shapes, very few and very many rows.
- Several input distributions, including large-mean, small-variance inputs that are sensitive to the Welford merge order.
- Every probe must produce bitwise-identical outputs, and the statistics saved for backward (mean, rstd) must be bitwise identical too. Claim bitwise only when all of them pass; if any fails, downgrade to N1 or keep fixing.
- The probe set is checked in with the kernel as a regression test. Run it on the target GPU: the library's partitioning may depend on device properties, so rerun the sweep on a different GPU.
- If the backward keeps the library implementation, bitwise-identical forward outputs and saved statistics are enough.

## What every fusion ships with

- A comparison of the reference formula and the fused implementation at the real call sites: forward outputs, all input gradients and parameter gradients, measured by relative L2; bitwise paths must be exactly equal.
- The list of guard conditions, a fallback counter and a fast-path hit counter.
- A microbenchmark: single-op time and bytes read/written before and after fusion, at production shapes, dtypes and strides.
- End-to-end results come from the bench. A kernel twice as fast may give only a few percent end to end under the power cap.
- An entry in the fast-path inventory: entry point, enabling conditions, fallback behavior, numerics contract, tests and negative results.
