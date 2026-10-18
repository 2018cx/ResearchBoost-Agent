---
name: train-opt-action-ladder
description: The optimization action ladder, L0 switches → L1 libraries → L2 configuration search → L3 graph/Python rewrites → L4 custom kernels, plus a symptom → first-suspect table. Read when deciding what to do this round.
---

# The action ladder

Use when: choosing a hotspot and changing code each round, to decide what to try next. Reuse existing optimizations before writing new code; every rung lists its typical numerics tier (defined in `train-opt-numerics-validation`).

## Check the existing fast paths first

**The fast-path inventory is a task asset**: built when the task is formulated, extended by the optimizing agent in the WORK_LOG every round. Each fast path records:

- its entry point and its kernel name in the trace;
- its enabling conditions: dtype, stride or contiguity, alignment, dimension limits, whether it requires compile or a graph;
- how it falls back when a condition is not met: silently, with a log line, or by disabling itself for the whole process after an error;
- its numerics contract: bitwise, ≤1 ULP, selection-equivalent, or approximate;
- its tests and microbenchmarks;
- negative results already tried.

"Reuse before writing new code" depends on this inventory.

**Symptom → first suspect** (fix enabling conditions before writing new code):

| Symptom | First suspect |
|---|---|
| A fused kernel is missing from the trace but its small component ops are present | a guard or enabling condition is not met |
| Projections that should be merged still run as several GEMMs | the packed path is not active, or the weight layout does not match |
| Many launch gaps at fixed shapes | an existing graph/compile path is not used |
| Communication dominates | existing compute/communication overlap is not enabled |
| Attention dominates | backend selection and shape guards |
| A fast path appears only some of the time | it disabled itself for the whole process after one runtime exception |
| A compiled function shows no compiler-generated kernels in the trace, only small eager ops | it was called once under some dispatch mode (see below) |

**Compile falls back to eager permanently after touching a dispatch mode.** Selective activation recomputation (SAC, `create_selective_checkpoint_contexts`) and dtype tracers are all TorchDispatchModes underneath. Measured on torch 2.7: once a compiled function has been called a single time under such a mode, **every** later call runs eager, including calls outside the mode, with no error and no log line. In training with SAC, compile inside the recomputed regions never takes effect, and both timing and output look exactly like "compile does nothing".

- Detect: look in the trace for compiler-generated kernels (Inductor's `triton_`-prefixed kernels), whether they appear and whether their count equals the call count; or add counters to the compile wrapper that count "compiled" and "eager" calls separately. If, for the same kind of compiled function, the copy called only on no_grad paths shows up in the trace but the copy that gets recomputed does not, this is the problem.
- Avoid: do not use compile on paths that are recomputed; use a hand-written Triton kernel instead (unaffected by dispatch modes); or make the compile wrapper check for an active dispatch mode first (for example `torch.utils._python_dispatch._get_current_dispatch_mode()` is not None) and, if there is one, run the eager formula without calling the compiled function, so it is never "poisoned".
- Tier: a hand-written kernel written to reproduce the library implementation bit for bit (see `train-opt-fusion-patterns`, "Reproducing library reductions bit for bit") can stay at N0.

## Classify first, then work in batches

After profiling, decide which class of bottleneck a hotspot belongs to; solve one class per round, doing the actions of that class together, each behind its own switch, screen out the useless ones with quick timings, then combine the rest into one formal evaluation.

| Bottleneck class | How it shows in a profile | Actions taken together in one round |
|---|---|---|
| Memory (elementwise, layout copies, norms, activations) | many short kernels; high bandwidth, low compute utilization; a large share of "copy / elementwise / layout" | whole-model or regional `torch.compile`; fusion libraries (fused norms, fused activations); hand-written fusion between GEMMs; removing redundant contiguous, transpose and copy calls. Common fusion paths and examples are in `train-opt-fusion-patterns` |
| Compute (GEMM, convolution, attention) | GEMM or convolution kernels dominate; check whether tensor cores are used (SIMT `ffma`/`sgemm` means they are not) | precision modes within the contract; a different library or backend (cuBLASLt, cuDNN, the SDPA backends); dimension alignment; batching small GEMMs |
| Launch and host | high GPU idle share and large gaps between kernels; a saturated CPU thread; syncs inside the step | CUDA graphs; compile with reduce-overhead; removing `.item()` and syncs inside the step; foreach/fused optimizers and EMA |
| Communication | NCCL exposed outside compute; stragglers between ranks | bucket size, `gradient_as_bucket_view`; overlapping compute and communication; regional compile to restore overlap |
| Data | the main process waits on `next(loader)`; H2D does not overlap with compute | workers, pinning, prefetch; moving decoding into worker processes or onto the GPU; precomputing deterministic parts offline |

Standard substitutions (SDPA backends, fused optimizers, compile modes, library options) are swept once, in the round for their class: time each option once, and if none gives a clear speedup, close the whole direction, log one "no gain" entry, and do not give it a round of its own.

## L0 switches and configuration

| Action | Tier | Prior |
|---|---|---|
| DataLoader `workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`; `non_blocking` H2D | N0 | do these first when the data stage is large; transfer uint8 to the GPU and convert there |
| Remove `.item()` and `print(tensor)` inside the step; accumulate metrics on the GPU and read them every N steps | N0 | sync points stop the CPU from queuing work ahead |
| `zero_grad(set_to_none=True)`, asynchronous checkpointing | N0 | asynchronous saves must finish their background write before exit |
| fused / foreach optimizers, foreach EMA | N1 | rounding differs from the for-loop version; never bitwise identical |
| `torch.compile` (no precision-related flags) | N1 | count cold compilation for short jobs; see the compile notes below |
| DDP bucket size, `gradient_as_bucket_view` | N1 | first confirm communication is actually exposed |
| TF32, bf16 autocast, wider autocast coverage | **N2** | first check whether upstream uses tensor cores at all; fp32 on SIMT can be 2× slower or more. But this is a precision change and needs sign-off |
| Turn off switches upstream enabled to save memory (activation recomputation, optimizer offload) | N0/N1 | with enough memory they are pure overhead; confirm peak-memory headroom first |

Compile notes:

- In some versions, compile combined with DDP's graph splitting silently drops gradients. After enabling it, check per parameter that gradients exist and match eager.
- Compiled graphs are sensitive to switching training phases within one process: a change in how backward is run (for example starting to use `retain_graph`) may raise an error or trigger recompilation. Phase switches need their own test case.
- `max-autotune` replaces cuBLAS with Triton GEMMs, with different numerics (Triton's TF32 truncates operands by default; cuBLAS rounds to nearest), and often brings no gain. Validate it as N1/N2; it is not a free switch.
- **Recompile limit**: with regional compilation, the code object of one forward is shared by dozens of layer instances, different grad modes (training versus EMA, no_grad) and different shapes; beyond `recompile_limit` (default 8) recompilations it silently falls back to eager, or fails outright with the error switch on. First use `TORCH_LOGS=recompiles` to see why each recompilation happens, then address the cause, for example compiling only the training path, fixing shapes, or marking dynamic dimensions; raising the limit comes last. Self-tests must use the full model and the full training flow (all layers, EMA, optimizer); a single-layer fixture does not reproduce this.
- **Proving compile and graphs are effective**:
  - recompilations and graph signature misses inside the timing window must both be 0, proven by counters;
  - warmup must cover every shape that appears later: the last incomplete batch, every training phase, eval;
  - every switch must be set before capture/compile. A switch turned on after capture is silently bypassed at replay; a graph that misses can be slower than eager.

## L1 a different library or backend

- Attention: availability and speed of the SDPA backends (cuDNN, flash, mem-efficient, math) depend on dtype. In fp32/TF32 there is often no fast path, and switching to SDPA can even be slower; measure per dtype before choosing. Tier N1.
- Use ready-made fused kernels from libraries (fused norms, fused optimizers and so on). Library activation functions may be approximations (for example tanh-approximated GELU), which changes the model function: N3.

## L2 parameter search

- Sweep only 2–6 candidates for compile modes, regional compilation, bucket size, worker counts and similar.
- Regional compilation may restore compute/communication overlap, or may be slower; measure it.

## L3 graph or Python rewrites

- Remove repeated backward passes: for example calling `autograd.grad` on the whole graph several times to get the gradient norm of one weight. Running backward once and combining by the chain rule is mathematically equivalent (N1).
  - **But check DDP's hidden state**: when `autograd.grad` replaces `.backward()`, DDP's reducer can be left waiting for gradients, so the next backward on that module triggers an all-reduce and changes how its gradients are synchronized. Confirm with cross-rank checksums.
- Cache the outputs of frozen modules: only deterministic views can be cached, random augmentation invalidates a cache, and the contract must explicitly allow caching, otherwise it is N5.
- Reuse intermediates across steps: if a weight update happens in between (for example the discriminator step re-forwards with the new generator weights after a generator update), reusing old results changes the training semantics: N4, never without sign-off.
- Remove host syncs inside the step and batch small kernels.

## L4 custom kernels

Hand over to a single-op kernel workflow: contract → draft → plan → small verified steps → evidence → promotion, with a read-only reviewer. The verifier must be strict; a lenient verifier will be gamed. Kernels need both forward and backward and are tested with the shapes, dtypes and strides of the real call sites.

Engineering rules:

- Register as a custom op with a fake/meta implementation; training kernels also register autograd so they do not break the graph under compile.
- Record the reason when a guard falls back, and count it.
- Coupled sites are switched on and off together.
- A path declared bitwise identical runs one live equality check against the reference computation on its first call and disables itself if they differ.
- Do not autotune on every call for paths called by every block.
- Add the new path to the fast-path inventory.

Priors:

- **Fuse between GEMMs; do not rush to fuse into them.** In fp32/TF32 training, fusing norm, activation and residual into a GEMM's prologue/epilogue often loses to "cuBLAS + a separate fused kernel" when the compiler has no warp specialization on the target architecture and the epilogue has to run after the main loop. Fuse the memory-bound work between GEMMs first: bias + residual + norm in one pass forward; norm gradient, residual gradient, γ/β gradients and the neighboring GEMM's bias gradient in one pass backward.
- After compilation, the remaining non-GEMM cost is often layout traffic: copies in backward from the qkv split, transposes, RoPE, and the split of gated activations. Fusing their forward and backward is a cost-effective target.
- Triton's TF32 `dot` truncates operands by default, with about 3× the error of round-to-nearest and a bias. To match cuBLAS, round to nearest-even explicitly.
- fp32/TF32 attention with few heads and small tiles can be occupancy-bound; its backward commonly suffers from register pressure and spills.
- The power cap eats part of a local gain: a kernel twice as fast may make the whole step only a few percent faster. End-to-end is what counts.

## Remedies by stage

Within each stage, try in the order "switch → move → swap library → rewrite", and re-measure the stage budget table after each accepted change:

- Data: workers/pinning → move GIL-holding decoding into worker processes or background prefetch → GPU decoding → precompute deterministic parts offline.
- Frozen modules: offline caching (if the contract allows) → lower precision or inference mode (N2) → compilation.
- Compute: lossless actions first (SDPA/fusion libraries → compilation → dimension alignment → layout and fused kernels, N0/N1); send precision changes (N2) for sign-off early, but do not build later gains on an N2 change that is not yet signed off. Put each lossless action behind its own switch and measure it at both precisions so attribution stays clean.
- Optimizer and EMA: fused/foreach, with counters kept on the host.
- Host syncs: merge `.item()` calls, move logging to a background thread, collect garbage manually.
- Eval and checkpointing: run eval in a separate process and save asynchronously. Reducing eval frequency is N5: it changes what is measured.
