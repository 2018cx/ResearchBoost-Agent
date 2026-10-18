---
name: train-opt-fsdp-multi-forward
description: Memory and communication patterns when an FSDP model runs several forwards per iteration (rollouts, scoring with several models): gather once per iteration, cast frozen models before wrapping, keep frozen models resident, caching, FSDP root copying dict/list kwargs, and re-planning recomputation with the memory freed. Read before choosing hotspots in multi-model or multi-forward FSDP training.
---

# FSDP: several forwards in one iteration

Use when: choosing hotspots and changing code for an FSDP-wrapped model that is called many times in one training iteration. Typical cases:

- autoregressive rollout: the generator produces block by block with a KV cache, and every block and every denoising step calls the same FSDP module;
- the teacher's cond and uncond forwards (CFG);
- a frozen text encoder running one forward every step;
- several models (generator, critic, teacher) alternating on the same GPUs.

**In the trace**:
- the number of all-gathers is roughly "forwards per iteration × FSDP units", with a visible NCCL share;
- a run of fp32 → bf16 cast kernels before every forward of a frozen model;
- a high GPU idle share in the rollout segment, limited by CPU launches.

**Estimate the ceiling first**: most FSDP all-gathers already overlap with compute. Take the NCCL time exposed outside compute in the profile, plus the share of casts and idle time, as the ceiling for the items below.

Every item below changes only communication, storage and scheduling, never computation, so each one should reach **N0**: bitwise identical to the reference in deterministic mode, plus counters proving it is actually taken. If an item fails the bitwise check, it changed the computation; find out how, rather than reclassifying it as N1.

## 1. All-gather once per iteration

- **How**: unshard the module's parameters at the start of an iteration (or a rollout), reuse them for every forward in the iteration, and reshard at the end. FSDP2 has `unshard()` / `reshard()`; with FSDP1, skip the post-forward reshard while holding.
- **Conditions**:
  - memory can hold every unsharded unit for the whole holding period;
  - no optimizer step on these parameters while they are held;
  - for forwards with gradients, confirm the backward uses the same parameters and that gradients are still reduce-scattered back into the shards as before.
- **Proof**: counters for unshards and all-gathers per iteration, which should equal the number of units; fewer all-gathers in the profile.

## 2. Cast frozen models to the compute dtype before wrapping in FSDP

- **How**: cast frozen models such as the teacher and the text encoder to the mixed-precision compute dtype before wrapping. FSDP then no longer casts fp32 shards to low precision on every forward, and no longer stores fp32 shards.
- **Conditions**:
  - the reference already computes these models in that dtype;
  - the models are frozen and need no fp32 master weights;
  - floating-point buffers (for example a positional-encoding frequency table) keep their original dtype: a plain `.to(dtype)` would convert them too, so exclude them.
- **Proof**:
  - the cast kernels before frozen-model forwards disappear from the trace;
  - shard memory halves;
  - outputs are bitwise identical. Casting once ahead of time rounds exactly like casting on every forward, so the results should match exactly.

## 3. Keep the frozen encoder's parameters resident

- **How**: after the frozen text encoder is unsharded once, never reshard it; keep the full parameters resident and save its all-gather every step.
- **Conditions**:
  - enough memory headroom: with the full parameters resident, the peak phase must still leave a margin;
  - the gain is only the part of its communication not hidden by compute; estimate it from the profile first.
  - A large teacher usually does not fit resident; holding it between cond and uncond (item 1) is enough.
- **Proof**: the resident unit is gathered only on the first forward and never again; count both events.

## 4. Cross-attention K/V cache

- **How**: when the text condition does not change within a rollout, compute the cross-attention K/V projections once and reuse them for the later steps.
- **Conditions**:
  - the condition tensor and the projection weights do not change while cached: no optimizer step within the iteration.
  - Reuse inside no_grad rollout steps is the safest.
  - If a path with gradients reuses them, gradients accumulate on the same tensor; confirm the tier with the bitwise gate.
- **Proof**: a counter of K/V projection calls, which should drop from "steps × layers" to "layers".

## 5. Remove host syncs

- **How**: keep indices such as the KV-cache write position and current length as host-side Python integers instead of reading them from GPU tensors with `.item()`; base conditional branches only on host-side state.
- **Pitfall: the FSDP root copies dict and list kwargs.**
  - Cause: `torch.distributed.utils._apply_to_tensors` rebuilds these containers while traversing the arguments, so the module's changes to them are invisible to the caller.
  - Effect: state is "updated", yet the next call sees the old value.
  - Fix: keep Python state that must survive across calls in an instance of a plain custom class, which is passed through by reference. dicts, lists and tuples do not work, and dataclasses may be traversed too.
- **Proof**: `torch.cuda.set_sync_debug_mode("warn")` confirms there are no implicit syncs in the rollout segment; the GPU idle time of the rollout segment shrinks in the profile.

## 6. Spend the freed memory on speed

- **How**: the items above free memory; then re-evaluate which extra activations selective activation recomputation (SAC) should save.
  - Switch per model (the bidirectional critic / teacher, or the causal generator) and per role (attention output projection, second FFN layer, cross-attention output and so on);
  - measure peak memory and per-phase time for each.
- **Conditions**:
  - leave headroom in peak reserved memory for fragmentation and fluctuation; do not choose a combination right at the limit.
  - Saving more activations only helps a GPU-bound backward. If the backward is CPU-bound (Python overhead of recomputation, autograd scheduling), saving more GEMM outputs does not remove that overhead. Check the phase's GPU idle share before choosing the scope.
- **Proof**: per-role counters of "saved in forward" and "reused in recomputation", which should be equal.
- The gain from this item is usually a few percent, too small for one formal evaluation to resolve; estimate it first with interleaved A/B (see `train-opt-measurement`, "Measure small gains differently").

## What ships with it

- one switch per item, off by default, each passing the bitwise gate on its own;
- runtime counters printed at exit;
- peak allocated and reserved memory recorded per phase;
- an entry in the fast-path inventory: enabling conditions, memory cost, fallback behavior.
