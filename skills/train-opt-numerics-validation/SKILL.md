---
name: train-opt-numerics-validation
description: Numerics tiers N0–N5, how to judge a precision change, per-tensor diagnostics versus training-level comparison, and isolating the one non-bitwise change behind a deterministic stand-in. Read when changing precision or kernels, or when check reports INCORRECT.
---

# Numerics tiers and validation

Use when: assigning a precision tier while changing code, choosing validation levels and thresholds while formulating a task, and locating the cause after `check` reports INCORRECT.

**Acceptance is decided by the task's `bench.py check`; the verdict flow is in [`docs/judge.md`](../../docs/judge.md): structure → benefit (a speedup that does not exceed `max(1.02, 1 + 2 × relative gap between the two reference runs)` is no_benefit, and precision is not evaluated) → bitwise (every tensor bitwise identical in deterministic mode passes) → statistical (paired-seed short training trajectories in deterministic mode: start with 3 seeds; any endpoint beyond δ with p < 0.05 fails, all within δ pass, otherwise extend to 6 seeds and a mean beyond δ fails; δ = twice the reference's own seed-to-seed standard deviation, frozen in advance).** The per-tensor thresholds, fp64 arbitration and T0–T4 in this card are diagnostics for locating problems; they never decide acceptance on their own.

The core definitions are **relative**: a candidate may be no farther from the true value than the legitimate implementations of the baseline already are from each other, and the quality change it causes may be no larger than the baseline's own run-to-run fluctuation. A fixed tolerance table (say, a uniform rtol of 1e-2) is only a rule of thumb, and an agent will push precisely up to its edge.

## N0–N5

| Tier | Definition | Examples | Validation | Sign-off |
|---|---|---|---|---|
| N0 equivalent | With fixed seeds and caches, loss, gradients and weights of the first steps are bitwise identical | workers, pinning, removing syncs, async checkpointing, CUDA graphs that keep the same kernels | bitwise check + unchanged workload counters + proof the fast path was hit | automatic |
| N1 reordered | dtypes and accumulation precision unchanged, same real-valued function, but not bitwise identical | compilation, fused/foreach optimizers, a different GEMM library or SDPA backend, bucketing and communication algorithms | op-level k·e_B; single-step comparison of loss, all gradients and updated parameters; short trajectories within the baseline's own drift band | automatic + read-only reviewer |
| N2 precision change | same real-valued function, but the dtype or accumulation precision of an op, a state or a communication changed | TF32, wider bf16 coverage, FP8, low-precision norm/softmax, lower-precision optimizer state | implementation correctness: the N1 criteria against a baseline that simulates the same rounding points; then a paired multi-seed equivalence test and one paired medium-length run | human |
| N3 approximation | the real-valued function changes, but objective, data and hyperparameters do not | approximate activations, sparse or low-bit attention, truncation, a different decoder | non-inferiority tests on distributional metrics, looking at the worst layer | human (technical + quality) |
| N4 training dynamics | the trajectory changes by design | batch/LR scaling, a different optimizer, EMA changes, reusing intermediates across weight updates | speed measured as wall-clock to equal quality, multi-seed two-sided tests | research lead, never automatic |
| N5 evaluation and workload | the evaluation, objective, workload or timing itself changed | a smaller eval set, less frequent eval, skipped EMA or optimizer steps, cached products, altered timing | not eligible for acceptance | the agent always rejects and rolls back |

A tier is **measured, never declared**. An audit re-derives it in this order and takes the highest result:

- a change in the effective numerics switches or the dtype inventory: at least N2;
- a reviewer finds an approximation, a hard-coded constant or a shape-specific branch in the diff: at least N3;
- a change of hyperparameters, data or scheduling configuration: at least N4;
- a change of the contract fingerprint or the workload counters: N5;
- bitwise check passed and the audit finds nothing: N0; otherwise N1.

When several changes are stacked, the whole stack takes the highest tier.

## How thresholds are set

**Op level and single step**: for every compared quantity,

```
gate     = max(k × envelope, floor)
envelope = the largest difference from the reference, on the same case, step and metric, among several legitimate variants of the reference
k        = 2 for forward quantities (loss, reconstruction), 3 for backward ones (gradients, updates, optimizer state, EMA, buffers)
```

- Legitimate variants are other correct implementations of the same math, for example a rerun of the reference, a switch that only affects rounding turned off, or the previous implementation.
- **The variants must cover the modules the candidate changes.** If no variant touches a module, the gates for that module's metrics are too tight and a correct candidate that changes it will be rejected. Add variants according to which modules the candidate changes.
- Variants with different semantics do not belong in the envelope. A variant that synchronizes gradients differently from the reference, for example, can widen the gate by tens of times; exclude it and record why.
- Freeze the thresholds before seeing any candidate result and write their hash into the contract. Never change them afterwards, and certainly not to let a candidate pass.
- **When only a tolerance is exceeded and every exact check passes, escalate to fp64 arbitration first**: compare the candidate's and the reference's distances to an fp64 gold standard; if the candidate's error is at most k times the reference's, it is within reason. Do not reject outright and do not let it through outright.

**Training level**: paired seeds (baseline and candidate share seed, data order and initial state), comparing the paired differences at each endpoint; the decision rule is the statistical gate above.

- Freeze δ before looking at results, anchored on the baseline's own seed-to-seed fluctuation (two standard deviations).
- There are four outcomes: equivalent, different, inconclusive, and suspiciously good (significantly beyond +δ; review it as a possible N5).
- Compare every candidate with the **original baseline**, never only with the last accepted candidate; otherwise error accumulates round by round.

## Cheap investigation order (T0 → T3)

| Level | What | Cost |
|---|---|---|
| T0 | effective switches and dtypes, a diff of the compiler-generated code, machine code if needed | 0–2 GPU minutes |
| T1 | one step on the same input, per tensor against fp64 and the legitimate variants | about 1 GPU minute per variant |
| T2 | about 100 steps with a fixed compile cache, bisecting switch by switch, checking whether results stay bitwise identical | 3–4 GPU minutes per probe |
| T3 | paired multi-seed statistics | GPU hours |

Escalation: if T1 is bitwise identical it is N0, stop; if it is within the gates, check the trajectory with T2/T3; if it exceeds a gate, stop and fix. In practice most "why is it different" questions are pinned to a single switch and a single op within T0–T2 for a few tens of GPU minutes; keep the statistical test for changes that still need confirmation after localization.

## When only one component cannot be bitwise: isolate it behind a stand-in

Situation: in the final version only one change is inherently not bitwise, for example an attention backend with a different rounding order, while every other change (fusions, removed syncs, caches, communication scheduling) claims N0. If every change is validated in a configuration that includes that component, each one can only go through the statistical gate (GPU hours), and bugs in the N0 changes are hidden by the component's inherent differences.

Approach:

- Keep a bitwise-identical stand-in for that component behind a switch, usually the implementation the reference already uses.
- Validate every other change, and every combination of them, through the bitwise gate in the stand-in configuration to prove N0. Each time a new N0 change is merged, rerun the bitwise check of the combination in the stand-in configuration, to catch interactions (for example a fusion that changes another fusion's input layout).
- Run the statistical gate once, on the final combination with the real component.
- The stand-in and the real component must share the interface: same call sites, same input and output shapes and dtypes, only the one kernel replaced. Otherwise N0 conclusions from the stand-in configuration do not carry over to the final combination.

This does not contradict "deterministic switches are for diagnosis only" below. That rule says not to make a candidate "look identical" to the reference by turning on deterministic mode; the stand-in is the reference's own implementation, used to take the one known non-bitwise change out of validation, and the N0 conclusions for the other changes remain genuine bitwise equality with the reference.

## Common pitfalls

- **Two runs with the same seed can take two numerical paths.** Autotuning and `cudnn.benchmark` may pick different algorithms in different processes. Bitwise comparisons need a fixed compile cache directory and a record of whether the cache was cold or warm; "bitwise identical with the switch off" only means something under the same cache.
- **Deterministic switches change the numerical path.** Use them for diagnosis, never to "make results agree".
- **Compare tensors, not just the scalar loss.** A mean-reduced loss hides ulp-level differences that low-precision GEMMs then amplify.
- **The per-GPU batch of a single-GPU dump must match real training**, otherwise kernel shapes and algorithm choices change.
- **Library defaults may not be the precision you think.** Convolutions allow TF32 by default even when matmul TF32 is off, for example. Verify the effective settings behind a so-called fp32 baseline.
- **A seemingly equivalent refactor can become N1 after compilation.** Saving an intermediate separately, for example, moves a fusion boundary, and a mean subtraction turns from "round, then subtract" into a single FMA.
- **Chaotic phases (such as GAN training) break pairing.** Do statistical validation in a stable phase; for chaotic phases, either run short paired runs from the same checkpoint or use a full long run.
- **A short window only proves "nothing broke early".** A step count chosen for cost may cover only a small part of the learning-rate warmup. State which part of training the window covers; it does not support a claim that convergence is unchanged.
- **Check phase switches when resuming from a checkpoint.** If phases are counted in epochs and the batch or GPU count changed, a resumed run may silently skip or disable a phase.
- **Cross-rank invariants**: a DDP-wrapped module called through `.module` does not synchronize gradients, and rewriting the backward pass can make the next backward synchronize unexpectedly. Confirm unchanged synchronization semantics with per-rank gradient checksums.
- **Keep the evaluator where the agent cannot write**, and check its hash. A lenient or modifiable evaluator will be exploited.
- **The evaluator itself can have bugs.** A common one is a verdict path that is not wired up: an arbitration function computes a result, but the return value never reaches the final verdict, and every candidate "passes". Before freezing, use mutation tests to prove that every failure path fires (the negative self-tests during formulation, and `python -m numerics selftest`).
- **Writing a cache in place and then overwriting it under no_grad leaves a "ghost gradient".** Autoregressive rollouts often do this: one step writes K/V into the cache in place during a forward with gradients, and a later no_grad forward overwrites the same slice with other values. Later blocks read the overwritten values, but the no_grad overwrite is not in the graph, so the cache tensor still carries the node of the original write with gradients (`CopySlices`). In backward, gradient flows through it into the K/V that was overwritten, and from there into every parameter. The forward loss is unaffected, but the gradient gains an extra term that is not the gradient of the computed loss.
  - How to check: write a minimal reproduction (one attention layer, a few blocks, the same write → overwrite → read order) and compare in fp64 the autograd gradient with finite differences that treat the overwritten slice as a constant.
  - **Principle: this is upstream semantics.** Keep it while optimizing, otherwise the candidate stops being bitwise identical to the reference; report the finding and the reproduction to the owner, who decides whether to fix it, and do not fix it in passing inside an optimization candidate. Optimizations (removing a copy, changing the cache layout) must not change this gradient path by accident either: the bitwise gate will catch it, and when it does, treat it as a semantic change, not as a rounding difference.
- **Parameters whose true gradient is always zero cannot be compared by relative error.** An example is the key bias in attention: it adds the same number to every logit of a row, softmax is invariant to that, so its gradient is exactly zero in exact arithmetic and what gets computed is rounding noise, whose relative error is always "out of tolerance". For such parameters either exclude them from the comparison with a stated reason, or compare absolute error only. To decide, check whether the gradient norm in fp64 is at rounding-noise level.
