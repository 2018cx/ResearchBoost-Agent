# DMD2: supervision record

*Translated from the Chinese original.*

One Codex `/goal` session (4 × H20) optimized DMD2 (SDv1.5 one-step distillation + GAN head, the upstream recipe published with FID 8.35); one supervisor formulated, launched and watched it. Target 3×, minimum 2×. Workspace layout: candidate `dmd2/` (branch `opt`), reference `targets/reference` (tianweiy/DMD2 `8d8fa55`).

## Results

Full check (candidate `f7d7a31`, 70 rounds within the 24-hour budget):

| Item | Value |
|---|---|
| status | **PASS** |
| performance.speedup | **1.7218×** (3.7368 → 2.1704 s/iteration; relative gap of the two reference runs 0.38%) |
| Gate 1 | window ratios of the three runs vs the frozen reference +0.34% / +0.21% / −0.06%, no issues |
| Gate 2 | tier **N2**; 8337 of 41661 tensors bitwise identical, 0 exact-item failures (not bitwise, so on to gate 3) |
| Gate 3 | **equivalent**, 3 seeds, all 7 endpoints within δ (the largest shift, `gan_classifier_loss`, +0.0029 against δ 0.0156) |
| Integrity | frozen inputs match at the start and at the end; source digest unchanged across the run |
| Cost | 4.62 GPU-hours for this check |

**Correct and faster, but below the 2× minimum: the goal was not reached.** The changes on branch `opt`, in order: cuDNN benchmark, merged scalar gathers, bf16 autocast for the no-grad UNet forwards, two redundant `zero_grad` calls removed, gradient checkpointing narrowed step by step, torch.compile on both UNets' forward (with `optimize_ddp=False`), batched classifier bottleneck.

## Formulation

- The workload changed from SDXL 4-step to SDv1.5: SDXL's FSDP path does not run on this machine with any installable torch (the torch 2.0.1 upstream pins hits an illegal instruction in flash-attention's backward on H20; 2.1.2 raises SIGFPE inside cuBLAS; FSDP ≥2.1 rejects the recipe toggling `requires_grad` on the guidance model, upstream issue #41). The SDv1.5 recipe uses DDP, which is the parallelism upstream specifies for it. Details in [`GOAL.md`](GOAL.md).
- One machine with 4 GPUs, with the same per-GPU load as upstream's 8 machines and 64 GPUs (32 per GPU, 512², bf16 autocast, gradient checkpointing, guidance 1.75, resumed from step 39k); global batch 128.
- Baseline 3.73 s/iteration, range across three runs 0.14%, peak 69.6 GiB per GPU (of 95 GiB).
- In strict deterministic mode the reference is bitwise identical across two recordings (41661/41661). The bitwise gate has to raise numerics' `warn_only=True` to strict determinism, otherwise atomic accumulation in flash's backward leaves only 9565/41661 identical for the reference itself.
- All seven acceptance items matched the expectations and `numerics selftest` passed: mathematically equivalent changes pass (bitwise gate, or statistical gate equivalent), and changes to training semantics fail (exact items, or the statistical gate). Script: [`formulation/prep/accept.sh`](formulation/prep/accept.sh).

## Risks to watch from formulation onwards

- Pace: at most 30 minutes per round, profile before changing, speed reported only through bench.py, a WORK_LOG entry every round.
- Random stream: gate 1 checks it iteration by iteration and rank by rank; a candidate that changes the draw order gets ERROR, and every ERROR needs its cause confirmed.
- Compile cache: the bench gives every run a new cache directory, but a candidate that hard-codes a fixed cache path in its code (bypassing TMPDIR) can turn a "cold compile" into a warm one. Check diffs for hard-coded cache paths.
- Read-only paths: `bench/`, `numerics/`, `task/` and `targets/` are read-only in the sandbox; the git state of `targets/reference` must stay clean.
- Memory: the reference peaks at 69.6/95 GiB, leaving the candidate little room; OOM is ERROR.

## Interventions (all at the level of rules; no optimization direction was ever given)

1. WORK_LOG covered only up to round 4, with rounds 5–11 missing → reminded to fill them in according to `GOAL.md` and to log every round (profile findings, change, time, check, start and end times, failures and rollbacks too). Every round complied from then on.
2. A round was cut off by "Selected model is at capacity" from the model endpoint, and the TUI sat at an empty prompt → reminded to first add this round's conclusion (OOM) to the WORK_LOG, roll the change back cleanly, then start the next round.
3. Another capacity stall → sent `/goal resume`, and told it that the previous round's profile had completed and did not need rerunning.
4. About 2 hours before the end of the time budget → asked it to run a **full check** on the best committed candidate (a `--quick` PASS cannot be the final result) and to write the wrap-up summary at the end of the WORK_LOG.

Capacity problems at the model endpoint caused about 1.5 hours of stalls over the 24 hours; [`goal/supervise/babysit.sh`](../../goal/supervise/babysit.sh) was written for this. At the end of the time budget the agent was still advancing round by round; it was not stopped, and the decision was left to the owner.

## What the judge actually caught

- **Random stream**: replacing attention in the three UNets with xformers made the random stream diverge from the reference from iteration 0; gate 1 reported ERROR, and the agent withdrew it itself.
- **Cross-GPU synchronization changed**: DDP `static_graph=True` measured 1.7311×, but 12 `cross_rank_pattern` exact items of the bitwise gate failed: the guidance gradients, updates and optimizer state that are equal across GPUs in the reference were no longer equal.
- **Compilation inside the timing window**: the `compiles_in_window` guard rejected a run that measured 5.922 s/iteration in a window.
- **Memory**: removing more recomputation, `max-autotune`'s extra workspace, and `reduce-overhead`'s private CUDA Graph pool (about 49 GiB) all ran out of memory and were ERROR.

## Known limits of the judge

The exact items "number of forward calls per model" and `capture_outputs` (outputs recorded by call index) reject the mathematically equivalent "merge several forwards into one batched forward", and whether they do depends on whether the merged calls go through `__call__`:
- merging the inputs of the two classifier UNet calls (which also merged two `__call__`s of `cls_pred_branch`) → INCORRECT, about 1.4% of gain not obtained;
- batching the classifier bottleneck (calling `.forward` directly, invisible to the counting hook) → PASS, in the final version;
- merging the guidance epsilon-loss inputs with the classifier real/fake inputs into one down/mid pass (the fastest of the whole run at 1.7536×) → INCORRECT, missing `fake_unet` call outputs.

The judge was not relaxed for this during supervision. Allowing "batching several forwards" in the future requires changing how the adapter counts calls and re-recording the reference records.

## Lessons

- Memory is this task's hard constraint: four OOMs, and once the headroom was used up, "one less recomputed layer" was the end of the road.
- The candidate's own run-to-run noise is about 0.3%: retiming the same committed code gave 1.7107×, against 1.7157× originally recorded. Changes of 0.1–0.3% cannot be resolved by a single `time`, which explains why the last 30-odd rounds brought no gain.
- A capacity error can hit the request **after** a round's profile: that round actually completed, so remind the agent to log it rather than rerun it.

## Handoff

- Results: the wrap-up summary at the end of `ws/WORK_LOG.md`, and the final check's `check.json`.
- Resume: `goal/supervise/relaunch_resume.sh dmd "<gpus>" <thread id>`, then type `/goal resume`; when capacity errors are frequent, run `goal/supervise/babysit.sh dmd "<gpus>"` in the background.
- Stop: `/stop` in the TUI to stop background tasks, then `tmux kill-session -t dmd_goal`.
- Rebuild the environment and data: in the order given in [`formulation/prep/README.md`](formulation/prep/README.md).
