# Goal

*Translated from the Chinese original the agent received.*

Bring the **time per training iteration** of DMD2 (SDv1.5 one-step distillation + GAN head, the upstream recipe published with FID 8.35) on 4 H20s **down to 1/3 of the reference (a 3× speedup)**, with a minimum of 2×, while passing the numerics checks below.

- Training code: `dmd2/`, upstream tianweiy/DMD2 at commit 8d8fa55, branch `opt`; this is where you make changes.
  - The entry point is `main/train_sd.py`: `args = parse_args(); trainer = Trainer(args); trainer.train()`.
  - The bench launches it from a copy of the tree with `torchrun --nnodes=1 --nproc-per-node=4` (the same launch as the upstream scripts); the tree root is first on `sys.path` (upstream relies on `setup.py develop` to make `main.*` importable).
- The workload is a frozen recipe, see `bench/recipe.py`. It is upstream `experiments/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_noode_resume_fixdata.sh` (the best SDv1.5 row in the model zoo, FID 8.35), changed only to one machine with 4 GPUs:
  - upstream uses 8 machines × 8 GPUs with torchrun + DDP (not FSDP); here it is 1 machine with 4 GPUs and the same torchrun + DDP.
  - **The per-GPU load is the same as upstream**: batch 32 per GPU, 512×512 (latent 64×64), fp32 weights + bf16 autocast, gradient checkpointing on, a one-step generator (conditioning timestep 999), one generator update every 10 iterations, the GAN classification head on clean images with diffusion-GAN noise, real guidance scale 1.75.
  - The global batch is 128 (upstream 2048). Upstream `train_sd.py` asserts `gradient_accumulation_steps == 1`, so accumulation is impossible without changing the training code, and the global batch is not matched.
  - The recipe resumes from the upstream-published step-39000 checkpoint (`--ckpt_only_path`), so `trainer.step` starts at 39000, as upstream.
  - Arguments the bench adds: `--log_path/--cache_dir/--output_path` pointing to temporary directories, and `--no_save` (upstream saves a checkpoint every 1000 iterations, never inside a timing window). wandb is replaced by an offline stub (`bench/wandb_stub.py`).
- The models and what one iteration does (`Trainer.train_one_step`):
  - **Generator**: the SDv1.5 UNet (860M), fp32 master weights + AdamW (lr 5e-7, betas 0.9/0.999, weight decay 0.01), one step: one forward on pure noise at t=999, then `get_x0_from_noise` gives the generated image. In every 10 iterations, 1 runs with gradients (the generator-update iteration) and the other 9 run under `torch.no_grad()` (with gradient checkpointing temporarily off).
  - **Guidance model**: real UNet (bf16, frozen, CFG 1.75), fake UNet (fp32 + AdamW, bf16 autocast), GAN classification head (on the fake UNet's bottleneck).
  - **Generator-update iteration** (`step % 10 == 0`): DMD distribution-matching loss (a no-grad fake UNet forward + a real UNet CFG forward, with the gradient built as `(p_real - p_fake)/|p_real|`) + 1e-3 × the GAN generator loss, backward into the generator, gradients clipped to 10, one AdamW step, then both optimizers' gradients zeroed.
  - **Every iteration** has one guidance update: the fake UNet's epsilon-prediction loss + 1e-2 × the GAN classification loss (real latents and generated latents each go through the classification head once, each with its own random diffusion-GAN noise), gradients clipped, one AdamW step.
  - Every 50 iterations there is an image log (VAE decode + plotting), outside the timing windows.
- Reference: `targets/reference`, the same commit, code unchanged.
  - The reference takes **3.73 s per iteration** in the timing windows (37.3 s per 10-iteration cycle), with a peak of 69.6 GiB per GPU and about 26 s from startup to the first iteration.
  - The range across three independent reference runs is 0.14%, and the per-GPU, per-iteration random streams are identical across all three.
  - The baseline profile and the shares of each part are in `task/baseline.md`.

## Why SDv1.5 and not SDXL

The upstream SDXL 4-step recipe trains under FSDP, and that path does not run on this machine with any installable torch (details in `bench/recipe.py` and `task/baseline.md`):
- the torch 2.0.1 that upstream pins (wheels only for cu117/cu118) fails in flash-attention's backward on H20 (sm_90) with `CUDA error: an illegal instruction was encountered`;
- torch 2.1.2+cu121 raises SIGFPE inside cuBLAS (a bf16 linear layer without bias, `(4,77,2048)×(1280,2048)`, exactly the SDXL cross-attention shape);
- FSDP in torch ≥ 2.1 rejects this recipe toggling `requires_grad` on the guidance model back and forth within an iteration, with `Invalid: ``_post_backward_hook_state``` (upstream issue #41, still unresolved).

The SDv1.5 recipe uses DDP, none of these three problems exist, and DDP is the parallelism upstream specifies for this recipe. The environment is torch 2.4.1+cu124 (other versions as upstream: accelerate 0.23.0, huggingface-hub 0.22.0, diffusers 0.28.2, transformers 4.40.2).

## What counts as reaching the goal (the single criterion)

```
python bench/bench.py check --tree dmd2 --config <switch file> --out <output dir>
```

It writes `<output dir>/check.json`.

- **Done**: `status` is `PASS` and `performance.speedup` ≥ 3.0.
- If profile evidence shows that 3.0 cannot be reached within the rules, `status` `PASS` with `performance.speedup` ≥ 2.0 is also acceptable for wrapping up. In that case, write down in `WORK_LOG.md` why each of the remaining hotspots cannot be addressed.

`check` applies the gates below in order and stops at the first that fails:

0. **Structure (coverage, CPU, about 2 minutes)**
   - `parse_args()`, the `if __name__ == "__main__":` block, and everything in `Trainer.train()` except the body of the `for index in range(...)` loop in `main/train_sd.py` must have the same AST as the reference. Different → ERROR.
     - Reason: the numerics adapter (`bench/dmd_adapter.py`) builds `args` and `Trainer(args)` the way the reference does, and runs your `train()` loop body unchanged at every step. Setup changes go where the training code calls them (`Trainer.__init__`, `train_one_step`, the model modules), or into the loop body.
   - The tree's dataset classes (`SDTextDataset`, `SDImageDatasetLMDB`) must produce bitwise-identical samples and first batches to the reference on the frozen fixture, otherwise INCORRECT.
1. **Benefit (about 12 minutes)**: reference → candidate → reference, three independent runs of the real `torchrun` entry point with cold compile caches.
   - Each run has 10 warmup iterations + 2 windows of 20 iterations (each window 2 full 10-iteration cycles). `performance.speedup` = the mean total window time of the two reference runs ÷ the candidate's.
   - A fingerprint (command, math mode, GPU UUIDs, library versions, timing plan) that differs from the first reference run → ERROR.
   - The random stream at the end of each iteration must match the frozen reference iteration by iteration and rank by rank, otherwise ERROR: the order in which random numbers are drawn determines what each iteration does (noise timesteps, diffusion-GAN timesteps, noise), so a changed stream means changed work and semantics.
   - Interference makes a run INVALID, to be rerun a little later: the reference deviating from the frozen reference by more than 5%; the same window differing by more than 3% between the two reference runs; the ratio of a run's two windows deviating from the frozen reference's ratio by more than 3% (10% for the candidate).
   - A speedup that does not exceed max(1.02, 1 + 2 × relative gap between the two reference runs) → UNAVAILABLE (no_benefit), rejected outright, without a precision evaluation.
2. **Bitwise (about 4 minutes)**: in deterministic mode (`torch.use_deterministic_algorithms`, cuDNN deterministic, `CUBLAS_WORKSPACE_CONFIG`), record two 1-step cases, one process per case:
   - `gen`: `trainer.step = 39000`, one generator-update iteration (generator update + guidance update);
   - `guidance`: `trainer.step = 39001`, one guidance-only iteration.
   - Every exact item must hold, otherwise INCORRECT: phase, RNG fingerprint before and after each step, number of model forward calls, invariants (learning rate, step counter, parsed arguments, the digest of every batch the four data iterators emit in this step), tensor names and shapes, all values finite, cross-rank pattern (both models declare `replicated`: under DDP every GPU's parameters and gradients must agree, and silently skipping synchronization is caught here).
   - All recorded tensors (`loss/`, `out/`, `grad/`, `delta/`, `optim/`, `buffer/`) bitwise identical → **PASS (N0)**, accepted directly.
     - Tensors larger than 4096 elements are stored as a linear CountSketch (with exact norms and hashes stored separately), so "bitwise identical" is still a bitwise judgment, without storing 3.4 GB of weights per step.
   - Order of judgment: **exact items first**. A failed exact item is always INCORRECT (the candidate does something different); only when every exact item holds and the tensors still disagree
     is it ERROR. "The candidate records fewer tensors than the reference" (typically: a loss term's gradient was removed, so those parameters have no gradient at all)
     counts as an exact-item failure, not ERROR.
   - The reference is bitwise identical to itself across two recordings in deterministic mode (verified during formulation).
3. **Statistical**: only when gate 2 is not bitwise identical.
   - In deterministic mode, run paired short training trajectories with the 6 frozen seeds: 100 iterations from step 39000, the natural generator/guidance alternation (1 generator in every 10 iterations), lr warmup compressed to 20 steps, one process per seed, about 7 minutes per seed.
   - 7 endpoints (the mean over the last 50 iterations, averaged across GPUs): `loss_fake_mean`, `guidance_cls_loss`, `guidance_grad_norm`, `loss_dm`, `gen_cls_loss`, `generator_grad_norm`, `dmtrain_gradient_norm` (the last four from the 5 generator iterations in each window).
   - δ = twice the standard deviation of the reference across its own seeds (6 seeds, frozen in `bench/traj/frozen.json`).
   - Start with 3 seeds: any endpoint deviating by more than δ and significantly (paired t-test p < 0.05) → INCORRECT; all within δ → PASS; otherwise extend to 6 seeds, and then any endpoint whose mean is beyond δ fails.
- **Tier**: N0 = everything bitwise identical with the same dtypes and math modes; N2 = the dtype or math mode of any tensor changed (TF32, low-precision reductions, SDPA backend); N1 otherwise. Beyond N2 is INCORRECT.
- Per-tensor tolerances (`bench/thresholds.json`) and fp64 arbitration are diagnostic output only and do not enter the verdict.

Two other modes:

- `--quick`: gates 0 and 2 only (exact items and bitwise, no timing, no trajectories), about 6 minutes.
- `--skip-timing`: skips gate 1.

Time for one full check: gate 0 about 2 minutes; gate 1 about 12 minutes; gate 2 about 4 minutes; gate 3 (only when not bitwise) about 21 minutes for 3 seeds, about 42 for 6. So screen with `time` and `check --quick` day to day, and run the full check once confident.

## What every round must do

1. **Profile first, then change.** Before each round's change, profile the current best version and choose the hotspot for this round from the profile:
   ```
   python bench/bench.py profile --tree dmd2 --config <switch file> --out runs/<round>/profile [--nsys] [--iters N]
   ```
   - By default torch.profiler records 10 iterations (one full cycle) after 10 warmup iterations and writes `profile.json`, `profile.md` and a Chrome trace (rank 0): GPU busy share, kernel time by category, the top few dozen kernels, forward GPU time inside each model call (generator, guidance_model, real_unet, fake_unet, GAN head, text encoder, VAE), CPU-side op self time, and each iteration's host time and kind.
   - `--nsys` records the same span with Nsight Systems instead (one NVTX range per iteration); break it down to kernels with `nsys stats` / `nsys export`.
   - For finer detail you may write small scripts that run only under the GPU lock for op-level breakdowns (wrap them in `flock /tmp/dmd_gpu.lock <command>`); they are for localization only, never for reporting speed.
   - In the first round, profile once before any change (at that point `dmd2` is the reference).
2. **Time only with bench.py**:
   - time on its own with `python bench/bench.py time --tree dmd2 --config <switch file> --out <dir>`, which also reports the speedup against the frozen reference (`speedup_vs_frozen_reference`);
   - use `check` for the formal verdict;
   - do not write your own timing scripts, and do not report speed any other way.
3. Every round appends four items to `WORK_LOG.md`:
   - profile findings: the top hotspots and their shares;
   - what was changed;
   - the `time` result;
   - the `check` result (if run).

## Pace: at most 30 minutes per round (mandatory)

- A round has four steps:
  1. profile;
  2. pick one hotspot and change it;
  3. re-time with `bench.py time`, adding `check --quick` when needed;
  4. log the round in `WORK_LOG.md` with start and end times.
- Each round's wall-clock time is at most 30 minutes, counted from taking the GPU lock. If the time is up before you finish, still wrap up: record this round's conclusion (failures and no-gain rounds too), then start the next round.
  - Do not pile several hotspots into one round;
  - do not spend long periods reading without measuring.
- Approximate times of the bench commands: `profile` about 5 minutes, `time` about 4 minutes, `check --quick` about 6 minutes. About 30 s of each is startup (loading the two UNets, reading the step-39000 checkpoint, DDP wrapping).
- The full `check` does not count towards a round. Run it only when the accumulated changes, as measured by `time`, are already enough for 2× or 3×. While it holds the GPUs you may do preparation that needs no GPU, but do not present untested changes as results.
- Changes that do not depend on each other can each get a switch (an environment variable) and be timed separately in the same round with different switch files, then combined according to the results.

## Constraints

- **Training semantics must not change** (gate 0 compares structure, gates 2 and 3 compare numerics); specifically:
  - data and order, per-GPU batch, resolution, models, learning rate and optimizers (AdamW hyperparameters, gradient clipping at 10), the composition and weights of the losses, guidance scale, the sampling range of timesteps;
  - the update schedule: one guidance optimizer step every iteration, one generator optimizer step every 10 iterations, in the same order (within an iteration, generator before guidance);
  - **the order and use of random draws**: the bench checks the random stream iteration by iteration and rank by rank (gate 1) and the RNG fingerprints before and after each step (gate 2).
  - It must stay on-policy: every iteration generates with the current weights; reusing generated results across iterations or lagging by one step is not allowed. Within an iteration the stages may overlap or be pipelined, as long as the result has the same semantics as serial execution.
- **Precision**:
  - The baseline is the upstream recipe's precision: fp32 master weights and fp32 optimizer state, bf16 autocast, the real UNet in bf16, the places where upstream explicitly computes in fp32 / fp64 (the no-grad generator forward and the fake UNet forward in DMD outside autocast, `get_x0_from_noise` in fp64), TF32 on.
  - Lowering to FP8 / INT8 / FP16 is not allowed, nor is removing the fp32 master weights or fp32 optimizer state.
  - Local precision changes are allowed, but their tier may not exceed N2, and they must pass the statistical gate.
- **Deterministic mode**: the bench enables torch's deterministic algorithms in gates 2 and 3. If a kernel you add has no deterministic implementation, it must take a deterministic path of the same precision when `torch.are_deterministic_algorithms_enabled()` is true (for example the same kernel's deterministic option, or a fallback to PyTorch's implementation at the same precision); this must not be used to change precision. Apart from that, the code may not read `DMD_BENCH_*` environment variables, and may not behave differently depending on whether it is being timed or judged.
- **Keep the bench's hook points**:
  - the loop `for index in range(self.step, self.train_iters):` in `Trainer.train()` (the adapter runs its body unchanged at every step, and `self.step` increases by 1 per iteration in the body);
  - the `Trainer` attributes `model` (`SDUniModel`, with `model.feedforward_model` the generator and `model.guidance_model` the guidance model), `optimizer_generator`, `optimizer_guidance`, `dfake_gen_update_ratio`, `step`, `args`, and the three data iterators `dataloader`, `guidance_dataloader`, `real_dataloader` (the adapter wraps them to record digests; take data with `next()`);
  - two `self.model(...)` calls per iteration (one with `generator_turn=True`, one with `guidance_turn=True`), returning `(loss_dict, log_dict)`, with unchanged `loss_dict` keys (`loss_dm`, `gen_cls_loss`, `loss_fake_mean`, `guidance_cls_loss`);
  - both optimizers are `torch.optim.Optimizer` (the implementation may change, for example to fused/foreach, but the bench delimits iterations with a global optimizer-step hook: every iteration ends with one guidance `step()`, preceded by the generator's `step()`);
  - the main process calls `wandb.log(..., step=self.step)` once per iteration with upstream's scalars (some statistical-gate endpoints come from there).
- **Switch file** format is `{"environment": {"X": "1"}}`: variables the bench manages itself may not be set (`reserved_env` in `bench/bench.json`: temporary directories, compile caches, devices, `PYTHONPATH`, HF offline, torchrun launch variables and so on).
- **Read-only paths and network**: `bench/`, `numerics/`, `task/`, `targets/` and `$ASSETS` are read-only; there is no network, nothing can be downloaded, and no new packages can be installed. You may write your own Triton kernels or compile CUDA extensions with `torch.utils.cpp_extension` (`CUDA_HOME=/usr/local/cuda-12.6`).

## Environment

- 4 H20s (Hopper, sm_90, about 96 GB each, NVLink), numbered 0–3 in the sandbox. The GPU lock is `/tmp/dmd_gpu.lock`:
  - `bench/bench.py` takes the lock itself; do not wrap it in flock;
  - wrap any other command that uses the GPUs as `flock /tmp/dmd_gpu.lock <command>`.
- Python: `$ASSETS/env/bin/python` (first on PATH). Main versions: torch 2.4.1+cu124, triton 3.0.0, cuDNN 9.1, NCCL 2.20.5, diffusers 0.28.2, transformers 4.40.2, accelerate 0.23.0, huggingface-hub 0.22.0, peft 0.10.0, xformers 0.0.28.post1, numpy 1.24. The full list is in `$ASSETS/env_freeze.txt`.
  - Why not upstream's pinned torch 2.0.1: see "Why SDv1.5" above.
  - Optional performance libraries are installed too (the reference recipe does not use them): xformers 0.0.28.post1, triton 3.0.0 (bundled with torch), ninja, py-spy.
- Nsight Systems: `nsys` is on PATH (2025.6.3). Nsight Compute (`ncu`) is in `/usr/local/cuda-12.6/bin`, but this container has no access to performance counters, so it most likely does not work.
- `TMPDIR=/tmp` (local disk, private, plenty of space). Delete your own experiment outputs once you no longer need them.
- Start long jobs with `setsid nohup ... &` and wait for them inside the session.
- The sandbox has its own network namespace with only 127.0.0.1, and hostnames do not resolve: when starting distributed processes yourself, pass `--master-addr 127.0.0.1` explicitly.
- HF weights are in a read-only cache, in offline mode (`HF_HUB_OFFLINE=1`).
- The frozen reference artifacts are under `task/` and readable:
  - `task/timing_ref.json`: the reference timing (every window, per-GPU per-iteration random fingerprints);
  - `task/baseline.md`: baseline numbers and profile;
  - `task/formulation.json`, `task/loader_digest.json`: fingerprints of the frozen inputs and the data-loading digests;
  - `bench/records/ref`, `bench/traj/`: the reference's bitwise records and trajectories.

## References (skills)

These skills are in `~/.codex/skills/`; read the relevant ones before choosing what to try:

- `train-opt-tiered-profiling`, `train-opt-measurement`, `train-opt-action-ladder`, `train-opt-numerics-validation`, `train-opt-fusion-patterns`, `train-opt-fsdp-multi-forward`;
- `sglang-diffusion-benchmark-profile`, `sglang-diffusion-performance`.
- They are priors summarized from other training-optimization tasks, covering tiered profiling, timing pitfalls, the action ladder, numerics verdicts, fusion patterns, parallelism with multiple forwards, and profiling and performance of diffusion models.
- Re-verify these priors on this task before treating them as conclusions.

## Wrap-up

- Commit each change separately in `dmd2/` (branch `opt`), with a message stating what changed, the tier and the `time` result.
- Once the goal is reached, write a summary at the end of `WORK_LOG.md`, including:
  - the final switch file, and the path and result of the final check;
  - the breakdown of gains by change;
  - directions that did not work and why.
