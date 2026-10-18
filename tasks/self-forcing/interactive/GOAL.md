# Goal

*Translated from the Chinese original the agent received.*

Bring the **time per iteration** of Self-Forcing's DMD training on 2 H200-class GPUs **down to 1/3 of the reference (a 3× speedup)**, with a minimum of 2×, while passing the numerics checks below.

- Training code: `sf/`, upstream guandeh17/Self-Forcing at commit 33593df; this is where you make changes. Entry point `train.py`, configuration `configs/self_forcing_dmd.yaml`.
- Workload:
  - a causal Wan-1.3B generator (starting from `checkpoints/ode_init.pt`, autoregressive rollout with a KV cache, 7 blocks of 3 latent frames each, random exit steps);
  - a frozen Wan-14B teacher (real score, two CFG branches);
  - a trainable Wan-1.3B critic (fake score);
  - a frozen UMT5-XXL text encoder;
  - everything under FSDP (HYBRID_SHARD, bf16 mixed precision). Batch 1 per GPU, latent `[1, 21, 16, 60, 104]`.
- One iteration is one pass of the loop in `ScoreDistillationTrainer.train()`. 1 in every 5 is a generator iteration (update the generator first, then the critic); the other 4 update only the critic.
- Reference: `targets/self_forcing/reference`, the same commit unchanged. The reference takes about 10400 ms per iteration (2 GPUs, batch 1 per GPU, three independent timings 10245–10617 ms; generator iterations about 23 s, critic iterations about 6–8 s). Peak memory 93.5 GiB allocated / 128 GiB reserved.

## What counts as reaching the goal (the single criterion)

```
python bench/bench.py check --tree sf --config <switch file> --out <output dir>
```

It writes `<output dir>/check.json`.

- **Done**: `status` is `PASS` and `performance.speedup` ≥ 3.0.
- If profile evidence shows that 3.0 cannot be reached within the rules, `status` `PASS` with `performance.speedup` ≥ 2.0 is also acceptable for wrapping up. In that case, write down in `WORK_LOG.md` why each of the remaining hotspots cannot be addressed.

`check` applies the gates below in order and stops at the first that fails:

0. **Structure**:
   - the AST of `train.py` must match the reference, and `configs/` must be byte-identical;
   - your trainer records a short 4-iteration case in deterministic mode: a generator iteration, a critic iteration, the iteration that creates the EMA, and the iteration that updates the EMA. These exact items must match the reference: iteration kind, step counter, the prompts fetched, random fingerprints, optimizer step counts, learning rates, top-level forward call counts of generator / fake_score / real_score / text_encoder, cross-rank pattern, tensor names and shapes.
1. **Benefit**: reference–candidate–reference timing in three independent processes (the real entry point `train.py`, 10 warmup iterations, then 2 windows of 10 iterations each, each window 2 full 5-iteration cycles). The speedup must exceed max(1.02, 1 + 2 × relative gap between the two reference runs). `performance.speedup` is the reference mean ms per iteration divided by the candidate's ms per iteration.
2. **Bitwise**: if every tensor of the 4 iterations recorded at gate 0 is bitwise identical, the candidate passes directly.
3. **Statistical**: if bitwise is not achieved, run paired-seed training trajectories of 100 iterations (upstream configuration, starting from ode_init, deterministic kernels) and look at 4 endpoints, the means over iterations 50–100 of critic_loss, critic_grad_norm, generator_loss and generator_grad_norm:
   - start with 3 seeds: if any endpoint deviates by more than δ (twice the reference's own seed-to-seed fluctuation, frozen) and significantly, it fails; if all are within δ, it passes;
   - otherwise extend to 6 seeds and decide again (now a mean beyond δ fails).

Two other modes:
- `--quick`: gate 0 only, results in about ten minutes;
- `--skip-timing`: skip timing and judge numerics only.

A full check is expensive: timing about 3 × 7 minutes, the short recording about 8 minutes, the statistical gate about 75 minutes for 3 seeds and about 150 for 6. So screen with `time` and `check --quick` day to day, and run the full check once confident.

## What every round must do

1. **Profile first, then change.** Before each round's change, profile the current best version and choose the hotspot for this round from the profile:
   ```
   python bench/bench.py profile --tree sf --config <switch file> --out runs/<round>/profile
   ```
   - It captures an Nsight Systems trace of one timing window (10 iterations, including 2 generator iterations), with one NVTX range per iteration;
   - it writes `profile.json`, `top_kernels.md` (kernels sorted by GPU time), `nsys_*.csv` (kernel, NVTX, memcpy and CUDA API summaries) and the `.nsys-rep`;
   - for finer breakdowns, analyze the `.nsys-rep` with `nsys stats` / `nsys export`.
   - In the first round, profile once before any change (at that point `sf` is the reference).
2. **Time only with bench.py**:
   - time on its own with `python bench/bench.py time --tree sf --config <switch file> --out <dir>`;
   - use `check` for the formal verdict;
   - do not write your own timing scripts, and do not report speed any other way.
3. Every round appends four items to `WORK_LOG.md`: profile findings (the top hotspots and their shares), what was changed, the `time` result, the `check` result.

## Constraints

- **Must not change**:
  - `train.py`, and any file under `configs/`;
  - hyperparameters, per-GPU batch, data, prompts, latent shape, the update ratio;
  - the training algorithm itself: the rollout's random exit, the denoising step table, the teacher's two CFG forwards, the critic regenerating its training samples, the EMA semantics (fp32 shadow weights);
  - the order in which random numbers are consumed; gate 0 compares the random fingerprints item by item.
- **The FSDP sharding must not change**: sharding strategy, wrapping and `use_orig_params` stay as they are. The short recording compares parameter shards per rank, so different shards do not match and give ERROR.
- **The number of top-level forward calls of each model must match the reference**, so the teacher's cond / uncond branches cannot be merged into one batched call.
- **The bench relies on these two points to drive training and read losses; keep them**:
  - every iteration takes batches from `self.dataloader` in the original order (2 for a generator iteration, 1 for a critic iteration);
  - `fwdbwd_one_step(batch, train_generator)` stays, and its returned dict keeps the original keys such as `generator_loss` or `critic_loss`.
- **Precision**:
  - the reference already uses FSDP bf16 mixed precision: bf16 parameters in compute, fp32 master weights, fp32 gradient reduction.
  - Lowering the whole training step to FP8/FP16 is not allowed, nor is removing the fp32 master weights and optimizer state.
  - Local precision changes are allowed but must pass the checks; the upper numerics tier is N2.
- **Deterministic mode**: gate 0's short recording and gate 3's trajectories run in deterministic mode. The bench enables `torch.use_deterministic_algorithms` and forces flash-attn's deterministic backward through a `flash_attn` module attribute. If you replace the attention implementation, you must take a deterministic path yourself when `torch.are_deterministic_algorithms_enabled()` is true. Otherwise two runs with the same seed differ, the pairing is drowned in noise, and the statistical gate becomes harder to pass.
- **Switch file**: the candidate's switches (environment variables) go in the switch file, in the format `{"environment": {"X": "1"}}`. Variables the bench manages itself may not be set, for example `TMPDIR`, compile cache directories, `CUDA_VISIBLE_DEVICES`, `PYTHONPATH`.
- **Read-only paths and network**: `bench/`, `numerics/`, `targets/`, `task/` and `../assets/` are read-only; `wan_models`, `checkpoints` and `prompts/vidprom_filtered_extended.txt` inside `sf/` are links to them. There is no network; nothing can be downloaded and no new packages can be installed.

## Environment

- 2 H200-class GPUs. nvidia-smi reports compute capability 8.9, but torch sees (9, 0) with 132 SMs and about 140 GB per GPU, so they are actually Hopper. The GPU lock is `/tmp/sf_gpu.lock`:
  - `bench/bench.py` takes the lock itself; do not wrap it in flock;
  - wrap any other command that uses the GPUs as `flock /tmp/sf_gpu.lock <command>`.
- Python: `$WORK/miniconda3/envs/self_forcing/bin/python`. The environment has torch 2.7.1+cu128 (with its own cuDNN, Triton, `torch.compile`, flex_attention and the SDPA backends) and flash-attn 2.8.3. No FA3.
- Nsight Systems: `source $WORK/opt/nsight/env.sh` first. No Nsight Compute.
- Space in `/tmp` is limited; delete what you no longer need.
- Start long jobs with `setsid nohup ... &` and wait for them inside the session.

## References (skills)

These skills are in `~/.codex/skills/`; read the relevant ones before choosing what to try:

- `sglang-diffusion-benchmark-profile`, `sglang-diffusion-performance`: benchmark/profile methods for sglang diffusion-model inference and their list of existing fast paths (originals, aimed at inference serving; the methods carry over);
- `train-opt-tiered-profiling`, `train-opt-measurement`, `train-opt-action-ladder`, `train-opt-numerics-validation`, `train-opt-fusion-patterns`: priors summarized from other training-optimization tasks, covering tiered profiling, timing pitfalls, the action ladder, numerics verdicts and fusion patterns. Re-verify these priors on this task before treating them as conclusions.
