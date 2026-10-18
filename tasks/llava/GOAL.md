# Goal

*Translated from the Chinese original the agent received.*

Bring the **time per optimizer step** of LLaVA-1.5 7B stage-2 full-parameter instruction tuning on 4 H20s **down to 1/2.5 of the reference (a 2.5× speedup)**, never below the 2.0297× already reached, while passing the numerics checks below.

> **Task statement revision 3 (the formulator changing the goal, not the judge)**: the main target rises from 2.0× to **2.5×**, and the wrap-up floor from 1.5× to **2.0297×** (= the measured value of the full check in `runs/r58/check/check.json`). Not a byte changed on the judge side: `evaluator_sha256` is still `3ad5fb1d11fa6f27790bae53d74a23ab46bdb89d34dac7311cf3202c6ec4cb63` (it covers only `numerics/**/*.py` and `bench/**`, not this file), and the thresholds of all four gates, δ and the 13 frozen inputs are all unchanged. **The "Constraints" section is also unchanged**: the precision rule (training may not be lowered to FP8 / INT8 / FP16, the fp32 master weights and fp32 Adam state may not be removed, local precision changes may not exceed tier N2) still applies.

- Training code: `llava/`, upstream haotian-liu/LLaVA at commit c121f04 (local commit c37aefa), branch `opt`; this is where you make changes.
  - The bench launches `torchrun --nproc-per-node 4 llava/train/train_mem.py <frozen arguments>` from a copy of `llava/`, with PYTHONPATH set to that copy's root.
  - Scripts of your own that import llava must also put `llava/` first on PYTHONPATH.
- The workload is a frozen recipe, see `train_args` in `bench/bench.json`. It is upstream `scripts/v1_5/finetune.sh` with only these changes:
  - the model is 7B: `vicuna-7b-v1.5` + `clip-vit-large-patch14-336` (frozen) + the pretrained projector `llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5` (mlp2x_gelu);
  - 4 GPUs: upstream is 8 GPUs × 16 per GPU × accumulation 1 = global 128; here it is **4 GPUs × 16 per GPU × gradient accumulation 2 = global 128**.
    - The load of each micro-batch on each GPU is the same as upstream: 16 samples per GPU, `model_max_length` 2048, bf16 + tf32, DeepSpeed ZeRO-3 (`scripts/zero3.json`), flash-attention 2 (`train_mem.py`), gradient checkpointing.
    - LLaVA's sampler builds megabatches of "GPUs × accumulation steps" (8 × 16 = 128), so each micro-batch also holds the same samples as upstream's 8 GPUs: what rank r gets at micro-step m is exactly the share of upstream GPU 4m+r.
      - That is the **reference's** assignment. Which 128 samples form each optimizer step cannot change, but how these 128 are spread over the 8 micro-batches is up to the candidate; see "Constraints → Data and order" below.
  - `--report_to none` (no network, no wandb); paths changed to local ones.
- Data: timing uses a fixed fixture (`$ASSETS/data/fixture/timing.json`, images in `.../data/images`).
  - It was drawn at random with a fixed seed from `llava_v1_5_mix665k.json`: image-text samples uniformly from all 625k (so the shares of coco/gqa/ocr_vqa/textvqa/vg match the 665K), text-only samples (ShareGPT) drawn separately.
  - With the sampler's grouping, the 18 steps used for timing contain 17 image-text steps and 1 text-only step (text-only is 6.1% of the 665K). The sampling method and hashes are in `task/formulation.json`.
- Reference: `targets/reference`, the same commit, code unchanged.
  - The reference takes **20968 ms per step** in the timing windows (`task/timing/reference.json`).
  - Time per part, memory and the top profile hotspots of the baseline are in `task/baseline.md`.

## What counts as reaching the goal (the single criterion)

```
python bench/bench.py check --tree llava --config <switch file> --out <output dir>
```

It writes `<output dir>/check.json`.

- **Done**: `status` is `PASS` and `performance.speedup` ≥ 2.5.
- If profile evidence shows that 2.5 cannot be reached within the rules, `status` `PASS` with `performance.speedup` ≥ 2.0297 is also acceptable for wrapping up. In that case, write down in `WORK_LOG.md` why each remaining hotspot cannot be addressed, giving a lower bound and its basis for ops, communication and recomputation separately; a single "it can't be done" is not enough.
- **2.0297× is the floor; no regression**: at wrap-up there must be a full `check` with `speedup` ≥ 2.0297 as evidence. The existing one is `runs/r58/check/check.json` (commit `add961a`, switches `runs/r58/packed17.json`), which already satisfies this, so this phase is a pure search upwards and a failure cannot leave you below it.

`check` applies the gates below in order and stops at the first that fails:

0. **Structure** (CPU, about 2 minutes):
   - `train()` and the three classes `ModelArguments`, `DataArguments` and `TrainingArguments` in `llava/train/train.py` must have exactly the same syntax tree as the reference (the judge's adapter and data probe reproduce this setup the reference's way). Different → ERROR. To change the setup, put the change into the modules it calls (model, trainer, data, the entry point `train_mem.py`).
   - For the first 16 micro-batches (= the first 2 optimizer steps) that the data pipeline (your dataset, collator, sampler, `get_train_dataloader`) produces from the timing fixture, **the global sample multiset of each optimizer step** must be bitwise identical to the reference (compared after sorting per-row digests: input_ids/labels without padding, images as bf16), and the global token count, label count and number of samples per micro-batch must be identical too (`bench/loader_probe.py`). Different → INCORRECT.
     - Not compared: which samples share a micro-batch, which GPU, which micro-step, and the row order within a micro-batch. Whether the candidate reassigned samples is only logged as a diagnostic.
1. **Benefit**: reference → candidate → reference, three independent torchrun runs of the real entry point with cold compile caches.
   - Each runs 18 steps: 2 warmup steps, then 2 timed windows of 8 steps. `performance.speedup` = the mean time per step of the two reference runs ÷ the candidate's.
   - The fingerprints of the three runs (math mode, GPUs, library versions, timing plan and so on) must match the first reference run, otherwise ERROR.
   - The following mean interference on the machine; the run is INVALID and rerun a little later:
     - the ratios of a run's two windows to the frozen reference differ by more than 6%;
     - the reference deviates from the frozen reference by more than 15%.
   - Per-step loss sanity check (this is not the precision gate; precision is judged by gates 2 and 3): the first logged loss happens before any optimizer step,
     so its relative difference from the frozen reference must be ≤ 0.5% (this covers "the same data, and the same forward"); later steps only have loose
     constraints: no 0 / NaN / inf (where the reference is nonzero), and a relative difference of at most 25% (divergence or a wrong computation goes far beyond this, while legitimate
     N1/N2 changes only perturb slightly within this band). Out of bounds is ERROR.
     - **Task statement revision 1**: if the candidate reassigns the samples within an optimizer step (as determined by the data probe), both bands above widen to
       48.4%. Reason: the logged loss is "the token-mean cross-entropy of each micro-batch, then the mean across micro-batches", so it moves when the assignment
       changes, even when the same 128 samples are trained. The number was measured: **the reference's own code** was run with `bench.py time` under two legitimate assignments,
       which shifted it by 0.1% and 24.2%; the band is twice the largest observed shift, and the measurement is frozen in
       `task/partition_sensitivity.json` (how the two assignments were built is on the formulation side, not here). **For candidates that do not reassign, both bands are unchanged** (0.5% / 25%).
     - The formulator tried to remove the effect of the assignment exactly with a "per-sample (L_i, c_i) table", to keep the first-step band at 0.5%, but **measurement
       disproved it**: the prediction for a measured assignment was 1.7938 against a measured 1.7149, a 4.6% gap, far from the 0.5% precision needed.
       So the band is the one above: 0.5% without reassignment, 48.4% with reassignment. The failed attempt is recorded in
       `task/judge_changes.md` and not kept in the verdict.
     - So for reassigned candidates, "has the computation been broken" is **not carried by gate 1's loss band**, but by gate 2 (per-tensor
       comparison of the real loss / fp32 gradients / master weights / Adam state, plus the `dead_gradient` exact item) and gate 3,
       which are much stronger than a loss proxy.
     - Likewise, "is it the same data" no longer relies on this loss proxy but on an **exact check**: `time` (and `sweep`, `check`) first run a CPU data probe
       before taking the GPUs and compare each optimizer step's sample multiset with the frozen reference probe (`task/loader_reference.json`); a mismatch is an immediate ERROR,
       without a single measured step. This is stricter than the old 0.5% proxy.
   - A speedup that does not exceed max(1.02, 1 + 2 × relative gap between the two reference runs) is UNAVAILABLE (no_benefit), rejected outright, without a precision evaluation.
2. **Bitwise**: in deterministic mode (torch deterministic algorithms, flash-attn deterministic backward), record two short cases: 1 optimizer step on an image-text batch and 1 optimizer step on a text-only batch (each 4 GPUs × 2 micro-steps × 16 per GPU, the same shapes as real training).
   - The judge's adapter (`bench/llava_adapter.py`) runs your entry point and your trainer: it catches the trainer when `train()` calls `trainer.train()`, then lets its own training loop advance step by step.
   - **Every exact item must hold, otherwise INCORRECT**, including:
     - the RNG fingerprint of each step, the number of optimizer steps, global_step, the number of micro-steps;
     - the **global sample multiset** each optimizer step hands to `training_step` (after an all-gather over the four GPUs, per-row digests sorted: input_ids/labels without padding, images as bf16), plus the global token count, label count and samples per micro-batch; per-micro-batch digests and the token share of each GPU are **not compared** (the candidate may choose its own assignment, see "Constraints");
     - learning rate, Adam hyperparameters, DeepSpeed's batch/accumulation/clipping/precision settings, the set of trainable parameters;
     - the dtypes of the fp32 master weights and Adam state;
     - the cross-rank pattern.
   - One more exact item: every trainable parameter whose gradient is nonzero in the reference must also have a nonzero (finite and not 0) gradient in the candidate. A whole module silently no longer training (for example the 33.5M-parameter
     projector in a 6.8B-parameter model) affects tens of steps of loss too little for the statistical gate to see, so it is judged here as an exact item.
   - All recorded tensors must be bitwise identical for **PASS (N0)**, accepted directly. These tensors are: each GPU's loss, the full fp32 gradients of all trainable parameters, the fp32 master-weight updates, and both Adam states. They are independent of the ZeRO stage; each GPU stores an equal slice.
   - Per-tensor tolerances and maximum deviations are only written into the report as diagnostics and do not enter the verdict.
3. **Statistical**: only when not bitwise identical.
   - **Task statement revision 2 (authorized by the owner, made after r36 was rejected by this gate)**: both endpoints are now **computed by the judge itself**: the judge drives the `model` forward with a batch containing only the frozen
     input keys (not `tr.compute_loss`, and not the return value of `training_step`) and takes the model layer's
     mean cross-entropy. Reason: once the task statement relaxed the loss normalization, numbers taken from each tree were not comparable between trees; in r36 the
     probe endpoint was `within` (−0.49σ) while the train endpoint `deviates` (+4.29σ): the difference was the ruler, not the training.
     - `train_loss_last20` → `train_loss_last20_canonical`: the mean of the model-layer loss of the step's 8 micro-batches,
       measured with the weights from **before** this step's training;
     - `probe_loss_last20`: the held-out batch goes through the judge's forward too (it used to go through `tr.compute_loss`, which depends on the normalization).
     - For the reference this canonical value **equals the old endpoint by construction** (the reference's `training_step` returns `c_mb/accum`, and the old endpoint's sum over the 2
       micro-steps is exactly the mean), so δ should barely change after refreezing; the refrozen numbers and the control experiment "a candidate that already
       passed before the change still passes under the new δ" are both recorded in `task/judge_changes.md`.
   - In deterministic mode, run paired short training trajectories with the frozen seeds: 4 samples per GPU × accumulation 2, 60 steps, cosine with warmup, the seed determining the data order. Only losses are recorded, plus a probe loss from one forward on 32 held-out samples at each of the last 20 steps.
   - Endpoints: the mean training loss over steps 40–60 and the mean probe loss over steps 40–60, both averaged across GPUs.
   - δ = 2 × the reference's own seed-to-seed standard deviation; the reference's 6 seeds are frozen in `bench/traj/frozen.json`.
   - Paired difference d = candidate − reference; look at the mean m and a paired t-test:
     - |m| > δ and p < 0.05 → deviates;
     - |m| ≤ δ → within;
     - otherwise → unclear.
   - Start with 3 seeds: any endpoint deviating → INCORRECT; all within → PASS; otherwise extend to 6 seeds. At 6 seeds any endpoint with |m| > δ → INCORRECT.
4. **Tier**:
   - N0: everything bitwise identical, with the same dtypes and math modes;
   - N2: the dtype or math mode of any tensor changed (TF32, low-precision reductions, SDPA backend);
   - N1 otherwise;
   - beyond N2 → INCORRECT.

Two other modes:
- `--quick`: gate 0 and the structural recording of gate 2 (exact items) only, about 7 minutes.
- `--skip-timing`: skips gate 1.

Time for one full check:
- gate 0 about 2 minutes;
- timing about 3 × 6 minutes;
- bitwise about 6 minutes;
- statistical gate about 25 minutes for 3 seeds, about 49 for 6.

About 40–50 minutes in total (statistical gate with 3 seeds). So screen with `time` and `check --quick` day to day, and run the full check once confident.

## What every round must do

1. **Profile first, then change.** Before each round's change, profile the current best version and choose the hotspot for this round from the profile:
   ```
   python bench/bench.py profile --tree llava --config <switch file> --out runs/<round>/profile [--steps K] [--detail]
   ```
   - By default Nsight Systems captures 2 steps after warmup (all 4 ranks in one `nsys_*.nsys-rep`) and `nsys stats` writes a kernel / memory-copy summary `profile_<label>.md`. For finer detail, break it down yourself with `nsys stats` / `nsys export`.
   - `--detail` uses the PyTorch profiler instead and captures 1 step (op shapes, dtypes, call stacks, nn.Module, memory events, op-to-kernel mapping), one trace per rank; `profile_<label>.md` has the top 40 kernels and ops of rank 0.
   - In the first round, profile once before any change (at that point `llava` is the reference).
2. **Time only with bench.py**:
   - time on its own with `python bench/bench.py time --tree llava --config <switch file> --out <dir>`, which also reports the speedup against the frozen reference (`speedup_vs_frozen_reference`);
   - screen several switches together with `sweep`;
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
- Approximate times of the bench commands:
  - `profile` about 2.5 minutes;
  - `time` about 6 minutes;
  - `check --quick` about 7 minutes.

  About 1 minute of each is startup: loading the 7B weights into ZeRO-3, CLIP and the projector, and building the DeepSpeed engine.
- The full `check` does not count towards a round. Run it only when the accumulated changes, as measured by `time`, are already enough for 2.0297× or 2.5×. While it holds the GPUs you may do preparation that needs no GPU, but do not present untested changes as results.
- Changes that do not depend on each other can each get a switch (an environment variable) and be screened in the same round with `sweep` over different switch files, then combined according to the results.

## Constraints

- **Training semantics must not change** (gate 0 compares setup and data, gates 2 and 3 compare numerics); specifically:
  - Data and order: dataset, preprocessing (conversation template, tokenization, images padded to square and then through the CLIP processor), sampler grouping by modality, megabatch membership (which 128 samples form each optimizer step), batch 16 per GPU, gradient accumulation 2, 4 GPUs; samples may not move across optimizer steps, be dropped or be duplicated.
    - **Relaxed by task statement revision 1** (the formulator changing the task statement, not loosening the judge): the assignment within an optimizer step is your choice:
      - how the 128 samples are spread over the 8 micro-batches (4 GPUs × 2 micro-steps): which samples share a micro-batch, which GPU, which micro-step, and the row order within a micro-batch;
      - the length key used for grouping: upstream `modality_lengths` uses a word count (split on spaces), which may be replaced by the real token length after tokenization; the sign convention stays (positive for image-text, negative for text-only), and modality grouping stays as before;
      - the bucketing target of `split_to_even_chunks` may change too (upstream balances the sums of word counts).
    - Such changes by definition cannot be bitwise identical (gate 2's tensors change, so N0 is out of reach); they are judged N1/N2 and **decided by gate 3, the statistical gate**: δ is already twice the reference's own seed-to-seed standard deviation (and what the seed changes is exactly the data order), the right ruler for this.
  - Loss: **relaxed by task statement revision 1**; the cross-entropy normalization may be either of:
    1. **upstream**: each micro-batch averages over its own valid label tokens, then the 8 micro-batches are averaged with equal weight;
    2. **global token mean**: over the whole optimizer step, Σ(cross-entropy of valid label tokens) ÷ Σ(number of valid label tokens).
    - Both count only valid label tokens: padding labels are already `IGNORE_INDEX` (-100), skipped by the cross-entropy, so padding
      never entered the loss. The two differ only in **the weights across micro-batches**: option 1 gives each micro-batch an equal weight of 1/8, so
      when the 8 micro-batches have different numbers of valid tokens, each token in the shorter ones gets a higher weight; option 2 weights every token
      equally. Under the reference assignment the 8 micro-batches were measured to have between 953 and 1865 supervised labels (a 1.96× spread), yet the first-step loss of the two options
      differs by only **0.52%** (1.3402 vs 1.3332, measured offline by the formulator), so under the reference assignment the two are close.
    - Option 2 **is invariant to the assignment by nature**, so reordering micro-batches no longer changes the objective; the price is a
      small difference from the reference (option 1), decided by gate 3 like any other N1/N2 change.
    - Still not allowed: weighting by sample count (instead of token count), skipping or repeating micro-batches, counting a valid token twice or missing one,
      switching to soft labels, changing the semantics of `IGNORE_INDEX`.
    - Implementation note: with gradient accumulation, option 2 needs the total number of valid tokens of the whole step before the first backward (an all-reduce across GPUs,
      or the data pipeline providing it ahead of time). That is an implementation burden, not a loosening of the judge.
    - **Normalization may only be done at the Trainer level** (`compute_loss` / `training_step`): `model.forward(labels=...)` must
      still return "the mean cross-entropy of this one batch over its own valid label tokens". Reason: gate 3's two endpoints are now measured by the judge
      driving the model forward itself (see above), so the model layer must give the same quantity in both trees; if you put your normalization in the Trainer, training still uses
      your normalization, but how it is measured is not up to you.
  - Learning rate 2e-5, cosine, warmup 0.03, AdamW (β, ε, weight decay 0), no gradient clipping, one optimizer step per step;
  - Model: CLIP frozen, the LLM and the projector trained with all parameters; trainable parameter names and shapes must match the reference (the judge looks parameters up by name, so do not merge q/k/v and the like into new parameters).
- **Precision**:
  - The baseline is the upstream recipe's precision: bf16 compute + tf32, fp32 master weights, fp32 Adam state;
  - training may not be lowered to FP8 / INT8 / FP16;
  - the fp32 master weights or fp32 optimizer state may not be removed (gate 2 compares their dtypes as exact items);
  - local precision changes are allowed, but their tier may not exceed N2, and they must pass gate 3.
- **Keep the bench's hook points**:
  - the entry point is `llava/train/train_mem.py`, which calls `train()`, which calls `trainer.train()`;
  - training goes through the HF Trainer loop: one `trainer.training_step(model, inputs)` call per micro-batch, with inputs in the reference's format (`input_ids`, `labels`, `attention_mask`, `images`); the Trainer's `on_step_end` callback fires at the end of every optimizer step;
  - the optimizer step goes through the DeepSpeed engine: `engine.optimizer.step()` is called once per step, and `deepspeed.utils.safe_get_full_grad / safe_get_full_fp32_param / safe_get_full_optimizer_state` can fetch the full gradients, master weights and Adam state before and after it. The ZeRO stage and other DeepSpeed settings (`llava/scripts/zero3.json`; the bench uses exactly this file from your tree) may change, but batch, accumulation, clipping and precision are exact items.
- The code may not read the bench's internal state, and may not behave differently depending on whether it is being timed or judged. Timing, bitwise and statistical runs must all exercise the same code path.
- **Switch file** format is `{"environment": {"X": "1"}}`:
  - only `environment`; the bench sets these variables for the candidate's processes only;
  - variables the bench manages itself may not be set, for example `TMPDIR`, the compile cache directories, `CUDA_VISIBLE_DEVICES`, `PYTHONPATH`, `CUBLAS_WORKSPACE_CONFIG`, `HF_*`.
- **Read-only paths and network**:
  - `bench/`, `numerics/`, `task/`, `targets/` and `$ASSETS/` are read-only;
  - there is no network; nothing can be downloaded and no new packages can be installed.

## Environment

- 4 H20s (Hopper, sm_90, about 96 GB each, NVLink), visible as GPUs 0–3 in the sandbox.
  - The GPU lock is `/tmp/llava_gpu.lock`: `bench/bench.py` takes the lock itself; do not wrap it in flock;
  - wrap any other command that uses the GPUs as `flock /tmp/llava_gpu.lock <command>`.
- Python: `$ASSETS/env/bin/python` (first on PATH). Main versions follow upstream `pyproject.toml`:
  - torch 2.1.2+cu121, torchvision 0.16.2, triton 2.1.0;
  - transformers 4.37.2, tokenizers 0.15.1, accelerate 0.21.0, deepspeed 0.12.6;
  - flash-attn 2.5.8 (built locally, sm_90), numpy 1.26.4.
- These performance libraries, unused by the upstream recipe, are preinstalled for you to choose from (the reference recipe uses none of them):
  - `liger-kernel` 0.4.2 (Triton RMSNorm / SwiGLU / RoPE / cross-entropy and more, verified to work with this environment's triton 2.1.0);
  - NVIDIA `apex` (built locally, sm_90, with CUDA extensions: `amp_C` (multi-tensor Adam/LAMB/L2), `fused_layer_norm_cuda`
    (FusedLayerNorm / FusedRMSNorm), `fused_dense_cuda`, `mlp_cuda`, `scaled_*_softmax_cuda`,
    `fused_rotary_positional_embedding`, `syncbn`);
  - DeepSpeed's own fused ops (FusedAdam, FusedLamb, transformer inference kernels and so on), JIT-compiled on first use;
  - `nsys` (see below).
- DeepSpeed's fused ops (FusedAdam and others) are JIT-compiled into `~/.cache/torch_extensions` on first use (nvcc 12.6 + ninja), which takes a few minutes and is reused afterwards.
- Nsight Systems: `nsys` is on PATH (2025.6.3). Nsight Compute is installed in /usr/local/cuda, but this machine's permissions do not allow sampling.
- `TMPDIR=/tmp` (local disk, private, enough space). The bench uses a new directory under `/tmp` for every run and deletes it afterwards.
- Start long jobs with `setsid nohup ... &` and wait for them inside the session.
- The frozen reference artifacts are under `task/` and readable:
  - `task/timing/` holds the reference's timing runs;
  - `task/baseline.md` has the baseline numbers and profile hotspots;
  - `task/formulation.json` has the hashes of the frozen inputs and the fixture's sampling method;
  - `bench/records/ref` holds the reference's bitwise records, and `bench/traj/` the reference trajectories and the frozen δ.

## References (skills)

These skills are in `~/.codex/skills/`; read the relevant ones before choosing what to try:

- `train-opt-tiered-profiling`, `train-opt-measurement`, `train-opt-action-ladder`, `train-opt-numerics-validation`, `train-opt-fusion-patterns`, `train-opt-fsdp-multi-forward`;
- `sglang-diffusion-benchmark-profile`, `sglang-diffusion-performance`.
- They are priors summarized from other training-optimization tasks, covering:
  - tiered profiling, timing pitfalls;
  - the action ladder, numerics verdicts;
  - fusion patterns, multiple forwards under FSDP;
  - benchmarking and optimization routines on the inference side.
- Re-verify these priors on this task before treating them as conclusions.

## Wrap-up

- Commit each change separately in `llava/` (branch `opt`), with a message stating what changed, the tier and the `time` result.
- Once the goal is reached, write a summary at the end of `WORK_LOG.md`, including:
  - the final switch file, and the path and result of the final check;
  - the breakdown of gains by change;
  - directions that did not work and why.
