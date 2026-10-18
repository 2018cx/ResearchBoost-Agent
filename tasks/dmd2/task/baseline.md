# Baseline (reference = upstream tianweiy/DMD2@8d8fa55, code unchanged)

The workload is in `bench/recipe.py`: the upstream SDv1.5 recipe (the published FID 8.35 row, continued from the step-39000 checkpoint),
one machine with 4 GPUs, torchrun + DDP, batch 32 per GPU, 512×512, bf16 autocast, gradient checkpointing, one generator
update every 10 iterations.

## Timing (`bench.py baseline`, three independent runs, `runs/baseline/baseline.json`)

| Quantity | Value |
|---|---|
| Per iteration | **3.73 s** (37.3 s per 10-iteration cycle) |
| Windows (20 iterations) | 74.80 / 74.66 s (frozen reference, `task/timing_ref.json`) |
| Total time of the three runs | 149.29 / 149.39 / 149.49 s |
| Range across the three runs | **0.14%** |
| Peak memory per GPU (allocated) | 69.57 GiB (95 GiB GPUs) |
| Startup to the first iteration | 26–28 s |
| Per-GPU, per-iteration RNG fingerprints | identical across the three runs |

Timing plan: 10 warmup iterations (including the first DDP bucket construction and allocator warmup), then 2 windows of 20 iterations, each window
2 full cycles. The 0.14% range is far below 1%, so the gate-1 threshold is set mainly by the 1.02 in `max(1.02, 1 + 2 × reference relative gap)`.

## Prerequisites for the numerics checks

- In deterministic mode the reference **is bitwise identical across two recordings**: `numerics compare --ref bench/records/ref --cand bench/records/ref2`
  gives `status PASS, measured_tier N0, checked 41661, bitwise_identical 41661`.
- This needs **strict** determinism (`torch.use_deterministic_algorithms(True, warn_only=False)`, which the adapter enables itself
  in deterministic mode): with numerics' default `warn_only=True`, flash attention's backward uses atomic accumulation and
  only 9565/41661 tensors of two reference recordings are bitwise identical (tier N1), which would make the bitwise gate meaningless.
- The per-tensor tolerance gates (`bench/thresholds.json`, 41661 gates) are calibrated with two legitimate variants: `sdpa_mem_efficient`
  (a different SDPA backend) and `tf32_off` (TF32 off). Both PASS under their own gates (tier N2).
  `numerics loo` (leave-one-out) does not pass, because there are only two variants and their envelopes differ by an order of magnitude; per the brief,
  **the per-tensor gates are diagnostic only and do not enter the verdict**; the verdict uses only the exact items and the bitwise/statistical gates.
- The `sdpa_math` variant was dropped: the math backend materializes the attention matrix explicitly and runs out of memory on all 4 GPUs.

## Profile (`bench.py profile`, 10 iterations = one full cycle, rank 0)

See `task/baseline_profile.md` (a copy of `runs/baseline_profile/profile.md`). Key points:

- One cycle (10 iterations) takes 37.6 s: the generator iteration 6.16 s, each of the 9 guidance-only iterations 3.49 s.
- Rank 0 GPU busy share 99% (37.1/37.6 s); cumulative kernel time 61.9 s (overlapping streams counted separately).
- Kernel time by category (ms, rank 0): other 24019, gemm 17411, attention 10960, memcpy/cast 4002,
  norm 980, nccl 951, elementwise/reduce 906, conv 851, optimizer 427.
  (The largest item in "other" is the `DistributedDataParallel.forward` record_function range, 17.8 s,
  which counts every kernel it wraps a second time; mind the nesting of ranges when reading absolute values.)
- Forward GPU time per model call (ms, rank 0; ranges are nested: guidance_model contains real_unet / fake_unet / gan_head):
  generator 9301 (10 calls), guidance_model 8384 (11), fake_unet 4903 (11), real_unet 653 (1),
  text_encoder 218 (20), gan_head 23 (21).
- Top kernels: fp32 mem-efficient attention `fmha_cutlassF_f32_aligned_64x64_rf_sm80` 4281 ms / 100 calls,
  flash backward 3239 ms / 194, flash forward 2192 ms / 314, cuDNN fprop (tf32) 1699 ms / 431,
  bf16 cuBLAS gemm 1694 ms / 4366, bf16 cuDNN fprop 1623 ms / 592.
- The ops with the highest CPU self time are in the last table of `task/baseline_profile.md`.

## Known facts about the environment (measured during formulation, relevant to optimization)

- The torch 2.0.1 that upstream pins is **unusable** on this machine (H20, sm_90): the forward of `F.scaled_dot_product_attention` works,
  but its backward fails with `CUDA error: an illegal instruction was encountered`.
- torch 2.1.2+cu121 raises SIGFPE inside cuBLAS: a bf16 linear layer without bias, `(4,77,2048) × (1280,2048)`, kills the process.
- FSDP in torch ≥ 2.1 rejects the upstream SDXL recipe toggling `requires_grad` on the guidance model back and forth within an iteration
  (`Invalid: _post_backward_hook_state`, upstream issue #41); the SDv1.5 recipe uses DDP and does not have this problem.
- The environment is therefore torch 2.4.1+cu124, with every other version as upstream.
