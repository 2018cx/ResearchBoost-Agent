# Speeding up Self-Forcing DMD training: notes from the interactive workspace

*Translated from the Chinese original.*

A workspace advanced interactively by the owner and an agent, with the same rules and criterion as [`GOAL.md`](GOAL.md); result 2.08× (full check PASS / N1 / statistically equivalent). These are the round-by-round notes; the switches are described in `docs/training.md` of the Faster-SelfForcing repository.

## Layout
- `sf/`: the candidate tree, a clean clone of upstream 33593df, branch `cc-opt`; weights, `ode_init` and prompts are symlinked to `$ASSETS`.
- `bench/`, `numerics/`: the task's frozen judge. Only a few places in `bench.py` changed, none affecting the verdict: the lock path can be set with `SF_GPU_LOCK`; `gpu_state` only looks at the GPUs in `CUDA_VISIBLE_DEVICES` (outside the sandbox nvidia-smi sees the whole machine's GPUs); optional graph-node-level tracing in profile mode.
- `bench/records/ref`, `bench/traj`, `task/`, `targets/self_forcing/reference`: one copy of the reference data, shared with the task's formal workspace.
- `b.sh`: bench wrapper that selects GPUs and the matching lock. `runs/`: per-round outputs.
- Environment: conda `self_forcing` (torch 2.7.1+cu128, FA2 2.8.3); FA3 built locally and placed into the candidate tree's `third_party/fa3`.

## Work log
(each round: profile findings / change / time / check)

### R00 baseline profile (GPU2/3)
- Tree: runs/r00/nvtx_tree (sf + plain NVTX ranges, patch runs/r00/nvtx_patch.py), outputs runs/r00/profile_nvtx/, phase attribution phase_breakdown.md (runs/r00/nvtx_attr.py: each kernel attributed to the innermost custom range at launch).
- A window of 10 iterations (2 gen) takes 106.1 s = 10.6 s/iter (under nsys; bench baseline 10.4 s); GPU busy 85%, idle 15%. Peak allocated 100.4 GB / reserved 135.3 GB.
- By kernel category (rank0 GPU 91 s): flash_fwd 39.3%, flash_bwd 18.3% (attention 58% in total); elementwise/copy 15.9% (of which RoPE's fp64/complex about 2.5 s, bf16→f32 casts 1.1 s); GEMM 14.9%; NCCL 5.5%; memcpy 3.6%; reduce/norm 2.1%.
- By phase (window GPU s): critic backward 26.5 (including 4.6 of attention forward recomputed by checkpointing); rollout 30.1 (denoise 14.3 / exit 7.9 / ctx 7.9); teacher cond+uncond 18.0; critic fake forward 7.4; gen backward 4.7; text encoding 2.2 (of which NCCL all-gather 1.85, exposed); DMD fake forward 1.7.
- CPU-dominated: EMA takes 2.8–3.5 s of CPU wall clock per gen iteration (the GPU only 0.35–1.0 s of D2H); rollout CPU wall clock exceeds GPU time by about 9 s per window (launch-bound + one .item sync per layer).
- FA3 microbenchmark (hdim128 bf16, locally built third_party/fa3): forward 1.65–2.05×, forward+backward 1.77–2.0×, errors the same as FA2, deterministic backward reproducible bit for bit.

### R01 FA3 (commit 0a800e9)
- Change: attention.py takes the vendored FA3 dense path when there is no padding; deterministic=True in deterministic mode; SF_ATTN=fa2 switches back.
- Result: time 8399 ms/iter (windows 8271/8526, spread 3.0%), about 1.24× against the reference's ~10400; check --quick PASS (structure gate), not bitwise, N1; peak 93.55/128.0 GiB unchanged.

### R02 four items in parallel (each in its own worktree + environment switch, off by default)
- wt/ema branch r02-ema (SF_EMA_FAST); wt/sac r02-sac (SF_SAC); wt/rope r02-rope (SF_ROPE_FUSED); wt/sync r02-sync (SF_NO_SYNC).
- Validation: first run check --quick with SF_ATTN=fa2 + the switch to see whether it is bitwise identical to the reference; then time with FA3 + the switch against R01's 8399 ms.
- The owner added: "compile selectively; try CUDA graphs too; mind the precision". Two more lines: wt/compile branch r02-compile (SF_COMPILE), wt/graph branch r02-graph (SF_CUDAGRAPH).

### R02 summary (each item alone: check --quick bitwise against the reference under FA2 / time under FA3)
- ema f49989c: N0; EMA 2.2 s → 0.2 s per call; time 7458 (spread 8.5%, noisy).
- sync 1d85631: N0; rollout syncs 920 → 65 per rollout; time 7770.5 (spread 0.07%).
- rope 41e4276: N0 (Triton, reproducing PyTorch's FMA / double rounding / -0 behavior); a single rope 7–13× faster; time ~7656 (spread 5.5%).
- sac f31eb4a: N0; 0 attention forward recomputations in backward; time 7850 (spread 6.8%); reserved +4.2 GiB.
- compile 883679d/36358b7: N0 (emulate_precision_casts + Triton FP fusion off at compile time); 34–40 µs of CPU per compiled call, no gain on the rollout; suggested SF_COMPILE_MIN_TOKENS=16384.
- graph f7ad001: N0 (7 captures per rank); 2–3% gain but +5 GiB of memory → not merged this round. Finding: FSDP re-all-gathers 2.6 GB on every forward, plus kwargs copies; a single-GPU rollout without FSDP takes 2.17 s against 3.10 s with FSDP on two GPUs.
- Merged into cc-opt: ema → sync → rope → sac → compile (a361e3c), all merged automatically without conflicts.
- R03 profile (FA3+ema+sync+rope, nsys): window 66.0 s vs R00's 106.1 s (1.61×); FA3 46%, GEMM 24%, elementwise 18%, NCCL 5.5%, idle 9.6%; 0 FA2 kernels.
- Owner decision: FSDP may skip repeated gathers during no-grad rollouts (sharding/strategy/wrapping/use_orig_params unchanged) → wt/fsdp branch r03-fsdp; FP8 attention not for now.

### R03 ablation of the merged tree (GPU2/3, cc-opt a361e3c, FA3 on throughout, one time per configuration, run sequentially)
- runs/r03/ablate/: all 6207.7 (windows 5986/6429, spread 7%); none (FA3 only) 7715.0.
- Leave-one-out (ms/iter without the item, and the increment): no_ema 7220.4 (+1013); no_rope 7083.3 (+876); no_sync 6777.3 (+570); no_compile 6628.3 (+421); no_sac 6419.2 (+211).
- All five are positive, but the increments sum to 3091, far above none − all = 1507: they do not add up. The FSDP agent was running at the same time, single-run noise is about 7%, and R01 with FA3 only measured 8399 against 7715 now, a drift of about 8% across time slots. The ordering is only a rough guide.
- Memory: all 93.56/127.38; no_sac 92.74/128.69; no_rope 94.36/127.72.

### R03-fsdp (d5a1b3d, fast-forwarded into cc-opt)
- SF_FSDP_HOLD=1: during the rollout the generator keeps the all-gathered full parameters and the no-grad forwards reuse them; they are released before the exit forward with gradients, which takes the normal path. All-gathers per no-grad rollout 28/35 → 1, per rollout with gradients 35 → 8.
- SF_XATTN_CACHE=1: the cross-attention cache becomes a CacheSlot passed by reference through FSDP's argument copying, so K/V are computed only once per rollout.
- check --quick (FA2 + sync + hold + xattn) PASS, N0, 40490/40490 bitwise identical; standalone two-GPU tests of hold and hold+xattn give bitwise-identical outputs and gradient shards.
- Two rounds of interleaved A/B on the same tree: all on 6313/6515, all on + hold + xattn 6063/6191, mean 6414 → 6127 (−4.5%). Allocated peak unchanged. Report in runs/r03/fsdp/REPORT.md.

### R03 fixing SF_NO_SYNC's host_index (2887600)
- The original CPU mirror was a list inside the kv_cache dict. The FSDP root copies dict and list arguments, so updates only reached the copy and the pipeline's cache always held [0, 0]. Without eviction (local_attn_size=-1, which is the case in this task) the result happened to be correct; with eviction it would go wrong.
- Fix: an _HostIndex object, which FSDP passes through by reference.
- Test runs/r03/test_host_index.py (FSDP, local_attn_size of 6 frames, 7×3 frames): bitwise identical to the .item path, the mirror always keeps up with the tensor; the old list mirror fails at the first eviction.

### R04 final full check (GPU2/3)
- Tree cc-opt 2887600, configuration runs/r04/final.json = the five switches + SF_FSDP_HOLD + SF_XATTN_CACHE. Outputs runs/r04/check_final/, log runs/r04/check_final.log, runs/r04/DONE written when finished.

### R04 profile (final tree 2887600 + final.json, GPU0/1; the previous attempt was INVALID because another job occupied the same GPUs)
- Outputs runs/r04/profile_nvtx/ (phase_breakdown.md). Window 62.89 s = 6289 ms/iter (under nsys), GPU busy 83–86%, idle 14–17% (R03 was 9.6%; with a faster GPU side the CPU overhead shows).
- By category: FA3 device_kernel 45%, GEMM 25.7%, elementwise/copy 17.8% (9.6 s), NCCL 4.6–5.8%, reduce/norm 3.5%.
- By phase (GPU s per window): critic_bwd 14.1; rollout 17.1 in total (denoise 7.7 / exit 4.8 / ctx 4.6); teacher cond+uncond 12.8; critic_fake_fwd 4.7; gen_bwd 2.8.
- The rollout is CPU-bound, CPU wall clock vs GPU: denoise 9.96 vs 7.66, exit 8.06 vs 4.76, ctx 5.81 vs 4.63, about 6.8 s extra per window. gen_bwd CPU 4.23 vs GPU 2.76.
- Two lines dispatched: r04-graph (the graph agent continues: CUDA graph v2 on top of hold, then FSDP's remaining CPU overhead); r04-fuse (fork: bit-exact elementwise fusion). Both on GPU0/1, sharing one lock.

### R04 final full check result (cc-opt 2887600 + final.json): PASS, speedup 1.7203
- Timing: ref_a 11043.5 (windows 11148/10939); cand 6444.7 (windows 6628/6261, spread 5.7%); ref_b 11129.5 (windows 10706/11554, spread 7.6%). Reference mean 11086.5, slower than the morning's ~10425: drift across time slots.
  - Another job on the same GPUs overlapped the second half of ref_b; using ref_a alone gives 1.714×.
- Memory: cand 93.55/131.71, ref 93.54/127.6. Allocated unchanged, reserved 4 GiB higher (with the same configuration, reserved jumps between 127 and 131 anyway).
- Precision:
  - Bitwise gate: N1 (FA3); 12379 of 40490 tensors bitwise identical. 8193 exceed the per-tensor tolerances, which are diagnostic only; the largest are in the Adam state and in the bias/weight updates of cross_attn.k, whose k.bias gradient is zero in theory and numerically noise.
  - Statistical gate (3 paired seeds, 100 steps): all four endpoints within δ, judged equivalent. Mean difference / σ: critic_loss −0.15, critic_grad_norm 0.07, generator_loss −0.04, generator_grad_norm 0.11; p-values 0.43/0.89/0.91/0.75.

### R04-graph (0c03e3e, fast-forwarded into cc-opt)
- SF_CUDAGRAPH v2 built on SF_FSDP_HOLD: static parameters are refreshed only once per rollout (each hold entry counts as one epoch). The addresses of held parameters change across rollouts, because the teacher and others allocate in between, so a copy is required. 8 graphs in total, with cross K/V handled through CacheSlot.
- Precision: on one GPU with FA3, graph vs eager is bitwise identical (outputs 6/6, gradients 825/825); also bitwise identical with FSDP on two GPUs; check --quick (FA2 + final + graph) PASS, N0, 40490/40490.
- A/B (GPU0/1, two interleaved rounds): final 6276/6120, final+graph 5928/5841, mean 6197.9 → 5884.5 (−5.1%). Allocated 93.55 → 96.45, reserved 130.8–132.4.
- FSDP's remaining CPU overhead: about 14 ms per forward under hold, hidden by GPU time once graphs are on; a ceiling below 2%, not pursued.
- New configuration runs/r04/final_graph.json (final + SF_CUDAGRAPH).

### R05 profile (cc-opt 0c03e3e + final_graph.json, GPU2/3)
- Fourth change to bench.py: with SF_NSYS_GRAPH_NODE set, nsys gets --cuda-graph-trace=node so the kernels inside graphs can be attributed individually. Only affects profile mode.
- The first launch lacked this argument; it was stopped and rerun. Outputs in runs/r05/profile_nvtx/.
- R05 profile result: window 55.30 s = 5530 ms/iter (nsys, node-level graph tracing), GPU busy 92.5%, idle 7.5%.
  - By category: FA3 46.5%, GEMM 26.3%, elementwise 17.4% (8.97 s), reduce/norm 3.5%, NCCL 3.2%.
  - The rollout is no longer CPU-bound, CPU vs GPU: denoise 3.44 vs 5.73, ctx 2.93 vs 4.76, exit 5.88 vs 5.17.
  - Still CPU-bound: gen_bwd 4.15 vs 2.80; EMA 1.11 vs 0.28.
  - Estimate: the GEMMs of the teacher's 14B forward run at about 830 TFLOPS and FA3 at about 610 TFLOPS, already near peak. Without changing numerics, only three areas remain: elementwise, checkpoint recomputation and idle time, with a rough ceiling of about 13%, i.e. 2.0–2.1×. Reaching 3× would need FP8 attention or GEMMs, which the owner decided not to do for now.
- r05-mem dispatched (fork): measure the memory timeline first, then extend SAC's save policy within the headroom to reduce recomputation in critic_bwd and gen_bwd, and assess whether EMA can overlap with the next iteration.

### R04-fuse (b6f4fed + ec286bf; cc-opt fast-forwarded to 24edb8b = r04-graph + r04-fuse)
- SF_FUSE_EW=1:
  - one Triton kernel each for RMSNorm forward and backward, with the row reduction reproducing the addition order of ATen's reduce_kernel, hence bitwise identical; below 16 rows it falls back to eager.
  - modulate/gate at rollout size and below MIN_TOKENS each fused into one kernel.
  - compiled kernels are cached and called directly, cutting CPU overhead per call from about 23 µs to about 5 µs.
  - per-path counters printed at exit.
- Precision:
  - unit tests: forward and gradients of RMSNorm and modulate/gate all bitwise identical.
  - check --quick FA2 + final + FUSE: N0, 40490/40490.
  - combined with graphs: N0, 40490/40490, 8 graphs captured per rank.
- A/B (ABBA order):
  - without graphs: 6111.7 → 5788.0 (−5.3%).
  - with graphs: 5901.4 → 5498.1 (−6.8%, noise 0.4–3%).
  - single GPU: RMSNorm 4680×1536 from 109 µs to 14 µs; no-grad rollout 11.6% faster.
- Memory: with graphs, allocated 96.44 → 91.6 GiB, reserved about 130 → 126 GiB.
- The full check of final2.json (final_graph + SF_FUSE_EW) was started on GPU2/3, outputs runs/r05/check_final2/, runs/r05/DONE written when finished. r05-mem was told to rebase onto 24edb8b and not to run timings during the check's timing phase.

### R05 full check (cc-opt 24edb8b + final2.json): PASS, speedup 2.0178
- Timing: ref_a 11086.4 (windows 11332/10841); cand **5364.4** (windows 5279/5450); ref_b 10562.5 (windows 10804/10321). Reference mean 10824.5, spread 4.8%.
  - During ref_a, r05-mem was running a check --quick record on GPU0/1, which may have slowed ref_a. Using ref_b alone gives 1.969×, ref_a alone 2.067×.
- Memory: cand 91.61/125.14, ref 93.55/128.1. Allocated 1.9 GiB below the reference.
- Precision:
  - Bitwise gate N1, 12379/40490 bitwise identical, exactly as in R04, because every item except FA3 is bitwise identical.
  - Statistical gate with 3 seeds equivalent; the mean_in_sigma and p-values of all four endpoints are also exactly the same as in R04, so the trajectory is determined entirely by FA3 and the other switches do not change the numerics.

### R05-mem (79e3a75, fast-forwarded into cc-opt)
- Memory timeline: the peak of 91.61 GiB is in gen_bwd; critic_bwd reaches 78.97.
- SF_SAC_SAVE=ffn2,o,co: bitwise identical, 2.3% faster per critic step (under 1% overall), but reserved reaches 135 GiB, so it stays off.
- SF_EMA_ASYNC=1: EMA's CPU update moves to a background thread, joined before every read. Bitwise identical (N0), blocking time 1.32 → 0.78 s, A/B −1.4% (within noise).
- gen_bwd's CPU overhead comes from 7 short-sequence forward recomputations with gradients and from autograd, unrelated to SAC; reducing it would need graphs with gradients.

### R06 full check (cc-opt 79e3a75 + runs/r06/final3.json = final2 + SF_EMA_ASYNC, GPU2/3, no other agent running this time)

### R06 full check (79e3a75 + final3.json): PASS, 1.9608
- ref_a 10890.8, cand 5436.8 (windows 5392/5481), ref_b 10430.2; memory cand 91.61/125.58. Asynchronous EMA shows no gain in the formal timing (cand 5437 vs R05's 5364, within noise).

### R07 profile (79e3a75 + final3, GPU0/1; scripts runs/r07/hot.py, hot2.py, output runs/r07/hot.txt)
- The first run on GPU2/3 was contaminated: another user was running processes on GPU2/3 (a policy service at 87% utilization on GPU3, simulations starting repeatedly on GPU2), and that run showed NCCL at 46% and 8608 ms/iter. Its outputs were renamed *_gpu23_contaminated. From now on, check for other users on 2/3 before timing.
- Clean data: window 52.30 s = 5230 ms/iter (under nsys; R00 was 10610), GPU busy 93%.
- Kernel time by type: FA3 forward 16.7 s (33.4%), FA3 backward 8.6 s (17.1%), GEMM 14.4 s (28.9%), elementwise 3.6 s, Triton 1.9 s, reduce/norm 1.3 s, NCCL 2.7 s (5.4%), idle 3.7 s (7.1%).
- Idle time concentrates on the generator's paths with gradients: roll_exit 1.33 s, gen_bwd 1.15 s, running eager and CPU-bound.
- Remaining elementwise: GELU 0.95, bf16 mul (broadcast, not vectorized) 0.84, LayerNorm 0.81, bf16 add 0.94, int64 sum reduce 0.39 (in critic_bwd/gen_bwd, source unknown), bf16 copies 0.29.
- NCCL AllGather: text_enc 0.63 (T5 all-gathers on every call, exposed), teacher 1.15.

### R08 memory-bound work (the owner: "mainly see whether anything memory-bound can be optimized")
- Memory-bound total (R07): elementwise 3.6 + Triton 1.9 + norm 1.3 + memcpy 0.5 ≈ 7.3 s per window, 14.7% of kernel time.
- R08-A (fork, wt/memfix, branch r08-memfix):
  - per-op bandwidth on one GPU with torch.profiler, compared with the measured peak; audit table in runs/r08/audit/AUDIT.md;
  - find the sources of the int64 sum, the non-vectorized bf16 mul/add, the casts and the D2D copies;
  - fix whatever can be fixed quickly.
- R08-B (fork, wt/normfuse, branch r08-normfuse): fuse RMSNorm(q/k) + RoPE; fuse LayerNorm + modulate provided Welford is reproduced bit for bit, and stop if that is not possible.
- GPU3 was occupied by another user's policy service, so two-GPU benches used only 0/1.
- R08-A audit conclusions (runs/r08/audit/AUDIT.md):
  - peak bandwidth about 4.3 TB/s.
  - **bug**: once a compiled function has been called a single time under any TorchDispatchMode (such as SAC's checkpoint context), it falls back to eager permanently. So SF_COMPILE's long-sequence modulate/gate never took effect in training, and the non-vectorized broadcast mul/add ran at only 34% of bandwidth, 1.23 s per window. Reproduction: runs/r08/audit/test_poison.py.
  - Other candidates:
    - the frozen teacher/T5 cast parameters from fp32 to bf16 on every forward, 0.29 s per window; storing them in bf16 would save about 20 GB per GPU.
    - with gradients, the generator's in-place KV-cache writes make the backward do a gradient D2D over the whole cache, 0.3–0.4 s per window.
    - the rollout's KV writes, 0.14 s.
  - The "int64 sum 0.39 s" was misattributed during grouping; it is actually the column reduction of bf16 modulation gradients, which would have to reproduce ATen's reduction order bit for bit; hard, not for now.
  - Fix order: modulate/gate as Triton without compile → frozen models stored in bf16 → KV-cache gradient → KV writes.
- R08-A done, r08-memfix fast-forwarded in (cc-opt 61ca98d):
  - SF_EW_LONG=1: modulate/gate of every size goes through Triton, with fused backward products; the compile wrapper detects a dispatch mode and goes around it, so the function is never pushed back to eager permanently.
  - SF_FROZEN_BF16=1: the frozen teacher/T5 are converted to bf16 before FSDP wrapping.
  - Both N0; ABBA 5279.8 → 5177.0 (−1.9%, near the noise); allocated 91.61 → 72.99, reserved 125.6 → 106.3 (about 19 GiB saved).
  - New configuration runs/r08/final4.json.
  - Found upstream's "phantom gradient": the exit step writes the KV cache in place with gradients, then the ctx forward overwrites the same slice under no_grad; when later blocks read the cache, gradient still flows through CopySlices back into the overwritten exit-step keys (runs/r08/memfix/test_phantom.py; in the toy example dL/dp1=10 instead of 0). Upstream semantics kept, not changed.
  - R08-A dispatched again for r08-memfix2: re-evaluate SAC_SAVE with the freed memory, and keep T5 resident to save its all-gathers.
- R08-B done, r08-normfuse merged into cc-opt (merge a93c229, no conflicts):
  - SF_FUSE_QKROPE=1: the forward of q/k RMSNorm + RoPE in one kernel, backward unchanged. At 32760×5120, 500.6 → 327.8 µs.
  - SF_FUSE_LNMOD=1: ATen's LayerNorm reproduced bit for bit in Triton (84/84 probes identical), then fused with modulate. At 32760×5120, 393.5 → 286.8 µs.
  - check --quick N0; counters show the SAC recomputation path also uses the fusions.
  - 8 interleaved runs on the same tree: 5356.5 → 5250.0 (−2.0%), memory unchanged.
- New configuration runs/r08/final5.json = final4 + QKROPE + LNMOD. The combined FA2 bitwise check --quick was queued on GPU2/3 (runs/r08/check_quick_final5_fa2), while R08-A's memfix2 timings ran on two pairs of GPUs.
- Combined bitwise check (cc-opt a93c229 + final5_fa2): check --quick PASS, bitwise, N0. Counts per rank: QKROPE nograd 1620 / grad 3000 / bwd 1500; LNMOD 1645/3025/1525; FUSE_EW gate 1620, gate_grad 3000, mul2 1500, rms 1200, rms_bwd 1500; CUDA graphs 8 per rank, all replaying normally.

## R08-A2 memfix2 → runs/r08/memfix2/REPORT.md
- a803b97 SF_T5_RESIDENT=1 (T5 all-gathered once, kept); 576d789 SF_SAC_SAVE counters. Both FA2 quick check N0.
- Best: final4 + T5R + SF_SAC_SAVE=ffn2,o,co (bidir scope): reserved 124.2–124.7 GiB; ABBA −1.9%, pooled 5v5 −2.3% (pair noise ±3–4%, node load ~105).
- s2 (7 roles) reserved 128.6 (margin too thin); causal scope no gain (gen_bwd CPU-bound); teacher resident not worth it (~1% ceiling, no memory).
- Phantom-gradient quantified (phantom_attn.py, fp64 toy): Wk grad +49.7%, Wv +61.9% vs. cut-graph; upstream semantics kept.
## R08 finals
- full check final5 (sf @ a93c229) on GPU2/3 → runs/r08/check_final5.
- wt/final6 = branch r08-final6 e61a2e0 (a93c229 + merge r08-memfix2); final6 = final5 + T5R + SAC_SAVE=ffn2,o,co.
  run_final6.sh: after final5 timing gate → quick FA2 check on GPU0/1, then full check final6 on GPU0/1
  (its timing overlaps final5 trajectory phase on GPU2/3 for the whole window, i.e. uniform background).
- final6 FA2 quick check (wt/final6 e61a2e0, GPU0/1): PASS, bitwise, N0. Counters per rank: SAC_SAVE bidir o/co/ffn2 saved 120 / reused 120 each; T5_RESIDENT 50 units gathered once over 8 forwards; QKROPE/LNMOD/FUSE_EW same as final5. Full check final6 started on GPU0/1.
- **check_final5 (GPU2/3): PASS, verdict equivalent (N1 statistical, 3 seeds), bench speedup 2.4413 — NOT trustworthy.**
  Two reference windows were contaminated: ref_a w0 14523 ms/iter, ref_b w1 15053, +36–41% vs the
  clean 10664 / 10316 (short jobs of another user on GPU2; none of our runs on GPU0/1 in those windows).
  cand windows clean 5293 / 5062 → 5177.2 (R06 5436.8: −4.8%). Clean-window ref 10490 → 2.03×; vs R05/R06 refs (10660–10825) → 2.06–2.09×.
- check_final6 timing windows (GPU0/1, overlapping final5 trajectory on GPU2/3 throughout): ref_a 9984/11208, cand 5140/5243, ref_b 10649/11403 → about 2.08×, cand 5191.5 (same as final5 within noise). Peak reserved cand 125.8 GiB.
- **check_final6 (wt/final6 e61a2e0, GPU0/1): PASS, equivalent (N1, seeds 1000–1002), speedup 2.0823** (ref 10811.0, cand 5191.9, ref spread 0.040; no contaminated window). Trajectory endpoint stats identical to final5 (only FA3 changes bits). cc-opt fast-forwarded to e61a2e0; final config = runs/r08/final6.json.
  T5R + SAC_SAVE not visible in this single check (cand 5191.9 vs final5 5177.2, within noise); ABBA in memfix2 said −1.9 to −2.3%.
