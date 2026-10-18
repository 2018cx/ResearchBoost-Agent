# LLaVA: supervision record

*Translated from the Chinese original.*

One Codex `/goal` session (4 × H20) optimized LLaVA-1.5-7B stage-2 fine-tuning; one supervisor formulated, launched and watched it. This file is the supervision guide, the record of interventions and the lessons. The full audit of judge changes is in [`task/judge_changes.md`](task/judge_changes.md).

## Results

| Phase | The single criterion | Speedup | Notes |
|---|---|---|---|
| Original task statement, stopped by itself after 24 rounds | full check PASS / N1 / statistically equivalent | **1.0947×** (20994.6 → 19178.0 ms/step) | ZeRO-3 parameter persistence, Apex FusedRMSNorm, bf16 reduced-precision reduction; other directions were noise-level, OOM or not allowed. The wrap-up audit gave a quantitative reason: GEMMs take about 84% of GPU time, with a lower bound of about 15.5 s/step, while 1.5× needs about 14 s/step |
| After three constraints were relaxed | full check PASS / N1 / statistically equivalent | **1.6138×** (20938.0 → 12974.6 ms/step) | token-length bucketing, global token loss, MLP on valid rows only, retaining the last layers' activations, balancing within a micro-step |
| After the target rose to 2.5× | full check PASS / N1 / statistically equivalent | **2.3851×** (20934.3 → 8777.1 ms/step) | packed attention and decoder, SwiGLU backward recomputation, memory-efficient RMSNorm, Liger RoPE, standalone FP32 CE, balancing over the whole optimizer step |

## Formulation

- Upstream haotian-liu/LLaVA `c121f04`. The reference repository puts files that upstream's `.gitignore` excludes, such as `scripts/zero*.json`, under version control; otherwise candidate and reference are not byte-identical.
- Workload: `scripts/v1_5/finetune.sh` unchanged, 7B, 4 GPUs × 16 per GPU × gradient accumulation 2 = global 128 (upstream 8 × 16 × 1).
- Reference baseline about 20960 ms/step (two runs 0.08% apart). Hotspots: bf16 GEMM about 68%, NCCL AllGather 14.9% + ReduceScatter 11.3%, flash-attn about 2%.
- Bitwise gate: the reference is identical across two runs in deterministic mode (1181 tensors per GPU per case).
- Statistical gate δ (frozen from 6 reference seeds): train_loss_last20 σ 0.0071 → δ 0.0142; probe_loss_last20 σ 0.0033 → δ 0.0065.
- `numerics selftest` catches all 12 items. The result of gate 3's control arm (learning rate × 1.02, script `formulation/prep/control_arm.sh`) is not stored in this directory. The ceiling estimate stays on the formulation side (`formulation/CEILING_NOTE.md`) and never enters any message to the agent.

Acceptance (at formulation, and rerun after every judge change):

| Candidate | Result |
|---|---|
| Unmodified (quick) | PASS, N0, bitwise |
| Unmodified (full check) | gate 1 0.9995 < 1.02 → `no_benefit` |
| Changed setup in `train()` | ERROR (setup guard) |
| Sampler dropping the gradient-accumulation factor | INCORRECT (gate 0, data pipeline) |
| Mathematically equivalent RMSNorm reordering | PASS, statistical gate equivalent |
| Projector detached (a whole block not training) | the statistical gate once gave a false negative → INCORRECT after adding the `dead_gradient` exact item |

## Interventions (all at the level of rules, facts or infrastructure; no optimization direction was ever given)

1. The candidate repository lacked files excluded by upstream's `.gitignore`, so candidate and reference differed → both now include these files; work started once `diff -r` matched.
2. The agent marked the goal blocked, but its wrap-up audit had only a list of failures with no per-hotspot explanation, and it had read only 3 skill cards → asked to complete it according to the wrap-up clause of `GOAL.md`.
3. The audit stated a fact given by the task statement as "Apex/Liger is not available in this environment" → asked to correct it or attach measured evidence.
4. The audit wrote "requires numerical validation" as "forbidden by the rules" → reminded that the only criterion is a check PASS, that N1 and N2 are legitimate tiers, and that the statistical gate exists for non-bitwise changes. Once these two were separated, it tried fused ops and reached 1.0925×.
5. The agent said "still in progress", but the goal was stalled and the GPUs idle → sent `/goal resume`.
6. **Judge correction (a calibration error by the formulator, ratified by the owner afterwards)**: gate 1's per-step loss gate was originally ≤5% at every step, which systematically blocked legitimate N1/N2 changes (FusedAdam deviated 5.22% in the high-curvature warmup). It became "step 1 ≤0.5%, later steps not 0, not NaN/inf and ≤25%"; the old and new rules were replayed on the 15 existing timing runs, and acceptance was rerun under the new threshold. A rule followed from this: **any threshold one wants to change only after it rejected something is cleared with the owner first.**
7. When pushing the new threshold, two fields of `bench.json` that had only been changed on the machine were overwritten with old values, and one of the agent's checks reported `reference checkout changed` → fixed, and the repository copy was aligned with the machine.
8. The best candidate had only `check --quick` as evidence → asked to run a full check.
9. A hole in the judge: the data probe ignored the switch file, so when a candidate put a reordering behind a switch the probe did not see it and wrongly applied the 0.5% band, and conversely a data change could hide behind a switch and slip past gate 0 → fixed, and the agent was told.
10. In one round the agent withdrew a switch that was bitwise neutral because it was judged to bring no speed gain; another round had been rejected by gate 3's correctness check, not by the benefit gate → these two facts about the judge were passed to it, and it then restored the implementation itself.

## Task statement relaxation (authorized by the owner)

With `bench/rank_report.py` the agent measured: padding takes about 33% of the token slots of the dense layers, the heaviest GPU of each optimizer step carries about 8% more than the four-GPU mean (worst steps 1.19–1.23), while the total of real tokens is balanced by upstream within 1%, so all of the imbalance comes from padding. It correctly concluded that "`GOAL.md` requires a bitwise-identical assignment, so the sampler cannot change", and did not try to work around the judge. So the owner decided whether to relax this, and relaxed three things:
- how the 128 samples of one optimizer step are distributed over the 8 micro-batches;
- the length key used for grouping may be the real token length;
- the loss may be the global token mean over the whole step (the loss only ever counted valid tokens; padding is filled with -100).

**Which 128 samples each optimizer step trains on is still an exact item; only how those 128 are used within the step changes.** This changed the task statement; it did not loosen the judge to let a candidate through: all three happened before any candidate was rejected by them. The judge changed in six places accordingly, and gate 3's two endpoints were later switched to values computed by the judge itself (this one came after a candidate was rejected; the owner was asked first, and the evidence attached shows δ unchanged and the old PASS candidate's result unchanged digit for digit under the new endpoints).

## What the judge actually caught

- **Broken gradient**: a forward-only Liger `swiglu_forward` bypassed autograd and measured 1.456×; `dead_gradient` found 96 parameters without gradient; with an implementation that has a backward, the gain fell back to 1.099×.
- **Objective changed**: length bucketing under upstream's "mean of means" normalization shifted each token's weight by 28%, and a 1.325× candidate was rejected at gate 3 (probe loss +2.3%, 7.04σ, same direction for all three seeds). The global-token-mean normalization reduced the shift to 0.75%.
- **The reported number was not the objective actually optimized**: one candidate used a loss-gradient hook to report the logged loss under the old normalization while scaling the backward under the new one; that PASS was voided. Once the endpoints were computed by the judge, the agent removed these hooks itself.
- **Timing and trajectories on different code paths**: a 1.3254× PASS was withdrawn by the agent itself because timing used 4 workers and the trajectories 1.
- **Memory**: a 1.5207× candidate ran out of memory while recording the bitwise gate (which also needs the fp32 gradients, master weights and both Adam states of every parameter); it was ERROR, and the speed did not count as a result.
- **Random stream**: Triton 2.1's file cache drew temporary file names from the global `random`, caught by the RNG-fingerprint exact item; the candidate replaced that module's reference with an independent `SystemRandom`, without asking for any relaxation.

## Lessons

- **Reference records must be recorded inside the sandbox.** `rng_fingerprint()` hashes `torch.cuda.get_rng_state_all()`, which depends on the number of visible devices: 4 in the sandbox, 8 on the host. A fingerprint recorded on the host can never be matched by a candidate, while all tensors are identical, so it looks like "the candidate is wrong". `bench.py` now checks the number of visible GPUs at recording time and fails outright.
- **Do not hand-roll torchrun to record reference records**: it crashes in glibc `malloc_consolidate` while exiting. Use the bench's own `record_cases()` → `run_gpu()` (`formulation/prep/record_via_bench.py`).
- **Kill leftover app-server daemons before restarting the TUI**, otherwise the new TUI may hang forever without an error ([`goal/supervise/relaunch_resume.sh`](../../goal/supervise/relaunch_resume.sh)).
- In the sandbox, torchrun uses a static rendezvous: `--master-addr=127.0.0.1 --master-port=<free> --local-addr=127.0.0.1`.
- The PATH in `sandbox.env` must end with `/usr/sbin:/sbin`: triton 2.1 finds libcuda with `ldconfig -p`, and without it neither torch.compile nor liger starts. The environment also needs `protobuf` and `setuptools==69.5.1`; each rank uses its own Triton and DeepSpeed autotune cache directories.
- When estimating a ceiling, list the constraints the owner might relax too. The 1.75–1.9× ceiling estimated under the original task statement rested on exactly the constraints that were relaxed later.

## Handoff

- Resume: `goal/supervise/relaunch_resume.sh llava "0 1 2 3" <thread id>`, then type `/goal resume`. Give rule-level reminders only; clear any threshold change with the owner first.
- Rerun the verdict: inside the sandbox, `python bench/bench.py check --tree llava --config <switch file> --out <new dir>` (about 1–1.5 hours).
- After a machine rebuild, rebuild the environment, assets and fixture following [`formulation/prep/README.md`](formulation/prep/README.md); the digests of the five frozen inputs must match `task/formulation.json`.
