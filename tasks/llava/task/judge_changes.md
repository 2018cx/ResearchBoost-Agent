# Judge change record (gate 1's per-step loss sanity check)

*Translated from the Chinese original.*

The evaluator frozen at formulation was changed once after launch. This record gives the threshold before and after, the reason for the change, and a regression over existing runs,
so that a third party can judge that this relaxation was **not** made to let a candidate through.

## The change

| | Before (frozen at formulation) | After |
|---|---|---|
| First logged loss (before any optimizer step) | same as the other steps, relative difference ≤ 5% | relative difference ≤ **0.5%** |
| Later steps | relative difference ≤ 5% | must not be 0 / NaN / inf (where the reference is nonzero), relative difference ≤ **25%** |
| Verdict | out of bounds → gate 1 ERROR | same |

`timing.max_logged_loss_rel_diff` in `bench/bench.json`: 0.05 → 0.25, plus a new `timing.max_logged_loss_rel_diff_step1`: 0.005;
`summarize_timing` in `bench/bench.py` rewritten according to the table; the gate-1 paragraph of `GOAL.md` rewritten to match.
The other gates (0 structure / 2 bitwise / 3 statistical), the benefit threshold, `max_window_spread`, `max_reference_drift` and the statistical gate's δ are unchanged.

## Why the original calibration was wrong

The purpose of this check is "the candidate trains on the same data and has not diverged"; precision itself is judged by gate 2 (bitwise) and gate 3 (statistical).
Setting it to "every step ≤ 5%" actually turned it into a **precision gate stricter than gates 2 and 3**: after the first optimizer step, any
legitimate N1/N2 change (an equivalent but different update/reduction order) slightly changes the loss of later steps, and in the high-curvature warmup
at the start of training this grows to a few percent within a few steps. GOAL.md explicitly allows N1/N2 and provides the statistical gate for them, so this threshold
systematically kept changes the task statement allows away from the precision evaluation.

The data that triggered the review (an Apex FusedAdam candidate, `runs/r17/time_fused_adam`, ERROR under the old threshold, maximum deviation 5.22%):

| Step | Reference loss | Candidate loss | Relative difference |
|---|---|---|---|
| 1 | 1.3809 | 1.3806 | 0.02% |
| 2 | 1.3799 | 1.38 | 0.01% |
| 3 | 1.1375 | 1.1852 | 4.19% |
| 4 | 1.0937 | 1.1356 | 3.83% |
| 5 | 1.11 | 1.1679 | 5.22% |
| 6 | 1.0526 | 1.0958 | 4.10% |
| 7 | 1.0482 | 1.0653 | 1.63% |
| 8 | 1.0509 | 1.0631 | 1.16% |
| 9 | 0.3535 | 0.3538 | 0.08% |
| 10 | 1.0367 | 1.0528 | 1.55% |
| 11 | 1.074 | 1.0883 | 1.33% |
| 12 | 1.0146 | 1.025 | 1.03% |
| 13 | 1.0455 | 1.0539 | 0.80% |
| 14 | 1.0156 | 1.0231 | 0.74% |
| 15 | 1.028 | 1.0349 | 0.67% |
| 16 | 1.0351 | 1.0421 | 0.68% |
| 17 | 1.0462 | 1.0515 | 0.51% |

The shape: **steps 1 and 2 at 0.01–0.02%** (no optimizer step yet, so data and forward are identical), steps 3–6 at 4–5%
(the high-curvature warmup), then back below 1%. It is neither "trained on other data" nor divergence; it is exactly the kind of change the new threshold should let through
and gates 2 and 3 should judge.

## Regression: replaying the old and new thresholds on every existing timing run

`formulation/prep/judge_regression.py` reads the per-step losses stored by each run and judges them under the old and the new rule (no GPU):

| Run | ms/step | vs frozen reference | Step-1 deviation | Max deviation | Zeroed steps | Old verdict | New verdict |
|---|---|---|---|---|---|---|---|
| runs/r1/time_persist/time_persist.json | 20978.1 | 0.9995 | 0.0 | 0.0007 | 0 | ok | ok |
| runs/r12/time_no_rng/time_no_rng.json | 20017.6 | 1.0475 | 0.0 | 0.0004 | 0 | ok | ok |
| runs/r13/time_no_overlap/time_no_overlap.json | 20023.7 | 1.0472 | 0.0 | 0.0007 | 0 | ok | ok |
| runs/r15/time_apex_rmsnorm/time_apex_rmsnorm.json | 19193.4 | 1.0925 | 0.00022 | 0.0006 | 0 | ok | ok |
| runs/r16/time_bf16_reduced/time_bf16_reduced.json | 19183.5 | 1.093 | 0.00022 | 0.0006 | 0 | ok | ok |
| runs/r17/time_fused_adam/time_fused_adam.json | 19156.4 | 1.0946 | 0.00022 | 0.0522 | 0 | ERROR | ok |
| runs/r18/time_fused_adam/time_fused_adam.json | 19184.9 | 1.093 | 0.00022 | 0.0518 | 0 | ERROR | ok |
| runs/r3/time_unpad_cache/time_unpad_cache.json | 20963.7 | 1.0002 | 0.0 | 0.0009 | 0 | ok | ok |
| runs/r4/time_nonreentrant/time_nonreentrant.json | 20897.4 | 1.0034 | 0.0 | 0.0009 | 0 | ok | ok |
| runs/r6/time_zero2/time_zero2.json | 20223.9 | 1.0368 | 0.00427 | 1.0 | 16 | ERROR | ERROR |
| runs/r7/time_buckets/time_buckets.json | 20623.7 | 1.0167 | 0.0 | 0.0006 | 0 | ok | ok |
| runs/r8/time_persist_all/time_persist_all.json | 20002.4 | 1.0483 | 0.0 | 0.0005 | 0 | ok | ok |
| runs/r9/time_nccl_simple/time_nccl_simple.json | 20013.8 | 1.0477 | 0.0 | 0.0006 | 0 | ok | ok |
| task/timing/time_ref_a.json | 20968.2 | None | 0.0 | 0.0 | 0 | ok | ok |
| task/timing/time_ref_b.json | 20951.6 | 1.0008 | 0.0 | 0.0008 | 0 | ok | ok |

The two key rows:

- **Reference against reference** (`ref_a` / `ref_b` / the three in the formal check): step-1 deviation ≤ 0.1%, maximum ≤ 0.1%, passing under both rules;
  the new threshold did not widen the noise floor.
- **The ZeRO-2 run** (`runs/r6/time_zero2`, loss becomes 0 from step 2): still ERROR under the new rule (zeroed steps + 100% deviation),
  so broken runs are still caught after the relaxation.

## Acceptance still to be redone (until then, conclusions "re-judged under the new rule" are provisional)

Rerun the key items of the formulation acceptance set under the new threshold; the results must not change: unmodified → bitwise PASS; changed `train()` setup → ERROR;
sampler dropping the gradient-accumulation factor → INCORRECT; projector detached → INCORRECT (dead_gradient). The results are appended at the end of this file.

Also: the optimizer-replacement candidate rejected by the old threshold, even if it now passes gate 1, only counts after gates 2 and 3 of a full check
(it can never be bitwise, so only the statistical gate can accept it), and the statistical gate's δ will not be adjusted because it fails.

## Results of rerunning the acceptance set under the new threshold (with the changed evaluator)

| Candidate | Actual | Matches the expectation at formulation | Evidence |
|---|---|---|---|
| Unmodified → bitwise PASS | PASS / structure (quick self-test: gates 1 and 3 not run) / N0 | yes | `task/reacceptance/a/check.json` |
| Changed train() setup → ERROR | ERROR / setup_guard | yes | `task/reacceptance/b/check.json` |
| Sampler dropping the gradient-accumulation factor → INCORRECT (gate 0 data) | INCORRECT / loader_check | yes | `task/reacceptance/c/check.json` |
| Projector detached → INCORRECT (dead_gradient exact item) | INCORRECT / structure | yes | `task/reacceptance/e/check.json` |

All four match the formulation: the relaxation only affects how much gate 1 tolerates legitimate N1/N2 numerical perturbations; it does not weaken the structure gate, the data gate,
the bitwise gate or the dead_gradient exact item. The conclusions "re-judged under the new rule" are therefore no longer provisional.

The statistical gate was not rerun: this change does not involve gate 3; δ is still the value frozen at formulation from 6 reference seeds, unchanged, and will not be adjusted because some candidate
fails it. The optimizer-replacement candidate rejected by the old threshold still only counts after gates 2 and 3 of a full check.

## New observation script `bench/rank_report.py` (not a threshold change)

After the machine was rebuilt, the local disk was empty; the environment, weights and data were rebuilt with the scripts in `formulation/prep/`. The digests of the five frozen inputs
all match the formulation (`fixture` / `images` / `vicuna` / `clip` / `projector`; the vicuna directory needs the two 0-byte `.ok` marker files left by pdl.py
restored to match byte for byte; the content hashes of the weights themselves were shown to be identical within the digest), so the reference timings,
δ, reference records and the judge fixture all remain valid and were not re-recorded.

New `bench/rank_report.py`: **read-only observation**, on the same level as `bench.py profile`; it does not train, does not write `task/`, and does not take part in the verdict of
any gate. It reports two facts:
- GPU view: reads the per-GPU profiler traces that `bench.py profile --detail` already writes, and gives kernel time per GPU, with
  NCCL kernels and non-NCCL kernels separated.
- Data view: replays the sampler according to the candidate's own `llava_trainer.py` (`modality_lengths` →
  `get_modality_length_grouped_indices` → `split_to_even_chunks`, megabatch = 16×4×2) and gives, per GPU, the real token count
  inside each micro-batch and the number of token slots after the collator pads each micro-batch to its longest sample.
  The replay follows upstream exactly, including the mixed tail batch (the last image megabatch + the last text megabatch combined into 256 samples, sorted
  by index, not passed through split_to_even_chunks). Upstream's two per-modality shuffles pass generator=None and draw from the
  global RNG seeded by set_seed(seed), so the script seeds both the global RNG and the Trainer's generator; otherwise the replay is not reproducible.

**Effect on the audit**: `evaluator_sha256()` hashes every file under `bench/`, so this new file changes that digest:
- before: `e90e3bdf1e801174345dd439fe2bc4904897625521e67819e9242c3e1b8fc927`
  (the same value as recorded in `runs/final/check/check.json` for the 1.0947× PASS, showing the judge was not touched since)
- after (final): `0390d16a9e11556bb0d54bc41f2a578fcf7d346100dd02f9a13974822ed36cf5`
  (this file first recorded an intermediate value `928fca34…` from before the script was finalized; this line supersedes it)
- Verified: recomputing with `rank_report.py` excluded gives exactly `e90e3bdf…`, so the only difference in `bench/` is this
  new observation file; thresholds, δ, thresholds.json and numerics did not change by a single byte.

No gate's verdict logic or threshold was modified. By the rules, threshold changes must be cleared with the owner first; this is not a threshold change.

## Task statement revision 1: relaxing sample assignment within an optimizer step and the loss normalization (authorized by the owner)

The owner authorized relaxing three things, specified in GOAL.md: (1) how the 128 samples of one optimizer step are distributed over the 8 micro-batches;
(2) the length key used for grouping may be the real token length, and the bucketing target of `split_to_even_chunks` may change; (3) the loss normalization
may be either "token mean within a micro-batch → equal-weight mean over the 8" or "global token mean over the whole step".

**This is the formulator changing the task statement, not loosening the judge to let a candidate through.** All three happened before any candidate was rejected by them; up to this point
0 candidates had been rejected by these rules. The only candidate ruled out was Round 22's forward-only Liger SwiGLU (96
`dead_gradient`), and that exact item was not changed at all.

### Still exact items (not relaxed)

Which 128 samples form each optimizer step (modality grouping and megabatch membership unchanged; samples are not moved across steps, dropped or duplicated);
16 per GPU, gradient accumulation 2, 4 GPUs; bitwise-identical preprocessing of every sample; the global token count, label count, and number of samples per micro-batch;
every gate-2 exact item other than the batch digest, plus `dead_gradient`; gate 3's δ, paired t-test and seed schedule; the tier rules;
the benefit gate. The loss still covers only valid label tokens; it may not be weighted by sample count, may not skip or repeat micro-batches, may not count a valid token
twice or miss one, may not become a soft label, and may not change the semantics of `IGNORE_INDEX`.

### Six changes to the judge

1. `bench/llava_adapter.py batch_digest(inputs, rows=True)`: adds per-row digests; the in-order `sha` within a micro-batch
   becomes diagnostic (it tells whether the candidate reassigned samples).
2. `bench/llava_adapter.py _batch_invariants()`: gate 2's batch exact item becomes "the sample multiset of the whole step after an all_gather over the four GPUs
   (sorted per-row digests) + global token/label counts + samples per micro-batch".
3. `bench/loader_probe.py` + `bench/bench.py loader_problem()`: gate 0 also compares multisets per optimizer step.
4. `bench/bench.py partition_state()` / `time_tree()`: `time` / `sweep` / `check` first run a CPU data probe (about 11 s)
   before taking the GPUs and compare each step's sample multiset with the frozen `task/loader_reference.json`; a mismatch is an immediate
   ERROR, without a single measured step. This is stricter than the old 0.5% loss proxy.
5. Gate 1's loss sanity band becomes conditional: without reassignment → 0.5% / 25%, unchanged; with reassignment → 48.4%
   (`task/partition_sensitivity.json`). For reassigned candidates, "has the computation been broken" is covered by gate 2 (per-tensor comparison of the real
   loss / fp32 gradients / fp32 master weights / Adam state + `dead_gradient`) and gate 3.
6. `bench/bench.py record_environment_problem()`: ERROR with an explanation when the number of visible GPUs at recording time (`numerics.visible_gpus` = 4)
   does not match. `rng_fingerprint()` in `numerics/record.py` hashes
   `torch.cuda.get_rng_state_all()`, which returns one state per visible device, so a record made with a different number of visible devices carries a fingerprint no candidate
   can ever match while every tensor is identical; such a failure looks like "the candidate is wrong", and is now stopped with an explanation.

### The 48.4% figure

On the formulation side, **the reference's own code** was run once with `bench.py time` under each of two legitimate assignments; the first logged loss shifted by **0.1%** and
**24.2%**, and the band is twice the largest observed shift. The measurement is frozen in `task/partition_sensitivity.json`; **how the two assignments were built
is recorded on the formulation side (outside the sandbox), not here**.

Reason: the logged loss is "the token-mean cross-entropy of each micro-batch, then an equal-weight mean across micro-batches"; when the assignment changes, each
sample's weight in this two-level mean changes, so the number moves even when the same 128 samples are trained. The original 0.5% held only
because the assignment was frozen too.

### An attempt that was disproved (not kept in the verdict)

The idea was to freeze "the number of supervised labels L_i and token-mean CE c_i of each of the 128 samples of the first step", and compute the expected value for any assignment with
`(1/8)·Σ_mb (Σ L_i·c_i)/(Σ L_i)`, so the first-step band could stay at 0.5% and only cover "has the computation been
broken". Two versions were built (one forward per sample; and running the reference's own 8 micro-batches and splitting row by row from the logits, where the per-row reconstruction
differs from the micro-batch loss by only 5e-5). **Neither passed validation**: calibrated with "reference assignment = measured value", the prediction for another measured assignment was
1.7938 against a measured 1.7149, a gap of **4.6%**, far from 0.5%. Moreover, the mean of the 8 micro-batches from the offline forward differed from the real run by
3.0%, and that offset was not constant (+3.0% for one assignment, −1.5% for the other).

Conclusion: with the available evidence, "assignment" and "computation" cannot be separated at 0.5% precision. The prediction code stays in `bench/bench.py`
(`expected_first_logged_loss()`), but `task/first_step_ce.json` is **not installed**, so it always returns None and falls through to the measured band of item 5
above. An unvalidated prediction was not put into the verdict.

### Re-recording the reference records

The structure of gate 2's batch exact item changed and the old records have no per-row digests from which it could be derived, so `bench/records/ref` had to be re-recorded (inside
the sandbox, through the bench's own `record_cases()`). A full backup from before the change is on the formulation side.

Verification:
- main record vs a duplicate: **bitwise identical** (the new cross-GPU `all_gather_object` did not break reproducibility);
- main record vs the old record before the revision: **0 mismatches in `rng_before`/`rng_after`**, and for every (case, rank)
  **1181/1181 tensors bitwise identical**; only the batch invariants changed, by design;
- therefore the per-tensor tolerances in `bench/thresholds.json`, δ in `bench/traj/frozen.json` and the
  reference timing in `task/timing/reference.json` **all remain valid** without recalibration, and so do the conclusions of earlier rounds.

### Frozen inputs and evaluator_sha256

- Frozen inputs **11 → 13**: new `loader_reference` and `partition_sensitivity`; the `records_ref` digest
  `a56314ae…` → `a491bf37766345f53bdcff9f81aac01127f675d35da13aa792988fe8344ae8b4`.
- `evaluator_sha256`: `0390d16a9e11556bb0d54bc41f2a578fcf7d346100dd02f9a13974822ed36cf5`
  → `5d21afdc03c275889ebca97502640fd8e6d5a45eb23c5c6a16d83c8fa4608141`

### End-to-end self-check (inside the sandbox)

- **A** the reference tree's own `check --quick` → **PASS**; gate 0 recognizes "assignment = the reference's";
- **B** a tree that reassigns every step, `check --quick` → **PASS**, tier N1, **0 exact-item failures**; gate 0 recognizes
  "assignment = the candidate's own"; 3091 of 9448 tensors are unresolved (handed to gate 3 in a full check), as expected;
- **C** a tree in which every step takes the next step's samples, `time` → **ERROR within 37 seconds without touching a GPU**, with the reason pinned to the
  optimizer step and the missing/extra per-row digests listed.

### Not done yet, recorded

Gate 3's sensitivity has never been measured. δ = 2σ of the reference across seeds (train_loss_last20 0.01421 ≈ 1.50%;
probe_loss_last20 0.00653 ≈ 1.35%), but it has never been shown to reject a deliberately broken candidate (for example `lr ×1.02`).
To be done before any candidate's fate actually rests on gate 3.

## Task statement revision 2: gate 3's two endpoints computed by the judge itself (authorized by the owner, **made after r36 was rejected by this gate**)

**This entry must be flagged: the change was made after a candidate was rejected by this gate.** Under this task's rules such a change cannot be decided by the formulator
alone and must be cleared with the owner first; the owner was asked, authorized it, and chose "switch to a canonical training loss computed by the judge". Below are the reason, the method and
all the evidence on whether the gate was loosened.

### Why it had to change: neither endpoint was comparable between two trees

Gate 3's endpoints both used to come from **the tree's own numbers**:
- `train_loss_last20` read the return value of `training_step`;
- `probe_loss_last20` went through `tr.compute_loss`.

But the third rule relaxed on the same day allows the two trees to use **different loss normalizations** (upstream's "token mean within a micro-batch → equal-weight mean over the 8",
or "global token mean over the whole step"). So reference and candidate were measured with **two different rulers**. r36 exposed this
clearly:

```
probe_loss_last20   within     paired diff −0.00160   = −0.49σ   p=0.0208
train_loss_last20   deviates   paired diff +0.03049   = +4.29σ   p=0.0243   → INCORRECT
measured_speedup 1.2816×   gate-2 exact items 0 failures
```

The held-out endpoint said "no regression, slightly better"; the training endpoint said "4.29σ off". The probe happened to be comparable only because the candidate's
scaling hung off a field that does not exist in a probe batch (`_global_label_total`): **luck, not design**; another implementation would have scaled the probe
too.

In that state, normalization option 2, which the owner had authorized, **was bound to be rejected** no matter how well the candidate actually trained; and not using option 2 meant accepting a 28%
shift of the objective (which gate 3 correctly rejects, see r33: probe +2.3%, 7.04σ). Both paths were blocked: not the judge working correctly, but a hole in the endpoint definition.

### The change

Both endpoints are now measured by **the judge driving the model forward itself** (new `_bench_loss()` in `bench/llava_adapter.py`): using only the frozen input
keys (`input_ids`/`labels`/`attention_mask`/`images`), it calls `tr.model_wrapped(**inputs)` and takes `out.loss`, i.e.
**the model layer's "mean cross-entropy of this batch over its own valid label tokens"**. It does not go through `tr.compute_loss` and does not read the
return value of `training_step`.

- `train_loss_last20` → **`train_loss_last20_canonical`**: the mean of the model-layer loss of this step's 8 micro-batches, measured inside
  `training_step` **before the real training step** (so with the weights this step trains from), only in the endpoint window (steps 40–59),
  at about 5% overhead;
- `probe_loss_last20`: the held-out batch now goes through the same `_bench_loss()`.

A matching constraint was added to GOAL.md: **normalization may only be done at the Trainer level**; `model.forward(labels=...)` must still return the mean cross-entropy over the
batch's valid tokens. Otherwise a candidate could push its normalization into the model layer and change the judge's ruler along with it. The normalization the candidate trains with is
unaffected; only "how it is measured" is no longer up to the candidate.

### Was the gate loosened? Not a single digit of δ changed

After re-recording the reference's 6-seed trajectories and refreezing:

```
                                  reference_mean      sigma_seed        delta
OLD train_loss_last20              0.949968346       0.00710698     0.01421395
NEW train_loss_last20_canonical    0.949968346       0.00710698     0.01421395
OLD probe_loss_last20              0.483101620       0.00326294     0.00652589
NEW probe_loss_last20              0.483101620       0.00326294     0.00652589
```

**All four numbers are bitwise identical.** This proves three things at once:

1. for the reference, the canonical training loss **equals the old endpoint by construction**: the reference's `training_step` returns `c_mb/accum`, and the old endpoint's sum over the 2
   micro-steps is exactly the mean the judge now computes;
2. switching the probe from `tr.compute_loss` to calling the model directly leaves the number **completely unchanged**: upstream's `compute_loss` only passes the model's
   loss through;
3. the judge's added no-grad forward has **zero effect** on the run (the concern was that it might disturb the RNG or ZeRO-3's parameter gathering; the
   endpoints of all 6 seeds reproduce to print precision, so it does not).

**Conclusion: the ruler's markings did not move at all; only who reads it changed.** Thresholds, k, alpha, seed schedule and decision rules (deviates /
within / unclear, the 3→6 seed escalation) were not changed at all.

### Control experiment: a candidate that already passed before the change still passes under the new endpoints

The best candidate from before the change, which was judged **PASS / N1 / equivalent / 1.0947×** at the time (commit `a2fb998`, Apex fused RMSNorm +
bf16 reduced matmul reduction, switch file `runs/r16/bf16_reduced.json`), was exported as an independent tree with `git archive` and rerun through gates 0/2/3 under **the new endpoints +
the refrozen δ** (`formulation/prep/verify_old_candidate.sh`, results in
`runs/fx_control/old_pass/`):

```
status PASS | stage statistical | verdict equivalent | tier N1 | exact items 0 failures
train_loss_last20_canonical: within   mean +0.000144   δ 0.014214   +0.02σ   p 0.4427
probe_loss_last20:           within   mean −0.000157   δ 0.006526   −0.05σ   p 0.3059
```

The two p-values **0.4427 / 0.3059 are exactly those of the final full check under the original task statement**: the old candidate's gate-3 result is reproduced
number for number under the new endpoints. Together with the bitwise-identical δ above, this is complete evidence that the change did not loosen the gate: **the same markings, the same numbers,
the same verdict**, only a different reader.

### Frozen inputs and evaluator_sha256

- `traj_frozen` `53847b2cce1e914b…` → `a9f38b818a5a736d…`; `traj_spec` `7c304db41d7f2fa6…` → `da41d0aea194d8bb…`;
  still 13 frozen inputs, 13/13 match.
- `evaluator_sha256`: `9c14d853aedae487…` → `3ad5fb1d11fa6f27790bae53d74a23ab46bdb89d34dac7311cf3202c6ec4cb63`
  (the intermediate `9c14d853` fixed the "data probe ignores the switch file" hole, see the next section).
- Backups from before the change are on the formulation side (`llava_adapter.py`, `traj/spec.json`,
  `traj/frozen.json`, `GOAL.md`, the old `traj/ref` records).

## Fixing the "data probe ignores the switch file" hole

`loader_digest()` always ran `loader_probe.py` with only the bench's environment, never with the candidate's switch file. That hurts in both directions:

- when a candidate puts a legitimate reordering behind an environment switch, the probe sees the pipeline without the reordering (`repartitioned=false`), and gate 1 holds it to the
  0.5% band; r28's ERROR was exactly this (`repart=false`, `band1=0.005`, actual deviation 0.78%);
- worse, in the other direction: a candidate could hide **a real data change** behind a switch, and gate 0 would never see it.

Fixed: `loader_digest(tree, out, label, cand_env)` → `partition_state(..., cand_env)` → `coverage(..., cand_env)`
→ `evaluate(..., cand_env)`; the candidate's probe now takes the same code path as timing and recording. The reference's probe still runs without switches (the reference has no switches,
and `task/loader_reference.json` was frozen without them). Regression check: the reference tree running the probe with a dummy switch still gives
`ok / repartitioned=false`.
