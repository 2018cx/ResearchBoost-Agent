---
name: train-opt-measurement
description: Pitfalls of timing and comparing training runs: exit 0 is not success, invalid-run signatures (compile fallback, recompilation, attention fallback), fingerprint comparison, cold caches, noise, per-window acceptance, detaching long jobs from the session. Read before timing anything or comparing results.
---

# Timing, power and cost

Use when: measuring the baseline, timing every round, running the full check, and accounting for GPU usage.

**These checks are the job of the fixed evaluation program.** They are built into the bench when the task is formulated, and each one is confirmed during acceptance. During optimization the optimizing agent reads the bench's output to verify them; it does not write its own driver to reimplement them. If the bench lacks a check, report it to the owner to fix.

## Per-GPU load follows the target deployment

The object of optimization is **the per-GPU load of the target deployment**, not the largest load that fits on the local machine. For a target of N GPUs and global batch B, the local per-GPU batch is fixed at B/N, and 1–2 GPUs are enough to optimize on; do not inflate the per-GPU batch to fill a few local GPUs. A different per-GPU batch changes kernel shapes, memory pressure and which ops are bottlenecked, so hotspots and gains found at a large batch do not represent the target deployment.

- Keep the target (`target_world_size`, `global_batch_size`) and the local setup (`per_rank_batch` = B/N, `world_size`) separate in the task statement (`GOAL.md` and `bench.json`). Learning rate and other hyperparameters stay as upstream; they do not change because the local machine has fewer GPUs.
- With fewer local GPUs than the target, gradient synchronization volume and topology differ from the target. Label any communication conclusion "measured on the local topology only".

## Steady-state timing

- Warm up for at least 20 steps and confirm compilation of the relevant paths has finished (the first entry into a new phase also triggers compilation). Then time 35 consecutive steps, repeated over 3 windows, and take the median.
- Turn off the profiler. Synchronize only at window boundaries; all ranks start at the same boundary, and throughput is computed from the slowest rank's window time.
- Run reference and candidate one after another on the same machine, interleaved as reference–candidate–reference or ABBA to cancel drift in machine state.
- Pin a set of GPUs by UUID and use the same set for reference and candidate throughout. Check the GPUs **before and after every run**: no other compute processes, memory use near zero, utilization ≤5%. If that does not hold, do not run, and do not make do. Also record host load, memory pressure, and whether copies use pinned or pageable memory; on a shared machine, treat differences smaller than the host's fluctuation as "not resolved".
- Gains smaller than `max(1%, 2 × window range)` cannot be accepted directly; remeasure them paired first.
- **Accept each window, not just the mean.** Comparing only the means of the two reference runs misses symmetric contamination, where each reference run had one slowed window: the means still agree, the reference is inflated as a whole, the speedup is inflated, and it passes the "1 + 2 × reference gap" threshold. If a run's internal window range (max − min) / median exceeds a threshold (10%, for example), mark the run INVALID and remeasure; it does not enter the comparison. List every window's time and start/end in the report; read the windows before trusting the speedup.
- **Stagger your own jobs too.** During a formal timing run, do not start or stop your own jobs (quick checks, sweeps, interleaved A/B) on other GPUs of the same machine: CPU and memory-bandwidth spikes at startup, compilation and exit land in some window of one arm and cause asymmetric contamination. Start the next job once the formal evaluation's log has entered a non-timing phase (for example the precision recording); a queue script should wait on log markers, not sleep for an estimated duration.
- **Measure small gains differently.** The resolution of a single formal evaluation is set by the reference's window noise; gains below twice the noise (a few percent or less) cannot be resolved by one evaluation, especially on a shared machine. Confirm them first with multiple interleaved A/B rounds (ABBA order, average the paired differences, ideally alternating within one process), or with timing of just the changed stage, and only then send them to a formal evaluation. If the formal evaluation cannot see a small gain, say so; do not substitute the interleaved A/B number for the formal one.

## Valid runs

- **Exit code 0 is not success.** Give every run a unique artifact name. If the timing record, the output or the trace is missing, the run is an ERROR.
- **Signatures of invalid runs.** When formulating a task, list for it the log lines or counters that appear when the wrong path is taken, for example:
  - compile falling back to eager;
  - recompilation inside a timing window;
  - graph capture failing, or graph signature misses;
  - attention falling back to the math backend;
  - a fast path disabling itself for the whole process after an error;
  - a compiled function permanently falling back to eager in the whole process after one call under a dispatch mode (selective recomputation, tracers). This one logs nothing; it can only be detected with "compiled / eager" counters or by whether compiler-generated kernels appear in the trace (see `train-opt-action-ladder`, "Check the existing fast paths first").

  The bench scans every timing run for these signatures, the reference arm included. A hit voids the run: it is not compared, not classified in a profile, and not averaged with valid runs. "Installed but not active" is indistinguishable from "useless" by timing alone; only signatures and counters catch it.

## Run fingerprints and single-variable comparisons

- Every timing record stores a run fingerprint, including:
  - the command line actually executed, with the resolved values of auto options;
  - shapes, dtypes, math modes;
  - topology and GPU UUIDs;
  - instrumentation switches: profiler, syncs, NVTX;
  - whether the compile cache was cold or warm;
  - versions of the key libraries.
- Diff the two fingerprints before comparing. If anything other than the one declared switch differs, refuse the comparison.
- The instrumentation itself is part of the fingerprint: numbers taken with per-stage syncs, a profiler or NVTX cannot be compared with numbers taken without them.
- For presets that follow the upstream configuration (batch, schedule, precision), check at the start of every round that they have not drifted from upstream.

## Report two views separately

1. **Steady state**: no eval, sampling or checkpointing inside the window.
2. **Real events**: these events at their real frequency inside the window (asynchronous checkpointing waits for the background write, and the wait is reported separately), amortized per step over the event period.

A configuration whose schedule is altered so that every step goes through a particular training phase must be labeled "timing configuration", used for timing only, never for quality validation.

## Power and clocks

- In every window, record power, SM clock and whether the power cap is hit via NVML at about 200 ms intervals, and report J/sample.
- At the power cap, the more efficient a path, the lower its clock is pushed, and speed fluctuates with power. Two runs with clearly different clocks cannot be compared directly; remeasure them paired.
- Record the duration and energy of startup and cold compilation separately. They are negligible in real training, but can dominate short jobs such as profiling and tuning.

## Compile cache

- Fix the compile cache directory and record for every run whether the cache was cold or warm.
- Report cold compilation time separately; do not mix it into the steady state.
- Bitwise comparisons only hold under the same cache (see `train-opt-numerics-validation`).
- Each run writes its compile cache, traces and checkpoints into a run directory with an ownership marker, never into an existing directory. Cleanup runs in a `finally` block, so it also happens on interruption, and appends one line to a cleanup log (bytes removed, files left). Directories without an ownership marker and shared caches are never cleaned.

## Waiting for long jobs

After submitting a long job in the background, do one health check to confirm it started and its log is growing, then wait for it to finish; do not poll repeatedly. Polling costs tokens and disturbs the machine.

**Start long jobs detached from the session, and wait inside the session.** Sweeps, formal evaluations and self-test queues longer than a few minutes start with `setsid nohup <command> > <log> 2>&1 &`, and their PID and log path go into the WORK_LOG. Then, if the session exits because of a model API error (for example "model is at capacity") or a timeout, the job is not killed with it, and the next session simply keeps waiting.

**Wait inside the session; do not end it.** Wait with one blocking command, for example `while kill -0 <pid> 2>/dev/null; do sleep 60; done`, or wait for the result file to appear. If the tool has a per-call time limit, wait in segments of 10–20 minutes. Once a session ends, resuming the goal or opening a new session means rereading the task statement, the WORK_LOG and the lessons, which costs several times the tokens of one wait. End the session only when it is close to timing out, after writing a handoff into the WORK_LOG. When debugging, first check whether the artifacts exist, then find the first traceback; noise printed while processes shut down is not the real cause of failure.

## Unchanged workload

Check the workload counters on every run: samples per step, optimizer steps, EMA updates, backward passes, re-forwards per phase, and that the key kernels appear in the trace every step. A changed counter means N5, no matter how much faster the run is.

## GPU usage

- For every GPU allocation, record start, end and number of GPUs; `GPU hours = hold time × number of GPUs`, written into the result's `cost.gpu_hours`.
- Cost estimates must include cold compilation, warmup and failed reruns. Counting steady-state steps only systematically underestimates.
- Aim for about 5 minutes per quick measurement; deep profiling and long validation get their own budgets, but everything counts towards the total.

## Final comparison

Quick measurements and short validations only show that this layer is sound. The final conclusion comes from full training:

- run the reference and the best candidate through the full upstream schedule;
- evaluate the checkpoint of every epoch on a fixed validation set;
- when possible, run the best candidate again with another seed as a noise yardstick.
- For candidates that are not bitwise identical, report two statistics for the quality comparison: the mean, and the worst sample, class or frame. Use at least two metrics, one for magnitude and one for structure or direction, such as PSNR plus SSIM, or relative L2 plus cosine. Also look at fixed sample grids at the beginning, middle and end of training. With only the mean and one magnitude metric, structural damage can hide.
