# RAE Stage-1 bench

This is the frozen judge for RAE Stage-1: `run.py check` applies the verdict flow written in [`../GOAL.md`](../GOAL.md) gate by gate and writes the result to `check.json`, the only criterion. The fingerprints of the frozen inputs are in [`../task/formulation.json`](../task/formulation.json).

## Deployment

Copy this directory to the root of the task workspace as `bench/` (`run.py` takes its parent directory as the workspace root), and mount it read-only once frozen. Also prepare:

- `targets/rae/reference`: a clean checkout of upstream bytetriper/RAE at the fixed commit `a4d18c4`, used as the reference source.
- `../assets/`: weights `models/`, `lpips/vgg.pth`; fixed inputs `fixtures/timing` (the timing set, 60 steps per epoch) and `fixtures/numerics` (the short-case set).
- `bench/cfg/numerics.yaml` and `bench/cfg/timing.yaml`: generated from the upstream configuration, changing only the keys the short cases and timing need (not included in this repository).
- The GPU lock file `/tmp/rae_gpu01.lock`: the bench takes the lock itself in every GPU phase; do not wrap calls in flock.

## Preparation during formulation

Run all commands from the workspace root, with the math mode matching `math_mode` in `bench.json`.

1. **Short recordings** (`cases.json`: one reconstruction step, one adversarial step):
   - record the reference twice in deterministic mode (`--deterministic`) and confirm every tensor is bitwise identical;
   - record the legitimate variants (the adapter's `variant`: `decoder_sdpa`, `encoder_eager`, `disc_sdpa`);
   - record the fp64 gold standard (`--golden`, with `golden_offload:RaeAdapterGolden` moving activations to host memory).
2. **Diagnostic thresholds**:
   - `python -m numerics calibrate --golden ...`;
   - `python -m numerics loo`: the false-rejection rate of the legitimate variants must be 0;
   - `python -m numerics selftest`.
3. **Statistical gate**: in deterministic mode, record the reference's trajectories for 10 seeds according to `traj/spec.json` (`--losses-only --deterministic`), then run `python -m numerics traj-freeze --k 2`. The trajectories must be deterministic: with nondeterministic kernels two runs with the same seed can differ by 5σ, and pairing breaks down.
4. **Acceptance**: walk known candidates through every gate. Each must give the expected result:

   | Candidate | Expected |
   |---|---|
   | Unmodified | `check --skip-timing` PASS at the bitwise gate; the full `check` reports no_benefit (no speedup) |
   | Changed setup in `main()` | ERROR |
   | Discriminator step without re-decoding | INCORRECT (structure) |
   | Mathematically equivalent reordering | passes the statistical gate |
   | LPIPS gradient removed | fails the statistical gate |

## Commands

| Command | Purpose | Used by |
|---|---|---|
| `run.py check --tree <source> --out <dir> [--config <switches>]` | Formal verdict: applies gates 0–3 to a copy of the source and writes `check.json` | optimizing agent |
| `run.py check ... --quick` | Gate-0 self-test only; not a formal score | optimizing agent |
| `run.py sweep --tree <source> --config a.json --config b.json ... --out <dir>` | Screen switch combinations: reference, each configuration once, reference again; all configurations in one command; not a formal score | optimizing agent |
| `run.py time --tree <source> [--config <switches>] --out <dir>` | Phase-weighted timing of one source tree, no comparison | optimizing agent |
| `run.py profile --tree <source> [--config <switches>] [--phase all] --out <dir>` | One nsys window per training phase; each phase has its own log and records, and checks that no compilation happens inside the sampled window and that the discriminator step count is right | optimizing agent |
| `run.py baseline --out <dir>` | Reference against itself: noise floor and bitwise reproducibility | formulator |

## Run-validity checks the bench does itself

Callers read the fields in the output to verify these; no wrapper driver is needed:

- **Idle GPUs**: every GPU run (timing, short recordings, trajectories) checks the visible GPUs after taking the lock and waits while another process is present (memory > 1 GiB or utilization > 5%); if they are still busy after 10 minutes the run is INVALID. They are checked again afterwards, and memory that was not released is recorded as a warning.
- **Timeouts**: `timeouts_s` in `bench.json`; on timeout the whole process group is killed and the run is TIMEOUT.
- **Short temporary directories**: every GPU run uses a fresh short TMPDIR under `/tmp`, deleted afterwards (DataLoader creates its AF_UNIX sockets under TMPDIR, and a deep path fails with `AF_UNIX path too long`).
- **Cold compile caches**: every timing run has its own inductor, triton and CUDA cache directories, deleted afterwards; nothing is shared between configurations or between reference and candidate.
- **Invalid-run signatures**: `invalid_run_signatures` in `bench.json` (compile falling back to eager at the recompile limit, CUDA graphs being skipped); a hit makes the run INVALID; torch.compile compiling a new graph inside a timing window makes it ERROR.
- **Fingerprints and accounting**: every timing records the command line, switches, math mode, GPU UUIDs, cache state and library versions, and every `sweep` row keeps its full fingerprint; a row whose fingerprint differs from reference a in anything other than the switches and command line is INVALID; the reference–candidate–reference timings of `check` compare fingerprints automatically too, and a mismatch is ERROR (not comparable). `sweep`, `check`, `time` and `profile` compute a source digest at the start and at the end; a mismatch at the end (the code rewrote its own directory while running) is ERROR. Every output carries `cost.gpu_hours` (lock hold time × GPUs, including startup, compilation, warmup and failed runs).
- **A failing configuration does not stop a sweep**: each configuration runs in its own process, and a failure is recorded only in its own row; `sweep.json` is rewritten after every configuration, with `complete` saying whether it finished; the reference runs before and after give the noise (`reference.spread`).
- **Input hashes**: every command (`check`, `sweep`, `time`, `profile`, `baseline`) checks, at the start and at the end, the content hashes of all fixed inputs against the values declared in `task/formulation.json`, including weights, the HF and torch hub caches the training code actually loads, fixtures, resume states, thresholds and frozen trajectories. A mismatch at the start means nothing is measured; a mismatch at the end is ERROR. The result is in the output's `inputs` field. These paths are also mounted read-only in the sandbox.
- **Space in /tmp**: with less free space than `min_scratch_free_gib`, the command exits immediately with the reason.
- A candidate configuration may not set variables the bench manages itself (TMPDIR, compile cache directories, CUDA_VISIBLE_DEVICES, PYTHONPATH and so on).

A candidate's switches go in the `environment` mapping of the configuration file passed with `--config`; the bench reads only this file, not the caller's environment variables.
