# Self-Forcing DMD training bench

The criterion and the rules of use are in `../GOAL.md`; this file only describes the files.

| File | Contents |
|---|---|
| `bench.py` | The only entry point: `check` (gate-by-gate verdict), `time` (timing), `profile` (nsys + kernel/NVTX summary), `baseline` (reference against itself) |
| `timing.py` | Timing hook: runs the tree's unmodified `train.py` via runpy and delimits iterations with a global optimizer-step hook; the training code is not patched |
| `sf_adapter.py` | numerics adapter: builds the training objects with the tree's own `ScoreDistillationTrainer`, releases one iteration's batch at a time, and runs the real `train()` loop |
| `bench.json` | Topology (2 GPUs, batch 1 per GPU), timing plan, thresholds, timeouts, invalid-run signatures |
| `cfg/numerics.yaml` | Configuration for the short recordings and the statistical gate; byte-identical to upstream `configs/self_forcing_dmd.yaml` |
| `cfg/timing.yaml` | Configuration for timing: only `ema_start_step` changed to 0, so the timing window covers EMA updates (400 of the 600 upstream steps have EMA) |
| `cases.json` | Short gate-0 cases: a generator iteration, a critic iteration, EMA creation, an EMA update |
| `records/ref`, `thresholds.json` | The reference's short recordings (deterministic mode) and the per-tensor thresholds used for diagnosis only |
| `traj/spec.json`, `traj/ref`, `traj/frozen.json` | Statistical gate: trajectory length, endpoints, the reference's 6-seed trajectories and the frozen δ |

Every command's output carries `cost.gpu_hours`, `inputs` (check of the fixed inputs), `source_digest` (whether the tree changed during the run) and `evaluator_sha256`.
