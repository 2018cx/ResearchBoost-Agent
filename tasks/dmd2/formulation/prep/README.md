# DMD2 prep

Scripts for preparing the DMD2 task. Copy this directory to `$WORK/dmd/prep` while preparing the task, and move it out of `$WORK/dmd` before launching the agent.

| Script | What it does | Needs |
|---|---|---|
| `fetch_assets.sh` | SD1.5 weights (pinned revision, sha256-checked), upstream prompts, SD1.5 latent LMDB, the 39k resume checkpoint, Nsight Systems CLI | `ASSETS`, optional `HF_ENDPOINT` |
| `pget.py` | Parallel ranged HTTP download used by `fetch_assets.sh` | |
| `install_env.sh` | Python 3.8 environment with torch 2.4.1+cu124 and upstream's pins; why not upstream's torch is in the header | `ASSETS`, `MINICONDA` |
| `build_fixture.sh` | Unpacks the latent LMDB and writes checkpoint digests | `ASSETS` |
| `sb.sh` | Runs a command inside the sandbox with `PREP=1`, detached, logging to `$WORK/dmd/logs/f_<name>.log` | `WORK`, `SANDBOX`, optional `GPUS` |
| `freeze_traj.sh` | Records the reference trajectories for the frozen seeds and freezes δ = 2 × seed σ | `WORK`, run with `PREP=1` |
| `mkref.py` | Freezes `task/timing_ref.json` from a `bench.py baseline` output | |
| `make_accept.py` | Builds the acceptance trees: copies of the reference with one known change each | |
| `accept.sh` | Acceptance of the judge: numerics self-test plus seven trees with expected verdicts | `WORK`, run with `PREP=1` |

Order: `fetch_assets.sh` → `install_env.sh` → `build_fixture.sh` → reference records (`bench.py record`) → `freeze_traj.sh` → `bench.py baseline` + `mkref.py` → `accept.sh`.
