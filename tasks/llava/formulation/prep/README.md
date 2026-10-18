# LLaVA prep

Scripts for preparing the LLaVA-1.5 task. Copy this directory to `$WORK/llava/prep` while preparing the task, and move it out of `$WORK/llava` before launching the agent.

| Script | What it does | Needs |
|---|---|---|
| `install_env.sh` | Python 3.10 environment with upstream's pins (torch 2.1.2 cu121, transformers 4.37.2, deepspeed 0.12.6, flash-attn 2.5.8 built for sm_90) | `ASSETS`, `MINICONDA` |
| `install_extras.sh` | Optional libraries for candidates (Nsight Systems CLI, apex 24.04.01, liger-kernel) | `ASSETS`, `APEX_SRC` |
| `build_apex.sh` | Rebuilds the apex wheel with its CUDA/C++ extensions | `ASSETS` |
| `check_extras.py` | Checks the optional libraries on one GPU | |
| `download_assets.sh` | Vicuna-7B, CLIP ViT-L/14-336, the stage-1 projector and the instruction JSON, pinned and sha256-checked | `ASSETS`, optional `HF_ENDPOINT` |
| `pdl.py` | Parallel ranged HTTP download | |
| `remotezip.py`, `build_fixture.py` | Fixture: a seeded subset of `llava_v1_5_mix665k.json` and only the images it needs, read out of the public image zips by HTTP range requests | `ASSETS`, optional `HF_ENDPOINT` |
| `freeze_inputs.py` | Writes the frozen-input digests into `task/formulation.json` | `WORK` |
| `formulate.sh` | Inside the sandbox: reference records (twice), a legitimate variant, calibration, reference timing, profile, reference trajectories | `WORK`, run with `PREP=1` |
| `master.sh` | Runs `formulate.sh` / `acceptance.sh` stages, one sandbox per stage, stopping at the first failure | `WORK`, `SANDBOX`, optional `GPUS` |
| `rec.sh`, `cmp_records.py` | Records one case file; compares two record sets bitwise | `WORK` |
| `record_via_bench.py`, `record_traj.py`, `freeze_traj.py` | Re-records reference records and trajectories through `bench.py`'s own code path and freezes δ | `WORK` |
| `make_baseline.py` | Writes `task/baseline.md` from the frozen timing, the noise-floor run and the profile | `WORK` |
| `make_acceptance.py`, `acceptance.sh` | Acceptance trees and their expected verdicts | `WORK`, run with `PREP=1` |
| `control_arm.sh` | Control arm: the reference with the applied learning rate × 1.02 must fail the statistical gate | `WORK`, `ASSETS`, run with `PREP=1` |
| `measure_partition.sh`, `freeze_sensitivity.py` | Measures how much a legal repartition of a step's samples moves the first logged loss, and freezes gate 1's tolerance from it (`task/partition_sensitivity.json`) | `WORK`, `ASSETS` |
| `test_gate0.py` | Gate 0's data rule accepts exactly what `GOAL.md` allows | `WORK` |
| `judge_regression.py` | Replays gate 1's old and new per-step loss guard over every recorded timing run (no GPU) and writes the regression table in `task/judge_changes.md` | `WORK` |
| `verify_old_candidate.sh` | Control for the gate changes: the candidate that passed before them must still pass under the re-frozen δ | `WORK`, `ASSETS`, run with `PREP=1` |

The reasoning behind the gate changes made during the run is in [`../../task/judge_changes.md`](../../task/judge_changes.md).
