# Self-Forcing prep

Scripts for preparing the Self-Forcing task; all run inside the sandbox with `PREP=1` from `$WORK/self-forcing/ws`, with this directory at `$WORK/self-forcing/prep`.

| Script | What it does |
|---|---|
| `freeze_inputs.py` | Writes the frozen-input digests into `task/formulation.json` |
| `prep_records.sh` | Reference records for gates 0 and 2, and the diagnostic thresholds |
| `prep_traj.sh`, `prep_traj2.sh` | Reference trajectories for the statistical gate, one process per seed |
| `accept_quick.sh` | Gate-0 acceptance on known trees: unchanged → PASS, `train.py` edited → ERROR, teacher unconditional pass dropped → INCORRECT |

All need `WORK`.
