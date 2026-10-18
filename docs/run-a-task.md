# Run a task

Step-by-step instructions for the agent (or person) that prepares a task, starts the optimizing Codex session and supervises it. Model access must already be set up: `goal/codex.env` filled in as described in the [README quick start](../README.md#quick-start). Background: [goal-mode.md](goal-mode.md) (how a run works), [judge.md](judge.md) (the verdict), [methodology.md](methodology.md) (the whole process). For a new task, also give the preparing agent [`goal/templates/FORMULATOR_BRIEF.md`](../goal/templates/FORMULATOR_BRIEF.md).

## 1. Variables

| Variable | Meaning |
|---|---|
| `WORK` | One directory per task: `$WORK/<task>/{ws,home,logs}`. Workspace names used here: `llava`, `dmd`, `rae`, `self-forcing`. |
| `ASSETS` | The task's Python environment, weights and data. Read-only inside the sandbox. Default `$WORK/<task>/assets`. |
| `LOCAL` | Task-private local disk; `$LOCAL/tmp` is the sandbox's `/tmp`. Default `/var/tmp/goal/<task>`. |
| `SANDBOX` | Path of `goal/sandbox/sandbox.sh`, needed by the prep scripts that start sandboxed stages. |
| `HF_ENDPOINT` | Optional Hugging Face mirror for the download scripts (default `https://huggingface.co`). |

The sandbox needs `bwrap` (bubblewrap), `tmux` and the `codex` CLI on the host; `TOOLS` can point at a directory holding node and codex if they are not on the system PATH.

## 2. Set up the task workspace

Take `<task>` from `tasks/` and follow its `GOAL.md` for the upstream commit and the candidate directory name.

```bash
T=$WORK/<task>; mkdir -p $T/ws $T/logs
git clone <upstream> $T/ws/targets/reference && git -C $T/ws/targets/reference checkout <commit>
git clone $T/ws/targets/reference $T/ws/<code> && git -C $T/ws/<code> checkout -b opt
cp -r tasks/<task>/bench tasks/<task>/numerics tasks/<task>/task $T/ws/
cp tasks/<task>/GOAL.md $T/ws/        # and GOAL_LINE.txt: the /goal line typed at launch
cp -r tasks/<task>/formulation/prep $T/prep   # formulation only; moved out before launch (step 4)
```

Build the environment and the assets with the task's prep scripts. Each prep directory has a README listing its scripts and the variables they need; for DMD2, for example:

```bash
export ASSETS=... MINICONDA=...
bash $T/prep/fetch_assets.sh && bash $T/prep/install_env.sh && bash $T/prep/build_fixture.sh
```

Write `$T/sandbox.env` (sourced inside the sandbox before every command): put `$ASSETS/env/bin` first on `PATH` and set `HF_HOME`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so training and judging never touch the network.

Then the agent's home: Codex config and skill cards.

```bash
goal/setup_home.sh <task>                   # all cards in skills/, or name them:
goal/setup_home.sh <task> train-opt-tiered-profiling train-opt-measurement train-opt-numerics-validation
```

Reference records, reference trajectories and frozen input digests are machine-specific and not stored in this repository. Re-create them inside the sandbox with `PREP=1`: record the reference twice and confirm the two recordings are bitwise identical, record the reference trajectories, freeze δ, and rewrite `task/formulation.json` (the prep README names the script for each step: `formulate.sh` and `freeze_inputs.py` for LLaVA, `bench.py record` and `freeze_traj.sh` for DMD2).

## 3. Verify the judge on the reference

Run inside the sandbox, from the workspace (`ws/` is the working directory there). RAE's bench is `bench/run.py` instead of `bench/bench.py`.

```bash
S=goal/sandbox/sandbox.sh
PREP=1 $S <task> "0 1 2 3" python bench/bench.py check --tree targets/reference --out runs/verify/quick --quick
PREP=1 $S <task> "0 1 2 3" python bench/bench.py check --tree targets/reference --out runs/verify/full
```

Expected in `runs/verify/*/check.json`:
- quick: `status` PASS, bitwise (gates 0 and 2);
- full: `status` UNAVAILABLE with verdict `no_benefit` and a measured speedup close to 1.00×. The unmodified reference is not faster than itself, and gate 1 rejects it before any precision gate.

Then run the task's acceptance script (`accept.sh` for DMD2, `acceptance.sh` and `control_arm.sh` for LLaVA, `accept_quick.sh` for Self-Forcing). Every known-bad tree must fail the gate it targets, the control arm (learning rate × 1.02) must be INCORRECT at the statistical gate, and `python -m numerics selftest` must pass. Do not launch until all of them match.

## 4. Launch

Move the formulation material where the sandbox cannot see it, then start the session:

```bash
mv $WORK/<task>/prep $WORK/<task>_formulation
goal/supervise/launch_goal.sh <task> "0 1 2 3"
```

This opens tmux session `<task>_goal`, starts the Codex TUI in the sandbox, records the screen to `$WORK/<task>/logs/tui_<time>.log` and types the line from `ws/GOAL_LINE.txt`. Check that the agent reads `GOAL.md` and starts with a profile.

Supervise from the host:

```bash
goal/supervise/watch.sh <task> "0 1 2 3" &     # exits with one EVENT line when something needs a look
goal/supervise/babysit.sh <task> "0 1 2 3" &   # sends /goal resume when the session idles on "model at capacity"
tmux attach -t <task>_goal                      # look at the TUI
```

After a host restart or a hung TUI, resume the same thread (the goal is bound to it):

```bash
ls -t $WORK/<task>/home/.codex/sessions/*/*/*/rollout-*.jsonl | head -1   # the thread id is the UUID at the end
goal/supervise/relaunch_resume.sh <task> "0 1 2 3" <thread id>
# then type /goal resume in the TUI
```

Supervision rules are in [goal-mode.md](goal-mode.md#supervision-rules): intervene on rules and facts only, never on optimization direction.

## 5. Where results land

| Path | Content |
|---|---|
| `$WORK/<task>/ws/WORK_LOG.md` | One entry per round: profile findings, the change, `time` result, `check` result |
| `$WORK/<task>/ws/runs/<round>/.../check.json` | The verdict: `status`, `correctness`, `performance.speedup`, fingerprints, `cost.gpu_hours` |
| `$WORK/<task>/ws/<code>/` branch `opt` | The candidate, one commit per change; the final switch file is named in the WORK_LOG summary |
| `$WORK/<task>/logs/` | TUI recordings and watchdog output |
| `$WORK/<task>/home/.codex/sessions/` | Codex rollouts; their mtime is the reliable liveness signal |

A result counts only if a full `check` (not `--quick`) says PASS. Before publishing it, run the real workload end to end (see [methodology.md](methodology.md#3-wrap-up-and-real-validation)).
