# Goal mode: one Codex `/goal` session runs a task unattended

One Codex session works from `GOAL.md` until it reaches the target, or until a profile shows the target cannot be reached within the rules. It runs in a sandbox that exposes only the task directory and the assigned GPUs, and its only criterion is the frozen judge's `bench.py check`. A separate supervisor (an agent or a person) formulates the task, launches the session and watches it: it intervenes on rules and facts only, never on optimization direction, and keeps its state in the task's `SUPERVISOR.md`. LLaVA and DMD2 were run this way, one session and 4 GPUs per task. The tools are in [`../goal`](../goal); the step-by-step procedure is in [run-a-task.md](run-a-task.md).

![One goal-mode round: profile → edit → time → full check → log](figures/goal-mode-loop.svg)

## Layout

```
$WORK/<task>/
  ws/                  Codex workspace (trusted project)
    GOAL.md            the task statement
    GOAL_LINE.txt      the /goal line typed at launch
    WORK_LOG.md        appended by the agent every round
    <code>/            upstream code; the agent edits it on its own branch
    bench/ numerics/ task/ targets/   the frozen judge, read-only in the sandbox
    runs/              the agent's outputs (check.json, profiles, timings)
  home/.codex/         config.toml, skills/, sessions/ (rollout jsonl)
  logs/                TUI screen recordings, watchdog logs
  sandbox.env          optional: sourced inside the sandbox before the command (PATH, HF_HOME, offline switches)
$ASSETS                environment, weights, data: read-only in the sandbox
$LOCAL/tmp             the sandbox's /tmp: task-private local disk
$WORK/<task>_formulation/   formulation material (ceiling estimate, acceptance evidence, prep scripts), invisible to the agent
```

## Model access

Fill in `OPENAI_API_KEY` and `OPENAI_BASE_URL` (optionally `CODEX_MODEL`) in `goal/codex.env`; the template is [`../goal/codex.env.example`](../goal/codex.env.example), and the file is not tracked by git. [`setup_home.sh`](../goal/setup_home.sh) uses it together with [`config.toml.example`](../goal/codex/config.toml.example) to write the task's Codex configuration. The sandbox passes these variables in at startup; the key only ever lives in the environment.

## Sandbox ([`../goal/sandbox/sandbox.sh`](../goal/sandbox/sandbox.sh))

Built on bubblewrap:
- only the `/dev/nvidiaN` devices assigned to the task are bound, renumbered 0..N-1 inside;
- the home directory is covered by a tmpfs and only `$WORK/<task>` is exposed (writable); `/var/tmp` is covered too, so other tasks' local data is invisible;
- the judge directories (`ws/bench`, `ws/numerics`, `ws/task`, `ws/targets`) and `$ASSETS` are read-only; `PREP=1` opens them for writing during formulation;
- `/tmp` is replaced by `$LOCAL/tmp`;
- the network is kept by default (`SANDBOX_NET=host`) because Codex has to reach the model endpoint. Training and judging must still run offline: put weights, data and wheels into `$ASSETS` beforehand and set `HF_HUB_OFFLINE=1` in `sandbox.env`. Formulation-only commands can run fully offline with `SANDBOX_NET=none`.

A few pitfalls inside containers:
- some containers cannot mount a fresh `/proc` in bwrap, and entering a new PID namespace with the host `/proc` makes CUDA context creation fail (error 304), so the sandbox does not create a PID namespace;
- without network (`SANDBOX_NET=none`) torchrun cannot use `--standalone`; pin `--master-addr 127.0.0.1` instead;
- in containers without access to performance counters `ncu` does not work; use `nsys` and `torch.profiler`.

## Launch and watchdogs

- [`launch_goal.sh`](../goal/supervise/launch_goal.sh) `<task> "<gpus>"`: starts the sandbox and the TUI in tmux session `<task>_goal`, records the screen to `logs/`, then types the `/goal` line from `ws/GOAL_LINE.txt`.
- [`relaunch_resume.sh`](../goal/supervise/relaunch_resume.sh) `<task> "<gpus>" <thread id>`: after a host restart or a hung TUI, `codex resume <thread id>` reattaches the original session, and `/goal resume` is typed inside it. **Kill any leftover `app-server-daemon` first**, otherwise the new TUI may attach to the stale daemon and hang forever without an error.
- [`watch.sh`](../goal/supervise/watch.sh) `<task> "<gpus>"`: the supervisor's background watchdog. It prints one `EVENT` line and exits when the tmux session disappears, WORK_LOG gains a round, the GPUs sit idle for 25 minutes, a full check passes, or disk runs low; the supervisor then takes a look.
- [`babysit.sh`](../goal/supervise/babysit.sh) `<task> "<gpus>"`: when the model endpoint keeps answering "Selected model is at capacity", the goal stops where it is. Every 5 minutes this script checks whether the bench is idle, the GPUs are idle and the TUI is not Working, and if so sends `/goal resume` once.
- **The only reliable liveness signal is the mtime of the rollout jsonl**; the mtime of `runs/` is not enough.

## Working with the TUI

- To interrupt a stuck turn, send `tmux send-keys -t <s> -H 1b` twice. A plain `Escape` does not interrupt it.
- Type text into the TUI with `tmux load-buffer` plus `tmux paste-buffer -p` (bracketed paste). With send-keys, a leading `/` is swallowed by the command palette.
- `/goal resume` fails in a fresh TUI (the goal is bound to its original thread); run `codex resume <session-id>` first, then `/goal resume`.
- After an interruption, feed back the results it missed: the full numbers, the report paths, and an explicit "do not rerun". Otherwise it reruns a check that takes tens of minutes.
- When the model endpoint drops, an in-flight request may never time out; the TUI keeps showing Working although the background check finished long ago.

## Supervision rules

- Intervene on rules and facts only: gaps in the task statement, factual mistakes (for example the agent reading "must pass numerical validation" as "forbidden by the rules"), an incomplete final audit, a quick result used as evidence. **Never give optimization direction.**
- If a task "unexpectedly" stopped, **search the whole WORK_LOG for a stop note before restarting** (the note is not necessarily at the end), then check whether another session is operating on the same machine. If it is still unclear, ask the owner; do not restart on your own.
- Treat background events only as a trigger to take a look; check every number against the `check.json` on disk.
- To change a shell script that is running, write a new file and `mv` it into place (bash reads scripts by byte offset).
- Ports on shared machines: confirm the port is free before starting; the readiness check must confirm the listener is your own process (`ss -ltnp` shows `pid=`).

## Brief for formulation and supervision

When an agent formulates, launches and supervises a task, give it [`../goal/templates/FORMULATOR_BRIEF.md`](../goal/templates/FORMULATOR_BRIEF.md) as the shared brief, and add the task-specific parts in the prompt (upstream repository and commit, target, assigned GPUs). The brief covers: never showing the agent the answer, the data policy, the environment, the judge logic, the formulation acceptance checklist and the supervision rules.
