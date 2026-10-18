# goal

Tooling for running one Codex `/goal` session on a task, unattended, inside a sandbox on a shared GPU host. Read [../docs/goal-mode.md](../docs/goal-mode.md) for how a run works and [../docs/run-a-task.md](../docs/run-a-task.md) for the steps.

| File | Role |
|---|---|
| `codex.env.example` | Model access: copy to `codex.env` (git-ignored) and set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, optionally `CODEX_MODEL` |
| `codex/config.toml.example` | Codex configuration template: provider from `OPENAI_BASE_URL`, key from the `OPENAI_API_KEY` variable, approval `never`, sandbox `danger-full-access` (the bubblewrap sandbox is the boundary) |
| `setup_home.sh` | `setup_home.sh <task> [skill ...]`: writes `$WORK/<task>/home/.codex/config.toml` and copies the skill cards |
| `sandbox/sandbox.sh` | `sandbox.sh <task> "<gpus>" [command ...]`: bubblewrap sandbox with only the assigned GPUs, the home hidden except `$WORK/<task>`, judge and `$ASSETS` read-only, a private `/tmp`; host network by default (`SANDBOX_NET=none` for fully offline commands) |
| `supervise/launch_goal.sh` | Start the sandboxed TUI in tmux `<task>_goal`, record the screen, type the `/goal` line from `ws/GOAL_LINE.txt` |
| `supervise/relaunch_resume.sh` | After a restart: kill stale app-server daemons, `codex resume <thread id>`, then type `/goal resume` |
| `supervise/watch.sh` | Supervisor watchdog: exits with one `EVENT` line on a new round, idle GPUs, a full-check PASS, low disk or a lost session |
| `supervise/babysit.sh` | Sends `/goal resume` when the session sits idle after "model at capacity" errors |
| `templates/FORMULATOR_BRIEF.md` | Brief for the agent that formulates, launches and supervises a task: no-answer rule, data policy, environment, judge logic, acceptance checklist, supervision rules |

Variables used by all scripts: `WORK` (one directory per task), `ASSETS` (the task's environment, weights and data; default `$WORK/<task>/assets`), `LOCAL` (task-private local disk; default `/var/tmp/goal/<task>`).
