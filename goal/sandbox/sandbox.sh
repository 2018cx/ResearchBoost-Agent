#!/bin/bash
# Run a command (default: the Codex TUI) for one optimization task inside a bubblewrap sandbox on a shared GPU host.
#   sandbox.sh <task> "<gpu indices>" [command ...]
#   task: a directory under $WORK (e.g. llava, dmd); GPUs: host indices, e.g. "0 1 2 3".
#
# Variables:
#   WORK          directory holding one directory per task: $WORK/<task>/{ws,home,logs}   (required)
#   ASSETS        the task's environment, weights and data, read-only inside   (default $WORK/<task>/assets)
#   LOCAL         task-private local disk; $LOCAL/tmp becomes the sandbox's /tmp   (default /var/tmp/goal/<task>)
#   CODEX_ENV     model access for Codex: OPENAI_API_KEY, OPENAI_BASE_URL, CODEX_MODEL   (default ../codex.env)
#   SANDBOX_NET   host (default): keep the host network, which Codex needs to reach the model endpoint;
#                 none: own network namespace, fully offline (formulation commands only, Codex cannot run)
#   TOOLS         optional directory with node/codex binaries, bound read-only and put first on PATH
#   PREP=1        formulation mode: ws/{bench,numerics,task,targets} stay writable
#   BWRAP         bubblewrap binary (default: bwrap on PATH)
#
# What the sandbox sees:
#   - only the listed GPUs (other /dev/nvidiaN are not bound; inside they are cuda:0..N-1);
#   - the home directory is hidden; only $WORK/<task> (read-write) and $TOOLS (read-only) are visible;
#   - /tmp = $LOCAL/tmp; /var/tmp is hidden, so other tasks' local data is not visible; $ASSETS is read-only;
#   - read-only inside the workspace (unless PREP=1): ws/bench ws/numerics ws/task ws/targets;
#   - HOME=$WORK/<task>/home (Codex config and skills in $HOME/.codex, see setup_home.sh).
# No PID namespace: in some containers bwrap cannot mount a fresh /proc, and with the host /proc in a new PID
# namespace the CUDA driver fails to create a context (cudaErrorOperatingSystem 304).
set -euo pipefail
TASK=$1; GPUS=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
T=${WORK:?set WORK}/$TASK
A=${ASSETS:-$T/assets}
L=${LOCAL:-/var/tmp/goal/$TASK}
ENVF=${CODEX_ENV:-$HERE/../codex.env}
[ -d "$T/ws" ] || { echo "no workspace $T/ws"; exit 1; }
mkdir -p "$L/tmp" "$A" "$T/home"
args=(--die-with-parent --dev-bind / / --dev /dev
      --tmpfs "$HOME" --tmpfs /var/tmp
      --bind "$T" "$T" --ro-bind "$A" "$A" --bind "$L/tmp" /tmp)
case ${SANDBOX_NET:-host} in
  host) ;;
  none) args+=(--unshare-net) ;;
  *) echo "SANDBOX_NET must be host or none"; exit 1 ;;
esac
for d in nvidiactl nvidia-uvm nvidia-uvm-tools nvidia-modeset nvidia-caps; do
  [ -e /dev/$d ] && args+=(--dev-bind /dev/$d /dev/$d)
done
for g in $GPUS; do args+=(--dev-bind /dev/nvidia$g /dev/nvidia$g); done
if [ "${PREP:-0}" != 1 ]; then
  for p in bench numerics task targets; do [ -e "$T/ws/$p" ] && args+=(--ro-bind "$T/ws/$p" "$T/ws/$p"); done
fi
P=/usr/local/bin:/usr/bin:/bin
if [ -n "${TOOLS:-}" ]; then args+=(--ro-bind "$TOOLS" "$TOOLS"); P=$TOOLS/bin:$TOOLS:$P; fi
# model access: exported here and inherited by the sandboxed command (not put on the bwrap command line,
# where any user could read the key in the process list)
if [ -f "$ENVF" ]; then set -a; . "$ENVF"; set +a; fi
[ -f "$T/sandbox.env" ] && args+=(--ro-bind "$T/sandbox.env" "$T/sandbox.env")
[ $# -gt 0 ] || set -- codex
exec "${BWRAP:-bwrap}" "${args[@]}" --chdir "$T/ws" --setenv HOME "$T/home" --setenv TMPDIR /tmp \
  --setenv WORK "$WORK" --setenv ASSETS "$A" --setenv PATH "$P" \
  bash -c '[ -f "$WORK/'"$TASK"'/sandbox.env" ] && . "$WORK/'"$TASK"'/sandbox.env"; exec "$@"' bash "$@"
