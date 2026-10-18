#!/bin/bash
# Relaunch the Codex TUI in tmux <task>_goal, resuming the campaign thread (the goal is bound to that thread: a
# fresh TUI cannot `/goal resume` it). App-server daemons left over from a previous TUI can keep the new TUI's
# MCP handshake from ever completing, so the turn hangs with no error; kill them first. Then type `/goal resume`.
#   usage: relaunch_resume.sh <task> "<gpu indices>" <thread id>
set -uo pipefail
TASK=$1; GPUS=$2; ID=$3
SANDBOX=$(cd "$(dirname "$0")/../sandbox" && pwd)/sandbox.sh
W=${WORK:?set WORK}/$TASK; S=${TASK}_goal
L=$W/logs/tui_$(date +%m%d_%H%M%S).log
tmux kill-session -t $S 2>/dev/null
pkill -f "$TASK/home/.codex/packages/app-server-daemon" 2>/dev/null
pkill -f "codex-code-mode-host" 2>/dev/null
sleep 3
tmux new-session -d -s $S -x 220 -y 50 script -qf -c "$SANDBOX $TASK \"$GPUS\" codex resume $ID" "$L"
echo "tui log: $L"
for i in $(seq 1 48); do
  sleep 5
  if tmux capture-pane -t $S -p 2>/dev/null | grep -q "Ask Codex"; then echo "TUI up after $((i*5))s"; break; fi
done
