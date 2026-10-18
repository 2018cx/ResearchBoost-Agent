#!/bin/bash
# Start the Codex TUI for <task> in tmux session <task>_goal inside the task's sandbox, record the screen to
# $WORK/<task>/logs/tui_<time>.log, and type the goal line from $WORK/<task>/ws/GOAL_LINE.txt.
#   usage: launch_goal.sh <task> "<gpu indices>"
set -uo pipefail
TASK=$1; GPUS=$2
SANDBOX=$(cd "$(dirname "$0")/../sandbox" && pwd)/sandbox.sh
W=${WORK:?set WORK}/$TASK; S=${TASK}_goal
mkdir -p $W/logs; cd $W
L=$W/logs/tui_$(date +%m%d_%H%M%S).log
tmux kill-session -t $S 2>/dev/null
tmux new-session -d -s $S -x 220 -y 50 "script -qf -c '$SANDBOX $TASK \"$GPUS\"' $L"
echo "tui log: $L"
for i in $(seq 1 60); do
  sleep 5
  if grep -qE "Ask Codex|to send|OpenAI Codex|⏎ send" $L 2>/dev/null; then echo "TUI up after $((i*5))s"; break; fi
done
sleep 3
tmux send-keys -t $S "$(tr -d '\n' < $W/ws/GOAL_LINE.txt)"
sleep 2
tmux send-keys -t $S Enter
sleep 20
echo "=== tail of the TUI log"
tail -c 2000 $L | tr -cd '[:print:]\n' | tail -20
echo LAUNCH_DONE
