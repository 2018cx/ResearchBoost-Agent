#!/bin/bash
# Nudge the Codex session with `/goal resume` whenever it sits idle. "Selected model is at capacity" errors
# flap for minutes at a time and leave the goal paused (about 1.5 h over one 24 h run).
# Prints one line per check, ITER checks GAP seconds apart.
# usage: babysit.sh <task> "<gpu indices>"
TASK=$1; GPUS=$2
W=${WORK:?set WORK}/$TASK; S=${TASK}_goal
for i in $(seq 1 ${ITER:-24}); do
  busy=$(pgrep -u $(id -u) -f "$TASK/ws/bench/bench.py" | wc -l)
  gpu=0
  for g in $GPUS; do u=$(nvidia-smi -i $g --query-gpu=utilization.gpu --format=csv,noheader,nounits); gpu=$((gpu+${u:-0})); done
  pane=$(tmux capture-pane -p -t $S | tr -d "\000")
  working=$(echo "$pane" | grep -cE "Working \(|Waiting for background terminal")
  cap=$(echo "$pane" | grep -c "at capacity")
  rounds=$(grep -c "^## " $W/ws/WORK_LOG.md)
  if [ "$busy" -gt 0 ] || [ "$gpu" -gt 20 ] || [ "$working" -gt 0 ]; then
    echo "$(date +%H:%M) ok: bench=$busy gpu=$gpu working=$working rounds=$rounds"
  else
    tmux send-keys -t $S "/goal resume"; sleep 2; tmux send-keys -t $S Enter
    echo "$(date +%H:%M) idle (capacity_banner=$cap rounds=$rounds) -> sent /goal resume"
  fi
  sleep ${GAP:-300}
done
echo BABYSIT_DONE
