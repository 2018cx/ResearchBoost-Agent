#!/bin/bash
# Run a formulation command inside the task sandbox with PREP=1 (bench/, task/, numerics/, targets/ writable),
# detached, logging to $WORK/dmd/logs/f_<name>.log.
#   usage: sb.sh <name> <command...>
#   GPUS (default "0 1 2 3": host GPU indices given to the sandbox)
name=$1; shift
T=${WORK:?set WORK}/dmd
SANDBOX=${SANDBOX:?set SANDBOX to the path of goal/sandbox/sandbox.sh}
mkdir -p $T/logs
setsid nohup env PREP=1 "$SANDBOX" dmd "${GPUS:-0 1 2 3}" bash -c "$*; echo SB_EXIT=\$?" \
  > $T/logs/f_$name.log 2>&1 < /dev/null &
echo "started $name: $T/logs/f_$name.log"
