#!/bin/bash
# Formulation master (runs on the host, one sandbox per stage; stops at the first failing stage).
#   master.sh <stage> [<stage>...]     stages: records variant calibrate timing profile traj_measure traj_rest
#                                              freeze_traj acc_abc acc_de
# Logs: $WORK/llava/logs/<stage>.log.
set -uo pipefail
P=$(cd "$(dirname "$0")" && pwd)
L=${WORK:?set WORK}/llava/logs; mkdir -p $L
SANDBOX=${SANDBOX:?set SANDBOX to the path of goal/sandbox/sandbox.sh}
run() {  # run <log name> <args to formulate.sh/acceptance.sh>
  local name=$1; shift
  echo "### $name start $(date +%T)"
  PREP=1 "$SANDBOX" llava "${GPUS:-0 1 2 3}" bash "$@" > $L/$name.log 2>&1
  local rc=$?
  grep -E "^== |^  ->|^  note|^  traj|IDENTICAL|tensors,|differing|\"status\"|speedup|ms_per_step|ok\":" $L/$name.log | tail -40
  echo "### $name rc=$rc $(date +%T)"
  return $rc
}
F=$P/formulate.sh
A=$P/acceptance.sh
for stage in "$@"; do
  case $stage in
    records)      run g1_records $F records || exit 1 ;;
    variant)      run g2_variant $F variant || exit 1 ;;
    calibrate)    run g3_calibrate $F calibrate || exit 1 ;;
    timing)       run g4_timing $F timing || exit 1 ;;
    profile)      run g5_profile $F profile || exit 1 ;;
    traj_measure) run g6_traj_s0 $F traj 0 || exit 1 ;;
    traj_rest)    run g6_traj_rest $F traj 1 2 3 4 5 || exit 1 ;;
    freeze_traj)  run g7_freeze $F freeze_traj || exit 1 ;;
    acc_abc)      run g8_acc_abc $A a b c || exit 1 ;;
    acc_de)       run g9_acc_de $A d e || exit 1 ;;
    *) echo "unknown stage $stage"; exit 2 ;;
  esac
done
echo "### ALL STAGES DONE $(date +%T)"
