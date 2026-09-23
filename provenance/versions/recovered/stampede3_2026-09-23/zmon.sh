#!/usr/bin/env bash
# zmon.sh <jobid> <logfile> <max_checks> [sleep]
JOB=$1; LOG=$2; N=${3:-16}; SL=${4:-30}
for i in $(seq 1 "$N"); do
  ST=$(squeue -j "$JOB" -h -o %t 2>/dev/null)
  if [ -z "$ST" ]; then echo "JOB-DONE after $i checks"; break; fi
  echo "state=$ST check=$i"
  sleep "$SL"
done
echo "===TAIL==="
tail -n 100 "$LOG" 2>/dev/null || echo "no log yet"
