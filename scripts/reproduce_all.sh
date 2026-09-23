#!/usr/bin/env bash
# Regenerate every processed table and figure from raw/ and the job logs,
# rerun the provenance replay and the time_ms check, then verify
# checksums.txt. Needs Python 3.10 or newer with matplotlib; set PYTHON to
# choose the interpreter.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:-python}"

for s in launch_windows drift analyze table3 logger_model companion_figures paper_numbers; do
  echo "== $s.py"
  "$PY" "$s.py" > /dev/null
done
echo "== replay_versions.py"
"$PY" replay_versions.py | tail -n 1
echo "== check_time_ms.py"
"$PY" check_time_ms.py | tail -n 1
echo "== checksums"
"$PY" checksums.py --verify
