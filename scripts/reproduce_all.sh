#!/usr/bin/env bash
# Regenerate the processed table and all figures from raw/, then verify
# raw-file checksums.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/analyze.py
python - <<'PY'
import hashlib, sys
bad = 0
for line in open("checksums.txt", encoding="utf-8"):
    digest, path = line.strip().split("  ", 1)
    if not path.startswith(("raw/", "transport_logs/")):
        continue
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if h != digest:
        print("MISMATCH", path)
        bad += 1
print("checksum verification:", "FAILED" if bad else "OK")
sys.exit(1 if bad else 0)
PY
