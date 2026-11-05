#!/usr/bin/env bash
set -euo pipefail

# This script runs the programmatic parts of the retrospective audit.
# ACM Digital Library exports remain a documented manual step.

python scripts/generate_query_register.py
python scripts/run_arxiv.py
python scripts/run_ieee.py
python scripts/run_openreview.py
python scripts/run_usenix.py
python scripts/run_citation_chaining.py
python scripts/merge_deduplicate.py
python scripts/audit_against_gold.py || true
python scripts/make_screening_log.py

echo "Complete manual screening in templates/screening_log.csv, then run:"
echo "  python scripts/count_flow.py"
