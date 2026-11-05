#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from common import ROOT

EXTRA_FIELDS = [
    "title_abstract_decision",
    "title_abstract_exclusion_reason",
    "title_abstract_notes",
    "full_text_retrieved",
    "full_text_decision",
    "full_text_exclusion_reason",
    "full_text_notes",
    "evidence_class",
    "taxonomy_unit_count",
    "taxonomy_bottleneck_type",
    "taxonomy_location",
    "taxonomy_primitive",
    "taxonomy_sync_mode",
    "taxonomy_mitigation",
    "first_author_initials",
    "second_author_audit_required",
    "second_author_decision",
    "resolution_notes",
]


def sampled(record_id: str, fraction: float) -> bool:
    h = int(hashlib.sha1(record_id.encode()).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) < fraction


def main() -> int:
    ap = argparse.ArgumentParser(description="Create the auditable title/abstract and full-text screening log.")
    ap.add_argument("input", nargs="?", default=str(ROOT / "results/merged/unique_records.csv"))
    ap.add_argument("--out", default=str(ROOT / "templates/screening_log.csv"))
    ap.add_argument("--audit-fraction", type=float, default=0.20)
    args = ap.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        base_fields = list(rows[0].keys()) if rows else []
    for r in rows:
        rid = r.get("record_id") or r.get("source_record_id") or r.get("title", "")
        for field in EXTRA_FIELDS:
            r.setdefault(field, "")
        r["second_author_audit_required"] = "random_sample" if sampled(rid, args.audit_fraction) else ""
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base_fields + [x for x in EXTRA_FIELDS if x not in base_fields])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} screening rows to {out}")
    print("Audit policy: second author independently reviews this fixed random sample, all uncertain cases, and all full-text exclusions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
