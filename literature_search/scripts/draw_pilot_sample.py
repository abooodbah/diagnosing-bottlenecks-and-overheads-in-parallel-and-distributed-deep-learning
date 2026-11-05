#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from common import ROOT


def stable_rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}\x1f{key}".encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Draw a deterministic stratified pilot sample for query-precision assessment.")
    ap.add_argument("input", nargs="?", default=str(ROOT / "results/merged/unique_records.csv"))
    ap.add_argument("--out", default=str(ROOT / "templates/query_pilot_sample.csv"))
    ap.add_argument("--per-stratum", type=int, default=20, help="Records per source/query stratum.")
    ap.add_argument("--seed", default="csur-bottleneck-audit-v1")
    args = ap.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        sources = [x for x in (r.get("all_sources") or r.get("source") or "").split(";") if x]
        queries = [x for x in (r.get("all_query_ids") or r.get("query_id") or "").split(";") if x]
        for source in sources or [""]:
            for query in queries or [""]:
                strata[(source, query)].append(r)

    output: list[dict] = []
    seen: set[str] = set()
    for (source, query), candidates in sorted(strata.items()):
        ranked = sorted(
            candidates,
            key=lambda r: stable_rank(args.seed, f"{source}|{query}|{r.get('record_id') or r.get('title')}")
        )
        taken = 0
        for r in ranked:
            rid = r.get("record_id") or r.get("source_record_id") or r.get("title", "")
            key = f"{source}|{query}|{rid}"
            if key in seen:
                continue
            seen.add(key)
            output.append({
                "pilot_source": source,
                "pilot_query_id": query,
                "record_id": rid,
                "title": r.get("title", ""),
                "year": r.get("year", ""),
                "abstract": r.get("abstract", ""),
                "relevant_title_abstract": "",
                "exclusion_reason": "",
                "reviewer_initials": "",
                "notes": "",
            })
            taken += 1
            if taken >= args.per_stratum:
                break

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output[0].keys()) if output else [
        "pilot_source", "pilot_query_id", "record_id", "title", "year", "abstract",
        "relevant_title_abstract", "exclusion_reason", "reviewer_initials", "notes",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(output)
    print(f"wrote {len(output)} deterministic pilot decisions to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
