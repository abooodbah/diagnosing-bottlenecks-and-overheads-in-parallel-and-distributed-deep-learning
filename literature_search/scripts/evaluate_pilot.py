#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from common import ROOT

YES = {"yes", "y", "1", "true", "relevant", "include", "included"}
NO = {"no", "n", "0", "false", "irrelevant", "exclude", "excluded"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize manually coded pilot precision by source and query.")
    ap.add_argument("input", nargs="?", default=str(ROOT / "templates/query_pilot_sample.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/query_pilot_precision.csv"))
    args = ap.parse_args()
    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stats = defaultdict(lambda: {"coded": 0, "relevant": 0, "irrelevant": 0, "uncoded": 0})
    for r in rows:
        key = (r.get("pilot_source", ""), r.get("pilot_query_id", ""))
        value = (r.get("relevant_title_abstract") or "").strip().lower()
        if value in YES:
            stats[key]["coded"] += 1; stats[key]["relevant"] += 1
        elif value in NO:
            stats[key]["coded"] += 1; stats[key]["irrelevant"] += 1
        else:
            stats[key]["uncoded"] += 1
    output = []
    for (source, query), s in sorted(stats.items()):
        precision = s["relevant"] / s["coded"] if s["coded"] else ""
        output.append({
            "source": source,
            "query_id": query,
            **s,
            "observed_title_abstract_precision": f"{precision:.4f}" if precision != "" else "",
        })
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output[0].keys()) if output else ["source", "query_id", "coded", "relevant", "irrelevant", "uncoded", "observed_title_abstract_precision"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(output)
    print(f"wrote {len(output)} source/query precision rows to {out}")
    if any(s["uncoded"] for s in stats.values()):
        print("Warning: some pilot rows remain uncoded; precision estimates are provisional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
