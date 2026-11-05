#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import ROOT, compile_arxiv, compile_boolean, load_queries


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a source-specific, immutable query register from queries.yaml.")
    ap.add_argument("--queries", default=str(ROOT / "queries.yaml"))
    ap.add_argument("--csv", default=str(ROOT / "results/query_register.csv"))
    ap.add_argument("--md", default=str(ROOT / "results/query_register.md"))
    args = ap.parse_args()

    cfg = load_queries(args.queries)
    rows: list[dict] = []
    for block in cfg["query_blocks"]:
        generic = compile_boolean(block)
        rows.extend([
            {
                "query_id": block["id"],
                "label": block["label"],
                "source": "ACM Digital Library / ACM Guide",
                "date_start": cfg["modern_search_start"],
                "date_end": cfg["cutoff_date"],
                "exact_query_or_rule": generic,
                "execution_mode": "manual title/abstract/keyword-equivalent search; export all results",
            },
            {
                "query_id": block["id"],
                "label": block["label"],
                "source": "IEEE Xplore Metadata API",
                "date_start": cfg["modern_search_start"],
                "date_end": cfg["cutoff_date"],
                "exact_query_or_rule": generic,
                "execution_mode": "querytext with start_year/end_year; page with max_records/start_record",
            },
            {
                "query_id": block["id"],
                "label": block["label"],
                "source": "arXiv API",
                "date_start": cfg["modern_search_start"],
                "date_end": cfg["cutoff_date"],
                "exact_query_or_rule": compile_arxiv(block, cfg["modern_search_start"], cfg["cutoff_date"]),
                "execution_mode": "Atom API; category filter from queries.yaml; page with start/max_results",
            },
            {
                "query_id": block["id"],
                "label": block["label"],
                "source": "OpenReview",
                "date_start": cfg["modern_search_start"],
                "date_end": cfg["cutoff_date"],
                "exact_query_or_rule": generic,
                "execution_mode": "enumerate accepted ICLR/TMLR notes, then apply normalized context-AND-mechanism rule locally",
            },
        ])
        for term in block.get("usenix_terms", []):
            rows.append({
                "query_id": block["id"],
                "label": block["label"],
                "source": "USENIX",
                "date_start": cfg["modern_search_start"],
                "date_end": cfg["cutoff_date"],
                "exact_query_or_rule": term,
                "execution_mode": "site search term; preserve result HTML and paper-page metadata",
            })

    csv_path = Path(args.csv); csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["query_id", "label", "source", "date_start", "date_end", "exact_query_or_rule", "execution_mode"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    md_path = Path(args.md); md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Registered source-specific search queries",
        "",
        f"Protocol version: {cfg.get('version', '')}",
        f"Date window: {cfg['modern_search_start']} through {cfg['cutoff_date']}",
        "",
        "This register is generated directly from `queries.yaml`. Archive both files before running the searches.",
        "",
    ]
    for block in cfg["query_blocks"]:
        lines.extend([f"## {block['id']}: {block['label']}", ""])
        for row in [r for r in rows if r["query_id"] == block["id"]]:
            lines.extend([
                f"### {row['source']}",
                "",
                f"Execution: {row['execution_mode']}",
                "",
                "```text",
                row["exact_query_or_rule"],
                "```",
                "",
            ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(rows)} registered source-query rows to {csv_path}")
    print(f"wrote human-readable register to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
