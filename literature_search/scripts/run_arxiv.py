#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from common import Record, ROOT, compile_arxiv, load_queries, parse_year, stable_id, today_iso, write_records

API = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def text(node: ET.Element | None, default: str = "") -> str:
    if node is None or node.text is None:
        return default
    return " ".join(node.text.split())


def parse_feed(content: bytes) -> tuple[int, list[dict]]:
    """Parse an arXiv Atom response without third-party XML dependencies."""
    root = ET.fromstring(content)
    total_node = root.find("opensearch:totalResults", NS)
    try:
        total = int(text(total_node, "0"))
    except ValueError:
        total = 0
    entries: list[dict] = []
    for entry in root.findall("atom:entry", NS):
        tags = [n.attrib.get("term", "") for n in entry.findall("atom:category", NS)]
        authors = [text(n.find("atom:name", NS)) for n in entry.findall("atom:author", NS)]
        links = [n.attrib for n in entry.findall("atom:link", NS)]
        abs_url = next((l.get("href", "") for l in links if l.get("rel") == "alternate"), "")
        entry_id = text(entry.find("atom:id", NS))
        entries.append({
            "id": entry_id,
            "url": abs_url or entry_id,
            "title": text(entry.find("atom:title", NS)),
            "summary": text(entry.find("atom:summary", NS)),
            "published": text(entry.find("atom:published", NS)),
            "authors": "; ".join(a for a in authors if a),
            "categories": ";".join(t for t in tags if t),
            "doi": text(entry.find("arxiv:doi", NS)),
            "journal_ref": text(entry.find("arxiv:journal_ref", NS)),
        })
    return total, entries


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the registered arXiv query blocks and preserve raw responses.")
    ap.add_argument("--queries", default=str(ROOT / "queries.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/raw/arxiv.csv"))
    ap.add_argument("--raw-dir", default=str(ROOT / "results/raw/arxiv_xml"))
    ap.add_argument("--query-id", action="append", help="Run only the specified query id; repeatable.")
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--max-per-query", type=int, default=0, help="0 means retrieve all results.")
    ap.add_argument("--delay", type=float, default=3.1, help="Pause between repeated arXiv API calls.")
    ap.add_argument("--no-category-filter", action="store_true")
    ap.add_argument("--count-only", action="store_true", help="Request one record per block to obtain reported totals only.")
    args = ap.parse_args()

    cfg = load_queries(args.queries)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.query_id or [])
    records: list[Record] = []
    log_rows: list[dict] = []
    contact = os.environ.get("CONTACT_EMAIL", "")
    ua = "distributed-training-survey-reconstruction/1.0"
    if contact:
        ua += f" ({contact})"
    headers = {"User-Agent": ua}

    for block in cfg["query_blocks"]:
        if selected and block["id"] not in selected:
            continue
        q = compile_arxiv(block, cfg["modern_search_start"], cfg["cutoff_date"])
        if not args.no_category_filter:
            cats = " OR ".join(f"cat:{c}" for c in cfg.get("arxiv_categories", []))
            q = f"({q}) AND ({cats})"
        start = 0
        total: int | None = None
        fetched = 0
        page_size = 1 if args.count_only else args.page_size
        while total is None or start < total:
            if args.count_only and total is not None:
                break
            if args.max_per_query and fetched >= args.max_per_query:
                break
            remaining = args.max_per_query - fetched if args.max_per_query else page_size
            max_results = min(page_size, remaining) if args.max_per_query else page_size
            params = {
                "search_query": q,
                "start": start,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            }
            resp = requests.get(API, params=params, headers=headers, timeout=120)
            resp.raise_for_status()
            raw_path = raw_dir / f"{block['id']}_{start:06d}.xml"
            raw_path.write_bytes(resp.content)
            reported_total, entries = parse_feed(resp.content)
            if total is None:
                total = reported_total
                log_rows.append({
                    "source": "arxiv",
                    "query_id": block["id"],
                    "query": q,
                    "reported_total": total,
                    "retrieved_rows": 0 if args.count_only else "pending",
                    "retrieved_at": today_iso(),
                })
            if args.count_only or not entries:
                break
            for e in entries:
                year = parse_year(e["published"])
                records.append(Record(
                    source="arxiv",
                    query_id=block["id"],
                    source_record_id=e["id"] or stable_id(e["title"], year),
                    title=e["title"],
                    authors=e["authors"],
                    year=year,
                    abstract=e["summary"],
                    doi=e["doi"],
                    url=e["url"],
                    publication_type="preprint",
                    venue=e["journal_ref"] or e["categories"],
                    retrieved_at=today_iso(),
                    raw_file=str(raw_path.relative_to(ROOT)),
                ))
            n = len(entries)
            fetched += n
            start += n
            if n < max_results:
                break
            time.sleep(args.delay)
        if log_rows and log_rows[-1]["query_id"] == block["id"]:
            log_rows[-1]["retrieved_rows"] = fetched

    write_records(args.out, records)
    log_path = Path(args.out).with_name("arxiv_query_log.csv")
    with log_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["source", "query_id", "query", "reported_total", "retrieved_rows", "retrieved_at"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(log_rows)
    print(f"wrote {len(records)} arXiv rows to {args.out}")
    print(f"wrote query log to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
